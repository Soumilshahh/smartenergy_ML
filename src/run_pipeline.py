"""Run once from the repository root: python -m src.run_pipeline.

All candidate choices are frozen before any test prediction is evaluated.
An explicit --overwrite flag is required to replace an existing final evaluation.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
from sklearn.base import clone
from .data.download_data import download,CSV
from .data.validate_data import validate
from .features import build_features,chronological_split
from .train import tune,SEED
from .baselines import predict_baselines
from .evaluate import comparison,day_block_bootstrap
from .analysis import eda,ablate,importance,errors
from .export_results import write_json,export_web

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--overwrite',action='store_true',help='Reproduce the predeclared experiment, replacing existing artifacts')
    args=parser.parse_args()
    if Path('results/metrics.json').exists() and not args.overwrite:
        raise SystemExit('Final evaluation exists. Use --overwrite only to reproduce the frozen protocol, never to tune on test.')
    start=time.perf_counter()
    download()
    df,audit=validate(CSV)
    x,y,b,inventory,groups=build_features(df)
    splits=chronological_split(x.index)
    tr,va,te=[splits[k] for k in ['train','validation','test']]
    for left,right in [(tr,va),(va,te)]:
        assert x.index[left[-1]]+pd.Timedelta(hours=1)<x.index[right[0]]
    print('Valid samples:',len(x),'Splits:',{k:len(v) for k,v in splits.items()},flush=True)
    results=Path('results');results.mkdir(exist_ok=True)
    write_json(results/'data_summary.json',audit)
    write_json(results/'feature_inventory.json',inventory)
    models,search,folds=tune(x.iloc[tr],y.iloc[tr])
    val_preds=predict_baselines(y.iloc[tr],b.iloc[va])
    for name,model in models.items(): val_preds[name]=model.predict(x.iloc[va])
    val_scores=comparison(y.iloc[va],val_preds)
    selected_ml=min(models,key=lambda n:next(r['mae'] for r in val_scores if r['model']==n))
    selected_model=min(val_scores,key=lambda r:r['mae'])['model']
    print('Validation selected:',selected_model,'ML candidate:',selected_ml,flush=True)
    ablation=ablate(models[selected_ml],x,y,tr,va,groups)
    imp=importance(models[selected_ml],x.iloc[va],y.iloc[va])
    threshold={'low':float(y.iloc[tr].quantile(.25)),'high':float(y.iloc[tr].quantile(.9)),
               'sudden_change':float(np.quantile(np.abs(y.iloc[tr]-b.iloc[tr].Persistence),.9))}
    bounds={}
    for k,idx in splits.items():
        bounds[k]={'n':len(idx),'origin_start':str(x.index[idx[0]]),'origin_end':str(x.index[idx[-1]]),
                   'target_start':str(x.index[idx[0]]+pd.Timedelta(minutes=10)),
                   'target_end':str(x.index[idx[-1]]+pd.Timedelta(hours=1))}
    frozen={'selected_model':selected_model,'selected_ml':selected_ml,'feature_protocol':'All history + time + household sensors (predeclared primary)',
            'best_parameters':{n:next(r['params'] for r in search if r['model']==n and r['selected_within_method']) for n in models},
            'split':bounds,'selection_metric':'validation MAE, smallest wins; no test-based changes',
            'selection_frozen_utc':datetime.now(timezone.utc).isoformat()}
    write_json(results/'selection_frozen.json',frozen)
    write_json(results/'hyperparameter_search.json',search)
    write_json(results/'ablation.json',ablation)
    write_json(results/'feature_importance.json',imp)
    # Refit on all eligible pre-test origins, including the formerly purged train/val seam.
    # Keep six origins purged before test, so every training label ends before test begins.
    fit=np.arange(0,te[0]-6)
    test_predictions=predict_baselines(y.iloc[fit],b.iloc[te])
    for name,model in models.items():
        refit=clone(model).fit(x.iloc[fit],y.iloc[fit])
        test_predictions[name]=refit.predict(x.iloc[te])
    test_scores=comparison(y.iloc[te],test_predictions)
    bootstrap=day_block_bootstrap(y.iloc[te],test_predictions[selected_ml].values,test_predictions.Persistence.values,x.index[te])
    nonoverlap=comparison(y.iloc[te[::6]],test_predictions.iloc[::6])
    error_groups=errors(y.iloc[te],test_predictions[selected_ml],x.index[te],threshold)
    sudden=np.abs(y.iloc[te].values-b.iloc[te].Persistence.values)>=threshold['sudden_change']
    from .evaluate import score
    error_groups.append({'group':'Sudden change',**score(y.iloc[te].values[sudden],test_predictions[selected_ml].values[sudden])})
    pred=test_predictions.copy()
    pred.insert(0,'actual',y.iloc[te])
    pred.insert(0,'target_end',pred.index+pd.Timedelta(hours=1))
    pred.insert(0,'target_start',pred.index+pd.Timedelta(minutes=10))
    for name in test_predictions:
        pred[name+' signed_error']=test_predictions[name]-y.iloc[te]
        pred[name+' absolute_error']=np.abs(test_predictions[name]-y.iloc[te])
    pred.index.name='origin'
    pred.to_csv(results/'test_predictions.csv',float_format='%.12g')
    pd.DataFrame(test_scores).to_csv(results/'model_comparison.csv',index=False,float_format='%.12g')
    eda_data=eda(df,y,results/'figures')
    meta={**frozen,'dataset':{'name':'UCI Appliances Energy Prediction','url':'https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction',
                             'doi':'10.24432/C5VC8G','license':'CC BY 4.0','sha256':hashlib.sha256(CSV.read_bytes()).hexdigest(),
                             'provenance':json.loads((CSV.parent/'provenance.json').read_text())},
          'seed':SEED,'executed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
          'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
          'hardware':{'platform':platform.platform(),'processor':platform.processor(),'cpu_only':True,'forest_threads':2},
          'cv':{'folds':folds,'gap_origins':6,'n_splits':3},'refit_samples':len(fit),'refit_last_origin':str(x.index[fit[-1]]),
          'target_definition':'sum Appliances(t+1) through Appliances(t+6), Wh; row t observed at interval end',
          'time_zone':'Naive dataset clock: source does not supply UTC offsets. Do not convert to visitor timezone.',
          'excluded':['rv1','rv2','lights','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint'],
          'sensor_assumption':'Household ten-minute sensor averages are assumed available at end of interval t. Airport interpolated weather excluded because it may use future hourly endpoints.',
          'bootstrap':bootstrap,'thresholds_train_only':threshold,'eligible_origins':len(x),'purged_origins':12,
          'integrity':'Model and hyperparameters frozen in selection_frozen.json before the single final-test scoring stage. All frozen candidates refitted and reported for transparency.'}
    metrics={'validation':val_scores,'test':test_scores,'nonoverlapping_test':nonoverlap,'bootstrap':bootstrap}
    write_json(results/'metrics.json',metrics)
    write_json(results/'experiment_metadata.json',meta)
    write_json(results/'error_analysis.json',{'groups':error_groups,'thresholds':threshold})
    write_json(results/'eda.json',eda_data)
    compact=[]
    for t,row in pred.iterrows():
        compact.append({'origin':str(t),'target_start':str(row.target_start),'target_end':str(row.target_end),
                        'actual':float(row.actual),**{name:float(row[name]) for name in test_predictions}})
    worst=(test_predictions[selected_ml]-y.iloc[te]).abs().nlargest(5).index
    bundle={'metadata':meta,'audit':audit,'metrics':metrics,'predictions':compact,'ablation':ablation,'importance':imp,
            'errors':error_groups,'eda':eda_data,'worst_origins':[str(t) for t in worst],'features':inventory}
    export_web(results,bundle)
    print(pd.DataFrame(test_scores).to_string(index=False),flush=True)
    print('Exported real research artifacts in',round(time.perf_counter()-start,1),'seconds',flush=True)

if __name__=='__main__': main()
