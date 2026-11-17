import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def score(y,p):
    return {'mae': float(mean_absolute_error(y,p)), 'rmse': float(np.sqrt(mean_squared_error(y,p))),
            'r2': float(r2_score(y,p)), 'p95_absolute_error': float(np.quantile(np.abs(y-p),.95)),
            'nmae_test_mean': float(mean_absolute_error(y,p)/np.mean(y)), 'n': len(y)}

def comparison(y, predictions):
    rows=[]
    persistence=mean_absolute_error(y,predictions['Persistence'])
    for name,p in predictions.items():
        rows.append({'model':name,**score(y,p),'improvement_pct':float(100*(persistence-mean_absolute_error(y,p))/persistence)})
    return rows

def day_block_bootstrap(y,model,baseline,index,seed=335,repetitions=2000):
    # Resample complete calendar-day error vectors, keeping paired errors together.
    import pandas as pd
    e=pd.DataFrame({'m':np.abs(np.asarray(y)-model),'b':np.abs(np.asarray(y)-baseline)},index=index)
    blocks=[g[['m','b']].to_numpy() for _,g in e.groupby(e.index.normalize())]
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repetitions):
        draw=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        samples.append(100*(draw[:,1].mean()-draw[:,0].mean())/draw[:,1].mean())
    return {'improvement_pct':float(100*(e.b.mean()-e.m.mean())/e.b.mean()),
            'ci95_pct':np.quantile(samples,[.025,.975]).tolist(),'days':len(blocks),
            'repetitions':repetitions,'seed':seed,
            'caveat':'Day blocks preserve within-day dependence; dependence across days and one-household sampling remain. Partial boundary days retained.'}
