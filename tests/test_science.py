"""Scientific invariants plus independently recomputed artifact checks."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from src.features import build_features,chronological_split,wh_to_kwh,SENSORS
from src.baselines import predict_baselines
from src.data.validate_data import validate
from src.data.download_data import CSV
from src.evaluate import comparison,day_block_bootstrap
from src.train import candidates

def synthetic(n=600):
    index=pd.date_range('2020-01-01',periods=n,freq='10min')
    return pd.DataFrame({'Appliances':np.arange(n,dtype=float),**{c:np.full(n,20.) for c in SENSORS}},index=index)

class ConstructionTests(unittest.TestCase):
    def test_exact_future_hour_target_and_no_partial_last_target(self):
        x,y,_,_,_=build_features(synthetic())
        self.assertEqual(y.iloc[0],sum(range(145,151)))
        self.assertEqual(x.index[0],synthetic().index[144])
        self.assertEqual(x.index[-1],synthetic().index[-7])
        self.assertEqual(len(x),600-144-6)

    def test_persistence_and_previous_day_indexing(self):
        _,_,b,_,_=build_features(synthetic())
        self.assertEqual(b.Persistence.iloc[0],sum(range(139,145)))
        self.assertEqual(b['Seasonal naive'].iloc[0],sum(range(1,7)))

    def test_future_perturbation_cannot_change_any_feature(self):
        df=synthetic();cutoff=df.index[350]
        x1,_,b1,_,_=build_features(df)
        changed=df.copy();changed.loc[changed.index>cutoff,:]+=10000
        x2,_,b2,_,_=build_features(changed)
        pd.testing.assert_frame_equal(x1.loc[:cutoff],x2.loc[:cutoff])
        pd.testing.assert_frame_equal(b1.loc[:cutoff],b2.loc[:cutoff])

    def test_prefix_invariance(self):
        df=synthetic();a=build_features(df)[0];b=build_features(df.iloc[:400])[0]
        pd.testing.assert_frame_equal(a.loc[b.index],b)

    def test_rolling_includes_origin_and_is_trailing(self):
        x=build_features(synthetic())[0]
        self.assertEqual(x.energy_sum_6.iloc[0],sum(range(139,145)))
        self.assertEqual(x.energy_mean_144.iloc[0],np.mean(np.arange(1,145)))
        self.assertEqual(x.recent_trend.iloc[0],6)

    def test_inventory_excludes_future_random_and_interpolated_fields(self):
        x,_,_,inventory,_=build_features(synthetic())
        self.assertEqual(len(x.columns),55)
        self.assertTrue(all(i['latest_offset_minutes']<=0 for i in inventory))
        self.assertTrue(all(i['available_at_origin'] for i in inventory))
        self.assertTrue(set(x).isdisjoint({'rv1','rv2','T_out','RH_out','Press_mm_hg','Windspeed','Visibility','Tdewpoint'}))

    def test_unit_conversion(self):
        self.assertEqual(wh_to_kwh(1500),1.5)

    def test_mean_baseline_uses_only_fitting_targets(self):
        b=pd.DataFrame({'Persistence':[10,20],'Seasonal naive':[5,6]})
        p=predict_baselines(pd.Series([1,2,3]),b)
        np.testing.assert_array_equal(p['Training mean'].to_numpy(),[2,2])

    def test_splits_and_cv_have_disjoint_target_intervals(self):
        x=build_features(synthetic())[0];s=chronological_split(x.index)
        for left,right in [('train','validation'),('validation','test')]:
            self.assertLess(x.index[s[left][-1]]+pd.Timedelta(hours=1),x.index[s[right][0]])
        self.assertEqual(len(x)-sum(len(v) for v in s.values()),12)
        for tr,va in TimeSeriesSplit(n_splits=3,gap=6).split(x):
            self.assertLess(x.index[tr[-1]]+pd.Timedelta(hours=1),x.index[va[0]])

    def test_pipelines_train_only_and_seeded_repeatability(self):
        x,y,*_=build_features(synthetic())
        model,_=candidates()['Ridge']
        a=clone(model).fit(x.iloc[:200],y.iloc[:200]);b=clone(model).fit(x.iloc[:200],y.iloc[:200])
        np.testing.assert_allclose(a.predict(x.iloc[200:]),b.predict(x.iloc[200:]),atol=1e-12)
        np.testing.assert_allclose(a.named_steps['standardscaler'].mean_,x.iloc[:200].mean())
        for name,(m,_) in candidates().items():
            self.assertIn('simpleimputer',m.named_steps)
            if name!='Ridge':self.assertEqual(m.steps[-1][1].random_state,335)
        self.assertFalse(candidates()['Gradient boosting'][0].steps[-1][1].early_stopping)

    def test_block_bootstrap_is_paired_and_reproducible(self):
        ix=pd.date_range('2020-01-01',periods=144*5,freq='10min')
        y=np.full(len(ix),100.);m=y+10;b=y+20
        a=day_block_bootstrap(y,m,b,ix,repetitions=30)
        self.assertEqual(a,day_block_bootstrap(y,m,b,ix,repetitions=30))
        np.testing.assert_allclose(a['ci95_pct'],[50,50])

class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df,cls.audit=validate(CSV)
        cls.x,cls.y,cls.baselines,_,_=build_features(cls.df)
        cls.s=chronological_split(cls.x.index)
        cls.pred=pd.read_csv('results/test_predictions.csv',parse_dates=['origin','target_start','target_end']).set_index('origin')
        cls.metrics=json.loads(Path('results/metrics.json').read_text())
        cls.meta=json.loads(Path('results/experiment_metadata.json').read_text())
        cls.web=json.loads(Path('website/public/research/experiment.json').read_text())

    def test_real_dataset_audit(self):
        self.assertEqual(self.audit['rows'],19735)
        self.assertEqual(self.audit['missing_total'],0)
        self.assertEqual(self.audit['irregular_intervals'],0)
        self.assertEqual(self.audit['duplicate_timestamps'],0)

    def test_all_exported_targets_and_baselines_match_raw(self):
        np.testing.assert_allclose(self.pred.actual,self.y.loc[self.pred.index],atol=1e-8)
        for c in ['Persistence','Seasonal naive']:
            np.testing.assert_allclose(self.pred[c],self.baselines.loc[self.pred.index,c],atol=1e-8)
        expected=self.y.iloc[:self.s['test'][0]-6].mean()
        np.testing.assert_allclose(self.pred['Training mean'],expected,atol=1e-7)
        self.assertTrue((self.pred.target_start==self.pred.index+pd.Timedelta(minutes=10)).all())
        self.assertTrue((self.pred.target_end==self.pred.index+pd.Timedelta(hours=1)).all())

    def test_metrics_independently_recomputed_from_csv(self):
        base=np.mean(np.abs(self.pred.actual-self.pred.Persistence))
        for row in self.metrics['test']:
            p=self.pred[row['model']].values;a=self.pred.actual.values
            mae=np.mean(np.abs(a-p));rmse=np.sqrt(np.mean((a-p)**2));r2=1-np.sum((a-p)**2)/np.sum((a-a.mean())**2)
            self.assertAlmostEqual(row['mae'],mae,places=7)
            self.assertAlmostEqual(row['rmse'],rmse,places=7)
            self.assertAlmostEqual(row['r2'],r2,places=7)
            self.assertAlmostEqual(row['improvement_pct'],100*(base-mae)/base,places=7)
            self.assertEqual(row['n'],len(self.pred))
            np.testing.assert_allclose(self.pred[row['model']+' signed_error'],p-a,atol=1e-7)
            np.testing.assert_allclose(self.pred[row['model']+' absolute_error'],np.abs(p-a),atol=1e-7)

    def test_web_predictions_and_metadata_match_evaluated_artifacts(self):
        self.assertEqual(self.web['metadata'],self.meta)
        self.assertEqual(self.web['metrics'],self.metrics)
        self.assertEqual(len(self.web['predictions']),len(self.pred))
        for w,(_,r) in zip(self.web['predictions'],self.pred.iterrows()):
            self.assertEqual(str(r.name),w['origin'])
            for key in ['actual']+[r['model'] for r in self.metrics['test']]:
                self.assertAlmostEqual(w[key],r[key],places=7)

    def test_selected_method_is_validation_winner_and_frozen_before_test(self):
        winner=min(self.metrics['validation'],key=lambda r:r['mae'])['model']
        self.assertEqual(self.meta['selected_model'],winner)
        self.assertEqual(self.meta['selected_ml'],winner)
        self.assertLess(self.meta['selection_frozen_utc'],self.meta['executed_utc'])
        self.assertLess(pd.Timestamp(self.meta['refit_last_origin'])+pd.Timedelta(hours=1),self.pred.index[0])
        self.assertEqual(self.pred.index[0],self.x.index[self.s['test'][0]])
        for fold in self.meta['cv']['folds']:
            self.assertLess(pd.Timestamp(fold['train_target_end']),pd.Timestamp(fold['validation_first_origin']))

    def test_thresholds_training_only(self):
        tr=self.y.iloc[self.s['train']]
        self.assertEqual(self.meta['thresholds_train_only']['low'],tr.quantile(.25))
        self.assertEqual(self.meta['thresholds_train_only']['high'],tr.quantile(.9))

    def test_nonoverlapping_robustness_subset(self):
        models={r['model']:self.pred[r['model']].iloc[::6] for r in self.metrics['test']}
        direct=comparison(self.pred.actual.iloc[::6],models)
        for a,b in zip(direct,self.metrics['nonoverlapping_test']):
            self.assertAlmostEqual(a['mae'],b['mae'],places=7)
            self.assertEqual(a['n'],653)

if __name__=='__main__':unittest.main()
