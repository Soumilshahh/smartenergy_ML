"""Small predeclared search: training-only purged CV, then external validation."""
import time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

SEED=335
def candidates():
    return {
        'Ridge': (make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge()),
                  {'ridge__alpha':[1,100,10000]}),
        'Random forest': (make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=160,n_jobs=2,random_state=SEED)),
                          {'randomforestregressor__max_depth':[10,None], 'randomforestregressor__min_samples_leaf':[8,24],
                           'randomforestregressor__max_features':[.7]}),
        'Gradient boosting': (make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(early_stopping=False,random_state=SEED)),
                              {'histgradientboostingregressor__learning_rate':[.05,.1],
                               'histgradientboostingregressor__max_iter':[160],
                               'histgradientboostingregressor__max_leaf_nodes':[7,15],
                               'histgradientboostingregressor__l2_regularization':[10],
                               'histgradientboostingregressor__min_samples_leaf':[30]})}

def tune(x,y):
    tuned,records={},[]
    folds=list(TimeSeriesSplit(n_splits=3,gap=6).split(x))
    for name,(estimator,space) in candidates().items():
        started=time.perf_counter()
        search=GridSearchCV(estimator,space,cv=folds,scoring='neg_mean_absolute_error',n_jobs=1,return_train_score=False)
        search.fit(x,y)
        tuned[name]=search.best_estimator_
        for i,p in enumerate(search.cv_results_['params']):
            records.append({'model':name,'params':p,'cv_mae':float(-search.cv_results_['mean_test_score'][i]),
                'fold_mae':[float(-search.cv_results_[f'split{j}_test_score'][i]) for j in range(3)],
                'selected_within_method':p==search.best_params_, 'method_search_seconds':time.perf_counter()-started})
        print(name,search.best_params_, 'CV MAE',-search.best_score_,flush=True)
    fold_records=[{'train_last_origin':str(x.index[a[-1]]),'train_target_end':str(x.index[a[-1]]+__import__('pandas').Timedelta(hours=1)),
                   'validation_first_origin':str(x.index[b[0]]),'n_train':len(a),'n_validation':len(b)} for a,b in folds]
    return tuned,records,fold_records
