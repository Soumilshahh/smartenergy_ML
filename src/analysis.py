"""Descriptive EDA is separate from held-out selection and forecasting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from .evaluate import score

def eda(df,y,out):
    Path(out).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                          'axes.labelcolor':'#24354b','axes.titleweight':'bold','figure.dpi':140})
    figures=[]
    def save(name,title,xlabel,ylabel,takeaway):
        plt.title(title);plt.xlabel(xlabel);plt.ylabel(ylabel)
        plt.figtext(.5,.01,'Source: UCI Appliances Energy Prediction (one Belgian household, 2016)',ha='center',fontsize=7)
        plt.tight_layout(rect=[0,.035,1,1]);plt.savefig(Path(out)/f'{name}.png');plt.close()
        figures.append({'file':name+'.png','title':title,'takeaway':takeaway})
    daily=df.Appliances.resample('D').sum()/1000
    plt.figure(figsize=(10,3.8));plt.plot(daily.index,daily,color='#087f8c');
    save('daily_energy','Daily appliance energy (boundary days are partial)','Dataset date','Energy (kWh/day)',f'Daily totals range from {daily.min():.2f} to {daily.max():.2f} kWh, including partial boundary days.')
    plt.figure(figsize=(8,3.8));plt.hist(df.Appliances,bins=50,color='#087f8c');
    save('distribution','Ten-minute appliance energy distribution','Energy per ten-minute interval (Wh)','Observations',f'Median {df.Appliances.median():.0f} Wh; 99th percentile {df.Appliances.quantile(.99):.0f} Wh. Large events are retained.')
    hour=df.groupby(df.index.hour).Appliances.mean()
    plt.figure(figsize=(8,3.8));plt.bar(hour.index,hour,color='#087f8c');
    save('hourly','Typical energy by clock hour','Dataset clock hour','Mean ten-minute energy (Wh)',f'The largest full-dataset hourly mean occurs at {int(hour.idxmax()):02d}:00; this is descriptive, not a causal explanation.')
    week=df.groupby(df.index.dayofweek>=5).Appliances.mean()
    plt.figure(figsize=(7,3.8));plt.bar(['Weekday','Weekend'],week.values,color=['#087f8c','#be7511']);
    save('weekend','Weekday and weekend consumption','Calendar group','Mean ten-minute energy (Wh)',f'Weekday mean {week.iloc[0]:.1f} Wh; weekend mean {week.iloc[1]:.1f} Wh in this limited record.')
    corr=df[['Appliances','T1','T6','T9','RH_1','RH_6']].corr()
    plt.figure(figsize=(7,5));plt.imshow(corr,vmin=-1,vmax=1,cmap='BrBG');plt.colorbar(label='Pearson correlation');plt.xticks(range(len(corr)),corr.columns);plt.yticks(range(len(corr)),corr.columns)
    save('environment','Contemporaneous sensor correlations','Variable','Variable','Correlation is descriptive and does not establish causation or future predictive benefit.')
    lags=[1,6,18,36,72,144,288]
    ac=[float(y.autocorr(k)) for k in lags]
    plt.figure(figsize=(8,3.8));plt.bar([str(k*10) for k in lags],ac,color='#087f8c');
    save('autocorrelation','Future-hour target dependence','Origin separation (minutes)','Target autocorrelation',f'Adjacent targets have correlation {ac[0]:.3f}; overlapping windows are not independent samples.')
    counts,edges=np.histogram(df.Appliances,bins=25)
    return {'daily':[{'date':str(t.date()),'kwh':float(v)} for t,v in daily.items()],
            'hourly':[{'hour':int(h),'wh':float(v)} for h,v in hour.items()],
            'distribution':[{'wh':float((edges[i]+edges[i+1])/2),'count':int(c)} for i,c in enumerate(counts)],
            'weekday_weekend':{'weekday':float(week.iloc[0]),'weekend':float(week.iloc[1])},
            'autocorrelation':[{'lag_minutes':k*10,'correlation':v} for k,v in zip(lags,ac)],
            'figures':figures}

def ablate(model,x,y,train,val,groups):
    rows=[]
    for label,columns in [('History',groups['history']),('History + time',groups['history']+groups['time']),
                           ('History + time + sensors',sum(groups.values(),[]))]:
        m=clone(model).fit(x.iloc[train][columns],y.iloc[train])
        p=m.predict(x.iloc[val][columns])
        rows.append({'group':label,'features':len(columns),**score(y.iloc[val],p)})
    return rows

def importance(model,x,y):
    result=permutation_importance(model,x,y,n_repeats=3,random_state=335,scoring='neg_mean_absolute_error',n_jobs=1)
    return sorted([{'feature':c,'mae_increase':float(v),'repeat_std':float(s)} for c,v,s in zip(x.columns,result.importances_mean,result.importances_std)],key=lambda r:r['mae_increase'],reverse=True)

def errors(y,p,index,threshold):
    a=np.asarray(y);p=np.asarray(p)
    masks={'Low demand':a<=threshold['low'],'Medium demand':(a>threshold['low'])&(a<threshold['high']),
           'High demand':a>=threshold['high'],'Daytime':(index.hour>=7)&(index.hour<22),
           'Nighttime':(index.hour<7)|(index.hour>=22),'Weekday':index.dayofweek<5,'Weekend':index.dayofweek>=5}
    return [{'group':g,**score(a[m],p[m])} for g,m in masks.items() if np.any(m)]
