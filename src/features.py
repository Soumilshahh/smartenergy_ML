"""Every X value is a deterministic function of the prefix ending at origin t."""
import numpy as np
import pandas as pd

HORIZON = 6
LAGS = [0,1,2,3,6,12,18,36,72,144]
WINDOWS = [6,18,36,144]
SENSORS = [f'T{i}' for i in range(1,10)] + [f'RH_{i}' for i in range(1,10)]

def build_features(df):
    a = df.Appliances
    x, inventory, groups = pd.DataFrame(index=df.index), [], {'history': [], 'time': [], 'environment': []}
    def add(name, values, group, definition, source, earliest, latest=0):
        x[name] = values
        groups[group].append(name)
        inventory.append({'name': name, 'group': group, 'definition': definition, 'source': source,
                          'earliest_offset_minutes': earliest, 'latest_offset_minutes': latest,
                          'available_at_origin': True})
    for lag in LAGS:
        add(f'energy_lag_{lag}', a.shift(lag), 'history', f'Appliances[t-{lag}]', 'Appliances', -10*lag, -10*lag)
    for w in WINDOWS:
        roll = a.rolling(w, min_periods=w)
        for stat in ['mean','std','min','max','sum']:
            add(f'energy_{stat}_{w}', getattr(roll, stat)(), 'history',
                f'{stat}(Appliances[t-{w-1}:t]) inclusive, std ddof=1', 'Appliances', -10*(w-1))
    add('recent_trend', a-a.shift(6), 'history', 'Appliances[t]-Appliances[t-6]', 'Appliances', -60)
    add('short_long_difference', a.rolling(6).mean()-a.rolling(144).mean(), 'history',
        'trailing mean 1h - trailing mean 24h, both inclusive', 'Appliances', -1430)
    hour = df.index.hour + df.index.minute/60
    dow = df.index.dayofweek
    for name,values,definition in [
        ('hour_sin',np.sin(2*np.pi*hour/24),'sin(2*pi*hour/24)'),
        ('hour_cos',np.cos(2*np.pi*hour/24),'cos(2*pi*hour/24)'),
        ('day_sin',np.sin(2*np.pi*dow/7),'sin(2*pi*weekday/7)'),
        ('day_cos',np.cos(2*np.pi*dow/7),'cos(2*pi*weekday/7)'),
        ('weekend',(dow>=5).astype(int),'weekday >= 5')]:
        add(name,values,'time',definition,'origin calendar',0)
    for c in SENSORS:
        add(c,df[c],'environment',f'{c}[t], household sensor average','household '+c,0)
    # Never sum with skipna: the final six origins must be ineligible.
    y = pd.concat([a.shift(-k) for k in range(1,7)],axis=1).sum(axis=1, min_count=6).rename('actual')
    baselines = pd.DataFrame({'Persistence': a.rolling(6).sum(),
        'Seasonal naive': pd.concat([a.shift(144-k) for k in range(1,7)],axis=1).sum(axis=1,min_count=6)},index=df.index)
    valid = x.notna().all(axis=1) & y.notna() & baselines.notna().all(axis=1)
    return x.loc[valid],y.loc[valid],baselines.loc[valid],inventory,groups

def chronological_split(index):
    n = len(index)
    b1,b2 = int(n*.6),int(n*.8)
    # Six origins are purged before each boundary. This is conservative by one row:
    # last earlier target end < next split origin, so no targets overlap.
    return {'train': np.arange(0,b1-6), 'validation': np.arange(b1,b2-6), 'test': np.arange(b2,n)}

def wh_to_kwh(value):
    return value/1000
