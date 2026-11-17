"""Fail closed on irregular timestamps instead of constructing ambiguous windows."""
import pandas as pd

def validate(path):
    df = pd.read_csv(path, parse_dates=['date'])
    numeric = df.drop(columns='date')
    summary = {
        'rows': len(df), 'columns': len(df.columns), 'schema': {c: str(t) for c,t in df.dtypes.items()},
        'start': str(df.date.min()), 'end': str(df.date.max()),
        'missing_total': int(df.isna().sum().sum()), 'missing_by_column': df.isna().sum().to_dict(),
        'duplicate_timestamps': int(df.date.duplicated().sum()),
        'irregular_intervals': int(df.date.diff().dropna().ne(pd.Timedelta(minutes=10)).sum()),
        'negative_energy': int((df.Appliances < 0).sum()),
        'invalid_humidity': int(((numeric.filter(regex='^RH')<0)|(numeric.filter(regex='^RH')>100)).sum().sum()),
        'zero_energy_count': int(df.Appliances.eq(0).sum()),
        'zero_energy_fraction': float(df.Appliances.eq(0).mean()),
        'energy_wh': df.Appliances.describe(percentiles=[.01,.5,.9,.95,.99]).to_dict(),
        'feature_distributions': numeric.describe().to_dict(),
        'cleaning': 'No readings removed or winsorized. Fail on missing, unordered or irregular records.'}
    assert df.date.is_monotonic_increasing and not summary['duplicate_timestamps'], 'Nonunique/unordered dates'
    assert summary['irregular_intervals'] == 0, 'Irregular timestamps: refuse row-index windows'
    assert summary['missing_total'] == 0 and summary['negative_energy'] == 0
    assert summary['invalid_humidity'] == 0
    return df.set_index('date'), summary

if __name__ == '__main__':
    import json
    from .download_data import download
    _, audit = validate(download())
    print(json.dumps(audit, indent=2))
