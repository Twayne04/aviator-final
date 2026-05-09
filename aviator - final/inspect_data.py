from pathlib import Path
import pandas as pd
base = Path('.')
files = sorted(base.glob('*.csv'))
print('csv count', len(files))
for f in files:
    try:
        df = pd.read_csv(f)
        if 'multiplier' not in df.columns:
            df.columns = ['multiplier'] + list(df.columns[1:])
        df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
        df = df.dropna(subset=['multiplier'])
        print(f.name, len(df), 'high>=10', int((df['multiplier'] >= 10).sum()), 'unique', int(df['multiplier'].nunique()))
    except Exception as e:
        print('ERR', f.name, e)
