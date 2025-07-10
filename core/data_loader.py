import pandas as pd
from pathlib import Path

def load_news(path="data/news"):
    files = Path(path).glob("*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['preproc'])
    df['preproc'] = df['preproc'].apply(lambda x: x.split())
    return df.rename(columns={'date': 'issuedate'})

def load_returns(path="data/time_series"):
    returns = {}
    for f in Path(path).glob("*.csv"):
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.upper()
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df = df.sort_values('TRADEDATE')
        df['RET'] = df['CLOSE'].pct_change().shift(-1)
        secid = df['SECID'].iloc[0]
        ret_series = df.set_index('TRADEDATE')['RET'].dropna()
        returns[secid] = ret_series
    return returns
