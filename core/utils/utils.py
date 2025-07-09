# Returns values from a 1D array/list `arr` at the multi-dimensional indices `idx`.
# This is used to retrieve tokens or tones for each row of indices in `define_sentiment_charged_tokens`.
import numpy as np
import pandas as pd


def access_via_nd_index(arr, idx):
    arr = np.array(arr)
    return arr[idx]


# Filters DataFrame `df` to only include rows whose datetime is between start_date (inclusive) and end_date (exclusive).
# Used to select news articles in a given weekly range (aligns with the expanding train/test splits:contentReference[oaicite:0]{index=0}).
def between_dates(df, start_date, end_date):
    # If DataFrame has a 'datetime' column use it, otherwise use the index
    if 'datetime' in df.columns:
        times = df['datetime']
    else:
        times = df.index
    mask = (times >= start_date) & (times < end_date)
    return df.loc[mask]


# Transforms an array `y` into relative ranks in [0,1].
# Equivalent to (rank - 1) / (n-1) using average rank for ties.
# Used to convert true return signs or values into a normalized rank form for evaluation.
def transform_to_rank(y):
    s = pd.Series(y)
    ranks = s.rank(method='average')
    # Scale to [0,1]
    return ((ranks - 1) / (len(s) - 1)).to_numpy()


# Creates expanding time-series train/validation splits by year.
# Expanding cross-validation: initial training window of `train_size` years, then each split adds `split_size`.
# This matches the paper's expanding-window scheme where each fold adds one year:contentReference[oaicite:1]{index=1}.
def time_series_split(df, datetime_col='datetime', train_size=2, val_size=1, split_size='1y', expanding=True):
    # Determine start and end of data
    times = pd.to_datetime(df[datetime_col])
    start = times.min()
    end = times.max()
    # Parse split increment (e.g., '1y' -> 1 year)
    import re
    num = int(re.match(r'(\d+)', split_size).group(1))
    unit = split_size[-1]
    if unit == 'y':
        inc = pd.DateOffset(years=num)
    elif unit == 'm':
        inc = pd.DateOffset(months=num)
    else:
        inc = pd.to_timedelta(split_size)
    train_offset = pd.DateOffset(years=train_size)
    val_offset = pd.DateOffset(years=val_size)
    splits = []
    # Initial train and validation bounds
    train_start = start
    train_end = start + train_offset
    # Loop until validation end goes beyond available data
    while train_end + val_offset <= end:
        val_start = train_end
        val_end = train_end + val_offset
        train_df = df[(times >= train_start) & (times < train_end)]
        val_df = df[(times >= val_start) & (times < val_end)]
        splits.append((train_df, val_df))
        if expanding:
            # Expand training window for next split
            train_end += inc
        else:
            # Slide both windows
            train_start += inc
            train_end = train_start + train_offset
    return splits


# Ensures DataFrame `df` has a 'date' column (year-month-day) corresponding to 'datetime' timestamps.
# If missing, creates 'date' by dropping the time component.
# In STTM data, each message has a datetime of release:contentReference[oaicite:2]{index=2}.
def add_date_if_not_present(df):
    if 'date' not in df.columns:
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date


# Converts datetime to date (dropping the time part).
# Two variants:
#  - Option 1 (preserve time): simply return datetime as-is.
def to_date_preserve(dt):
    return pd.to_datetime(dt)


#  - Option 2 (drop time): normalize datetime to midnight (effectively keeping only date).
def to_date(dt):
    return pd.to_datetime(dt).dt.normalize()
