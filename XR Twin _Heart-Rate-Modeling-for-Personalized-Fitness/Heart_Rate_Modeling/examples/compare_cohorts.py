"""
Compare the final cohort under the committed sport filter (["bike","run"]) against
the one the paper describes (["run"]), to test which reproduces the reported
38,323 workouts / 665 users.

Reuses the cached parse from diagnose_preprocess.py. preprocess.py is untouched;
the filter chain below mirrors it exactly apart from the sport selection.
"""
import os
import numpy as np
import pandas as pd
import tqdm

tqdm.tqdm.pandas()
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import haversine_distances, interpolate

CACHE = "../output/_diag_raw.pkl"
PAPER_WORKOUTS, PAPER_USERS = 38323, 665


def build_cohort(df, label):
    df = df.copy()
    df['timestamp_dt'] = df['timestamp'].apply(lambda a: np.array(a, dtype="datetime64[s]"))
    df['start_dt'] = df['timestamp_dt'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['end_dt'] = df['timestamp_dt'].apply(lambda x: x[-1] if len(x) > 0 else None)
    df = df.sort_values(by='start_dt')
    df = df[~df.duplicated(subset=["userId", "start_dt"], keep='first')]

    df['duration'] = df['end_dt'] - df['start_dt']
    df = df[df['duration'].dt.total_seconds().between(15 * 60, 2 * 60 * 60)]
    df = df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"])
    df = df[df["heart_rate"].apply(min) > 45]
    df = df[df["heart_rate"].apply(max) < 215]
    if len(df) == 0:
        return df, None

    df["time_grid"] = df.progress_apply(
        lambda row: pd.date_range(row["start_dt"] + pd.Timedelta(1, "s"), row["end_dt"], freq="10s").values, axis=1)
    for c in ["latitude", "longitude", "altitude", "heart_rate"]:
        df[c] = df.progress_apply(lambda row: interpolate(row["timestamp_dt"], row[c], row["time_grid"]), axis=1)
    df = df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"])

    df["distance"] = df.progress_apply(lambda row: haversine_distances(row["longitude"], row["latitude"]), axis=1)
    df["total_distance"] = df["distance"].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    df = df[df["total_distance"] >= 1000]

    df["speed_h"] = df.apply(lambda r: np.diff(r["distance"]) / (np.diff(r["time_grid"]).astype(float) / 1e9), axis=1)
    df["speed_v"] = df.apply(lambda r: np.diff(r["altitude"]) / (np.diff(r["time_grid"]).astype(float) / 1e9), axis=1)
    df["time_grid"] = df["time_grid"].apply(lambda x: x[1:])
    df["start_dt"] = df["time_grid"].apply(lambda x: x[0])

    df = df[df["speed_h"].apply(max).between(5 / 3.6, 40 / 3.6)]
    df = df[df["speed_v"].apply(lambda x: np.abs(x).max()).between(0, 20 / 3.6)]
    if len(df) == 0:
        return df, None

    by_user = df.groupby("userId")[["id"]].agg(list)
    by_user["n"] = by_user["id"].apply(len)
    valid = by_user[by_user["n"].between(10, 200)]
    df = df[df["userId"].isin(set(valid.index))]
    return df, valid


raw = pd.read_pickle(CACHE)
print(f"cached parse: {len(raw):,} workouts, {raw['userId'].nunique():,} users")
print(f"target: {PAPER_WORKOUTS:,} workouts / {PAPER_USERS:,} users (paper Table 2)\n")

for label, sports in [("as committed: ['bike','run']", ["bike", "run"]), ("as paper says: ['run']", ["run"])]:
    sub = raw[raw["sport"].isin(sports)]
    df, valid = build_cohort(sub, label)
    n_w, n_u = len(df), (df["userId"].nunique() if len(df) else 0)
    print(f"\n{'='*64}\n{label}\n  input:  {len(sub):,} workouts")
    print(f"  FINAL:  {n_w:,} workouts / {n_u:,} users")
    if n_w:
        print(f"  vs paper: workouts {100*n_w/PAPER_WORKOUTS:.1f}%   users {100*n_u/PAPER_USERS:.1f}%")
