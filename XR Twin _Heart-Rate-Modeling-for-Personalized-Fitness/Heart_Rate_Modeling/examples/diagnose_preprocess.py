"""
Diagnostic mirror of preprocess.py main(), reporting row counts after each filter.

preprocess.py itself is left untouched so its output stays attributable to the
original code. This script duplicates its steps only to locate where the frame
collapses to zero rows.
"""
import sys, os
import numpy as np
import pandas as pd
import tqdm

tqdm.tqdm.pandas()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, haversine_distances, interpolate

CACHE = "../output/_diag_raw.pkl"


def step(df, label):
    print(f"{label:<52} rows={len(df):>8,}  users={df['userId'].nunique():>6,}", flush=True)
    if len(df) == 0:
        print(">>> COLLAPSED HERE", flush=True)
        sys.exit(1)
    return df


if os.path.exists(CACHE):
    print("loading cached parse...", flush=True)
    df = pd.read_pickle(CACHE)
else:
    df = load_data("../output/endomondoHR_proper.json", ["bike", "run"])
    df.to_pickle(CACHE)
step(df, "loaded (target_activities=['bike','run'])")

df['timestamp_dt'] = df['timestamp'].apply(lambda a: np.array(a, dtype="datetime64[s]"))
df['start_dt'] = df['timestamp_dt'].apply(lambda x: x[0] if len(x) > 0 else None)
df['end_dt'] = df['timestamp_dt'].apply(lambda x: x[-1] if len(x) > 0 else None)
df = df.sort_values(by='start_dt')
df = step(df[~df.duplicated(subset=["userId", "start_dt"], keep='first')], "after duplicate drop")

df['duration'] = df['end_dt'] - df['start_dt']
df = step(df[df['duration'].dt.total_seconds().between(15 * 60, 2 * 60 * 60)], "after duration 15min-2h")

df = df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"])
df = step(df, "after dropna lat/lon/alt/hr")
df = step(df[df["heart_rate"].apply(min) > 45], "after hr min > 45")
df = step(df[df["heart_rate"].apply(max) < 215], "after hr max < 215")

grid_interval = 10
df["time_grid"] = df.progress_apply(
    lambda row: pd.date_range(row["start_dt"] + pd.Timedelta(1, "s"), row["end_dt"], freq=f"{grid_interval}s").values, axis=1)
for c in ["latitude", "longitude", "altitude", "heart_rate"]:
    df[c] = df.progress_apply(lambda row: interpolate(row["timestamp_dt"], row[c], row["time_grid"]), axis=1)
df = df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"])
df = step(df, "after interpolation + dropna")

df["distance"] = df.progress_apply(lambda row: haversine_distances(row["longitude"], row["latitude"]), axis=1)
df["total_distance"] = df["distance"].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
df = step(df[df["total_distance"] >= 1000], "after total_distance >= 1000m")

df["speed_h"] = df.apply(lambda row: np.diff(row["distance"]) / (np.diff(row["time_grid"]).astype(float) / 1e9), axis=1)
df["speed_v"] = df.apply(lambda row: np.diff(row["altitude"]) / (np.diff(row["time_grid"]).astype(float) / 1e9), axis=1)
df["heart_rate"] = df["heart_rate"].apply(lambda x: x[1:])
df["time_grid"] = df["time_grid"].apply(lambda x: x[1:])

df["start_dt"] = df["time_grid"].apply(lambda x: x[0])
df["end_dt"] = df["time_grid"].apply(lambda x: x[-1])

df = step(df[df["speed_h"].apply(max).between(5 / 3.6, 40 / 3.6)], "after speed_h max in 5-40 km/h")
df = step(df[df["speed_v"].apply(lambda x: np.abs(x).max()).between(0, 20 / 3.6)], "after |speed_v| max <= 20 km/h")

df = df.sort_values("start_dt")
workouts_by_user = df.groupby("userId")[["id", "start_dt"]].agg(list)
workouts_by_user["n_workouts"] = workouts_by_user["id"].apply(len)
print(f"\ngroupby -> {len(workouts_by_user):,} users")
print("n_workouts describe:")
print(workouts_by_user["n_workouts"].describe())
print(f"\nusers with 10 <= n_workouts <= 200: {workouts_by_user['n_workouts'].between(10, 200).sum():,}")
print(f"users with n_workouts >= 10:        {(workouts_by_user['n_workouts'] >= 10).sum():,}")
print(f"users with n_workouts > 200:        {(workouts_by_user['n_workouts'] > 200).sum():,}")
