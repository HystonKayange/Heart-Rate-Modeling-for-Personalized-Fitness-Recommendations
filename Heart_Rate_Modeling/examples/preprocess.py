import argparse
import json
import os
import torch

import numpy as np
import pandas as pd
import tqdm
from geopy.distance import geodesic

tqdm.tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(input_path: str, target_activities):
    """Load the endomondo data from the json file and return a pandas dataframe filtered by target activities."""
    lines = []
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line.replace("'", '"'))
            if data.get("sport") in target_activities:
                lines.append(data)
    df = pd.DataFrame(lines)
    return df


def haversine_distances(longitudes, latitudes):
    """Compute the distances between consecutive locations from a list of longitudes and latitudes."""
    earth_radius = 6_371_000  # in meters
    phis = np.radians(latitudes)
    delta_phi = np.radians(np.diff(latitudes))
    delta_lambda = np.radians(np.diff(longitudes))

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phis[:-1]) * np.cos(phis[1:]) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = c * earth_radius
    return np.cumsum(np.insert(distances, 0, 0))


def interpolate(time, values, target_grid, make_cumulative=False, remove_offset=False, max_consecutive_nan=5):
    """Interpolate the values to the target grid."""
    if values is None:
        return None

    target_grid = pd.DataFrame(index=target_grid)
    source_df = pd.DataFrame(index=time, columns=["values"], data=values).dropna()
    if make_cumulative:
        source_df["values"] = source_df["values"].cumsum()

    source_df = source_df.sort_index().loc[~source_df.index.duplicated(keep='first')]

    if not isinstance(source_df.index, pd.DatetimeIndex):
        print("Index is not a DatetimeIndex:", type(source_df.index))
        return None

    if not source_df.index.is_monotonic_increasing:
        print("Index is not monotonically increasing even after sorting")
        return None

    target_df = pd.concat([target_grid, source_df], axis=0).sort_index()
    target_df.interpolate("time", inplace=True, limit=max_consecutive_nan, limit_area="inside")
    target_df = target_df[~target_df.index.duplicated(keep="first")]
    target_values = target_df.reindex(target_grid.index)

    if target_values["values"].isnull().any():
        return None

    if remove_offset and len(target_values):
        index = target_values["values"].first_valid_index()
        if index is None:
            return None
        target_values["values"] -= target_values["values"][index]

    return target_values["values"].values.tolist()


def plot_endomondo_workout(workout):
    """Plot a workout from the Endomondo dataset."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].plot(workout["time_grid"], workout["heart_rate"], color="red")
    ax[0].set_ylabel("Heart rate (bpm)")

    ax[1].plot(workout["time_grid"], workout["speed_h"] * 3.6)
    ax[1].set_ylabel("Speed (km/h)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylim(0, 30)
    ax2 = ax[1].twinx()
    ax2.plot(workout["time_grid"], workout["speed_v"], color="green")
    ax2.set_ylabel("Vertical speed (m/s)")
    ax2.set_ylim(-5, 5)

    plt.show()


def calculate_distance(latitudes, longitudes):
    """Calculate the total distance of the workout based on latitude and longitude."""
    coords = list(zip(latitudes, longitudes))
    distances = [geodesic(coords[i], coords[i + 1]).meters for i in range(len(coords) - 1)]
    return sum(distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw Endomondo dataset: `endomondoHR_proper.json`")
    parser.add_argument("--output_path", type=str, default="./", help="Output directory of the processed data: `endomondo_filtered.feather`")
    args = parser.parse_args()

    print(f"Output path: {os.path.join(args.output_path, 'endomondo_filtered.feather')}")

    target_activities = ["bike", "run"]
    df = load_data(os.path.join(args.input_path, "endomondoHR_proper.json"), target_activities)

    # Convert 'timestamp' values to datetime objects and calculate start and end datetimes
    df['timestamp_dt'] = df['timestamp'].apply(lambda a: np.array(a, dtype="datetime64[s]"))
    df['start_dt'] = df['timestamp_dt'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['end_dt'] = df['timestamp_dt'].apply(lambda x: x[-1] if len(x) > 0 else None)

    # Sort DataFrame by 'start_dt' and remove duplicates
    df = df.sort_values(by='start_dt')
    df = df[~df.duplicated(subset=["userId", "start_dt"], keep='first')]

    # Calculate the duration and filter
    df['duration'] = df['end_dt'] - df['start_dt']
    df = df[df['duration'].dt.total_seconds().between(15 * 60, 2 * 60 * 60)]

    # Remove invalid data
    df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"], inplace=True)
    df = df[df["heart_rate"].apply(min) > 45]
    df = df[df["heart_rate"].apply(max) < 215]

    grid_interval = 10
    df["time_grid"] = df.progress_apply(
        lambda row: pd.date_range(row["start_dt"] + pd.Timedelta(1, "s"), row["end_dt"], freq=f"{grid_interval}s").values, axis=1)

    columns_to_interpolate = ["latitude", "longitude", "altitude", "heart_rate"]
    for c in columns_to_interpolate:
        df[c] = df.progress_apply(lambda row: interpolate(row["timestamp_dt"], row[c], row["time_grid"]), axis=1)
    df.dropna(subset=columns_to_interpolate, inplace=True)

    df["distance"] = df.progress_apply(lambda row: haversine_distances(row["longitude"], row["latitude"]), axis=1)
    df["total_distance"] = df["distance"].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    df = df[df["total_distance"] >= 1000]

    df["speed_h"] = df.apply(lambda row: np.diff(row["distance"]) / (np.diff(row["time_grid"]).astype(float) / 1e9), axis=1)
    df["speed_v"] = df.apply(lambda row: np.diff(row["altitude"]) / (np.diff(row["time_grid"]).astype(float) / 1e9), axis=1)
    df["heart_rate"] = df["heart_rate"].apply(lambda x: x[1:])
    df["time_grid"] = df["time_grid"].apply(lambda x: x[1:])

    # Additional Feature Engineering
    
    df['elevation_gain'] = df['altitude'].apply(lambda x: sum(np.diff(x)[np.diff(x) > 0]))
    df['average_speed'] = df['speed_h'].apply(lambda x: np.mean(x))
    df['speed_variability'] = df['speed_h'].apply(lambda x: np.std(x))  # Standard deviation of speed_h
    df['max_heart_rate'] = df['heart_rate'].apply(lambda x: max(x))
    df['hr_recovery'] = df['heart_rate'].apply(lambda x: x[-1] - x[0])  # Difference between last and first heart rate
    df['gender_encoded'] = df['gender'].map({'male': 1, 'female': 0})

    df = df[
        ["time_grid", "heart_rate", "speed_h", "speed_v", "userId", "id", "distance",
         "elevation_gain", "average_speed", "speed_variability", "max_heart_rate", "hr_recovery", "gender_encoded", "sport"]
    ]
    df["start_dt"] = df["time_grid"].apply(lambda x: x[0])
    df["end_dt"] = df["time_grid"].apply(lambda x: x[-1])
    df["time_grid"] = df["time_grid"].apply(lambda x: x.astype(np.int64) / 1e9)
    df["time_grid"] = df["time_grid"].apply(lambda x: (x - x[0]) / (20 * 60))

    df = df[df["speed_h"].apply(max).between(5 / 3.6, 40 / 3.6)]  # 5km/h to 40km/h
    df = df[df["speed_v"].apply(lambda x: np.abs(x).max()).between(0, 20 / 3.6)]  # -20m/s to 20m/s
    df["heart_rate_normalized"] = df["heart_rate"].apply(lambda x: (np.array(x) - 142) / 22)

    df = df.sort_values("start_dt")
    workouts_by_user = df.groupby("userId")[["id", "start_dt"]].agg(list)
    workouts_by_user["n_workouts"] = workouts_by_user["id"].apply(len)
    workouts_by_user = workouts_by_user[workouts_by_user["n_workouts"].between(10, 200)]
    valid_users = set(workouts_by_user.index)
    df = df[df["userId"].isin(valid_users)]

    # Define train-test split
    test_proportion = 0.2
    workout_train = workouts_by_user["id"].apply(lambda x: x[: int(len(x) * (1 - test_proportion))])
    train_ids = set(np.concatenate(workout_train.values))
    df["in_train"] = df["id"].isin(train_ids)

    # Assign unique index to each subject
    df["subject_idx"] = df["userId"].astype("category").cat.codes

    # Check DataFrame before saving
    print(f"DataFrame shape: {df.shape}")
    if df.empty:
        print("DataFrame is empty after filtering.")
    else:
        print(df.head())

    # Save the data as a feather file
    try:
        df.reset_index(drop=True).to_feather(os.path.join(args.output_path, "endomondo_filtered.feather"))
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
