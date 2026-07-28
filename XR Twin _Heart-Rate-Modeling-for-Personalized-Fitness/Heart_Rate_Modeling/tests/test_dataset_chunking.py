import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.data import WorkoutDataset, WorkoutDatasetConfig


def row(workout_id, length):
    return {
        "subject_id": 1,
        "workout_id": workout_id,
        "time_grid": np.arange(length, dtype=float),
        "time_start": pd.Timestamp("2024-01-01") + pd.Timedelta(days=workout_id),
        "heart_rate": np.full(length, 140.0),
        "heart_rate_normalized": np.zeros(length),
        "speed_h": np.ones(length),
        "speed_v": np.zeros(length),
    }


def test_chunked_dataset_skips_short_workouts_and_keeps_exact_length():
    df = pd.DataFrame([row(1, 64), row(2, 128), row(3, 129)])
    config = WorkoutDatasetConfig(
        activity_columns=["speed_h", "speed_v"],
        weather_columns=[],
        chunk_size=128,
        stride=64,
    )

    dataset = WorkoutDataset(df, config)

    assert len(dataset) == 3
    assert [len(dataset[i]["heart_rate"]) for i in range(len(dataset))] == [128, 128, 128]
    assert dataset.workout_ids.tolist() == [2, 3, 3]
