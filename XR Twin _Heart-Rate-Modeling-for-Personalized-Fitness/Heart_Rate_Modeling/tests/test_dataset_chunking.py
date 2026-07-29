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
        "history_allowed": True,
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


def test_history_allowed_column_excludes_rows_from_future_history():
    rows = [row(1, 3), row(2, 3), row(3, 3)]
    rows[0]["heart_rate_normalized"] = np.full(3, 0.1)
    rows[1]["heart_rate_normalized"] = np.full(3, 0.9)
    rows[2]["heart_rate_normalized"] = np.full(3, 0.3)
    rows[1]["history_allowed"] = False
    df = pd.DataFrame(rows)
    config = WorkoutDatasetConfig(
        activity_columns=["speed_h", "speed_v"],
        weather_columns=[],
        history_allowed_column="history_allowed",
        history_max_length=16,
        chunk_size=None,
        stride=None,
    )

    dataset = WorkoutDataset(df, config)
    third = dataset[2]["history"]

    assert np.isclose(third[:, 0].max(), 0.1)
    assert not np.any(np.isclose(third[:, 0], 0.9))
