import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.activity_features import (
    RUN_PERSONAL_COLUMNS,
    add_activity_features,
    chronological_train_val_masks,
)


def test_basic_feature_set_preserves_activity_columns():
    df = pd.DataFrame({"speed_h": [[1.0, 2.0]], "speed_v": [[0.0, 0.1]]})

    out, columns = add_activity_features(df, "basic")

    assert out is df
    assert columns == ["speed_h", "speed_v"]


def test_run_intensity_features_are_per_step_and_bounded():
    df = pd.DataFrame(
        {
            "speed_h": [np.array([1.0, 2.5, 3.0])],
            "speed_v": [np.array([0.0, 0.5, -0.5])],
            "time_grid": [np.array([0.0, 0.05, 0.2])],
        }
    )

    out, columns = add_activity_features(df, "run_intensity")

    assert columns == [
        "speed_h",
        "speed_v",
        "speed_abs",
        "speed_delta",
        "abs_speed_delta",
        "low_speed_fraction",
        "warmup_fraction",
        "incline_proxy",
    ]
    for column in columns:
        assert len(out.iloc[0][column]) == 3
    assert np.allclose(out.iloc[0]["speed_delta"], [0.0, 1.5, 0.5])
    assert np.allclose(out.iloc[0]["warmup_fraction"], [1.0, 0.5, 0.0])
    assert out.iloc[0]["low_speed_fraction"][0] > 0
    assert out.iloc[0]["low_speed_fraction"][2] == 0
    assert np.all(np.abs(out.iloc[0]["incline_proxy"]) <= 1.0)


def _personal_df():
    """Two users; user 1 has three chronological runs with different paces/HRs."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "id": [10, 11, 12, 20],
            "sport": ["run", "run", "run", "run"],
            "start_dt": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-01"]
            ),
            "time_grid": [
                np.array([0.0, 0.1]),
                np.array([0.0, 0.1]),
                np.array([0.0, 0.1]),
                np.array([0.0, 0.1]),
            ],
            "heart_rate": [
                np.array([100.0, 110.0]),
                np.array([150.0, 160.0]),
                np.array([120.0, 130.0]),
                np.array([140.0, 145.0]),
            ],
            "speed_h": [
                np.array([2.0, 2.0]),
                np.array([4.0, 4.0]),
                np.array([3.0, 3.0]),
                np.array([3.0, 3.0]),
            ],
            "speed_v": [
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
            ],
            "in_train": [True, True, False, True],
        }
    )


def test_run_personal_columns_and_relative_speed():
    df = _personal_df()
    # Population from train rows only.
    population_mask = df["in_train"]
    out, columns = add_activity_features(
        df,
        "run_personal",
        population_mask=population_mask,
        sport="run",
    )

    assert columns == RUN_PERSONAL_COLUMNS
    # First workout for user 1: cold start → prior_missing=1, population fallback.
    assert out.iloc[0]["prior_missing"][0] == 1.0
    # Second workout: priors from first only (HR ~100-110, speed 2.0).
    assert out.iloc[1]["prior_missing"][0] == 0.0
    assert np.isclose(out.iloc[1]["prior_hr_p50"][0], 105.0)
    assert np.isclose(out.iloc[1]["prior_speed_p50"][0], 2.0)
    # Relative speed at 4.0 m/s vs prior p50=2.0 → 2.0
    assert np.allclose(out.iloc[1]["rel_speed_p50"], [2.0, 2.0])


def test_run_personal_priors_never_use_future_workouts():
    df = _personal_df()
    out, _ = add_activity_features(
        df,
        "run_personal",
        population_mask=df["in_train"],
        sport="run",
    )
    # Third workout priors must not include its own HR (120-130) as the only source.
    # Priors from workouts 1+2: HR in {100,110,150,160}, max should be 160 not 130-only.
    assert out.iloc[2]["prior_missing"][0] == 0.0
    assert out.iloc[2]["prior_hr_max"][0] == 160.0
    # Must not equal the current workout's own mean.
    assert out.iloc[2]["prior_hr_p50"][0] != 125.0


def test_population_fallback_ignores_heldout_mask():
    df = _personal_df()
    # Only user 1 first workout is train; held-out has high HR that must not enter population.
    population_mask = df["id"].eq(10)
    out, _ = add_activity_features(
        df,
        "run_personal",
        population_mask=population_mask,
        sport="run",
    )
    # User 2 cold start should match user 1 train workout stats (~100-110 HR, speed 2).
    assert out.iloc[3]["prior_missing"][0] == 1.0
    assert np.isclose(out.iloc[3]["prior_hr_p50"][0], 105.0)
    assert np.isclose(out.iloc[3]["prior_speed_p50"][0], 2.0)


def test_chronological_train_val_masks_are_disjoint_and_later_in_val():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2],
            "start_dt": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-02-01",
                    "2020-02-02",
                ]
            ),
            "in_train": [True] * 6,
        }
    )
    base = df["in_train"]
    train_fit, val = chronological_train_val_masks(df, base, val_fraction=0.25)

    assert not (train_fit & val).any()
    assert (train_fit | val).equals(base)
    # User 1: 4 workouts, ~25% → 1 val = last workout.
    assert train_fit.iloc[0:3].all()
    assert val.iloc[3]
    assert not train_fit.iloc[3]
    # User 2: 2 workouts, 1 val = later.
    assert train_fit.iloc[4]
    assert val.iloc[5]
