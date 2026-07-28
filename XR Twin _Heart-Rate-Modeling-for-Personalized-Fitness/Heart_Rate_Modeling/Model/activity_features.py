import numpy as np
import pandas as pd


BASIC_ACTIVITY_COLUMNS = ["speed_h", "speed_v"]
RUN_INTENSITY_COLUMNS = BASIC_ACTIVITY_COLUMNS + [
    "speed_abs",
    "speed_delta",
    "abs_speed_delta",
    "low_speed_fraction",
    "warmup_fraction",
    "incline_proxy",
]
# Per-step relative effort + constant-over-time prior channels (broadcast).
SUBJECT_PRIOR_SCALAR_COLUMNS = [
    "prior_hr_p10",
    "prior_hr_p50",
    "prior_hr_p90",
    "prior_hr_max",
    "prior_speed_p50",
    "prior_speed_p90",
    "prior_missing",
]
RELATIVE_SPEED_COLUMNS = [
    "rel_speed_p50",
    "rel_speed_p90",
    "subject_low_speed_fraction",
]
RUN_PERSONAL_COLUMNS = RUN_INTENSITY_COLUMNS + RELATIVE_SPEED_COLUMNS + SUBJECT_PRIOR_SCALAR_COLUMNS

# Safe defaults when a subject has no prior workouts and no population stats.
_DEFAULT_POPULATION = {
    "prior_hr_p10": 110.0,
    "prior_hr_p50": 140.0,
    "prior_hr_p90": 165.0,
    "prior_hr_max": 185.0,
    "prior_speed_p50": 2.8,
    "prior_speed_p90": 3.6,
}

_EPS = 1e-3


def _as_float_array(values):
    return np.asarray(values, dtype=float)


def _speed_delta(speed_h):
    speed_h = _as_float_array(speed_h)
    if len(speed_h) == 0:
        return speed_h
    return np.diff(speed_h, prepend=speed_h[0])


def _warmup_fraction(time_grid):
    time_grid = _as_float_array(time_grid)
    # time_grid is normalized by 20 minutes in preprocessing. Two minutes = 0.1.
    return np.clip(1.0 - (time_grid / 0.1), 0.0, 1.0)


def _constant_channel(length, value):
    return np.full(length, float(value), dtype=float)


def _prior_stats_from_arrays(heart_rates, speeds):
    """Compute subject prior stats from concatenated prior-workout arrays."""
    hr = np.concatenate(heart_rates) if heart_rates else np.array([], dtype=float)
    sp = np.concatenate(speeds) if speeds else np.array([], dtype=float)
    if hr.size == 0 or sp.size == 0:
        return None
    return {
        "prior_hr_p10": float(np.percentile(hr, 10)),
        "prior_hr_p50": float(np.percentile(hr, 50)),
        "prior_hr_p90": float(np.percentile(hr, 90)),
        "prior_hr_max": float(min(np.max(hr), 210.0)),
        "prior_speed_p50": float(max(np.percentile(sp, 50), _EPS)),
        "prior_speed_p90": float(max(np.percentile(sp, 90), _EPS)),
    }


def compute_population_prior_stats(df, population_mask=None, sport=None):
    """
    Aggregate HR/speed percentiles over a population of workouts.

    Used only as cold-start fallback. Callers should pass train-only masks so
    held-out labels never enter the fallback distribution.
    """
    if population_mask is None:
        subset = df
    else:
        subset = df.loc[population_mask]
    if sport is not None and "sport" in subset.columns:
        subset = subset.loc[subset["sport"].eq(sport)]
    if len(subset) == 0:
        return dict(_DEFAULT_POPULATION)

    hrs, speeds = [], []
    for _, row in subset.iterrows():
        hrs.append(_as_float_array(row["heart_rate"]))
        speeds.append(_as_float_array(row["speed_h"]))
    stats = _prior_stats_from_arrays(hrs, speeds)
    return stats if stats is not None else dict(_DEFAULT_POPULATION)


def chronological_train_val_masks(
    df,
    base_mask,
    subject_id_column="userId",
    time_column="start_dt",
    val_fraction=0.15,
):
    """
    Split rows under base_mask into train_fit / val chronologically per subject.

    For each subject, the last ~val_fraction of their base_mask workouts (by
    start time) become validation. Subjects with a single workout stay entirely
    in train_fit so training is never emptied.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    train_fit = pd.Series(False, index=df.index)
    val = pd.Series(False, index=df.index)
    if val_fraction == 0.0 or not base_mask.any():
        train_fit.loc[base_mask] = True
        return train_fit, val

    subset = df.loc[base_mask]
    for _, group in subset.groupby(subject_id_column, sort=False):
        group = group.sort_values(time_column)
        n = len(group)
        if n < 2:
            train_fit.loc[group.index] = True
            continue
        n_val = int(round(n * val_fraction))
        n_val = max(1, n_val)
        n_val = min(n_val, n - 1)
        train_fit.loc[group.index[:-n_val]] = True
        val.loc[group.index[-n_val:]] = True
    return train_fit, val


def _add_run_intensity_features(df):
    df = df.copy()
    speed_h = df["speed_h"].apply(_as_float_array)
    speed_v = df["speed_v"].apply(_as_float_array)
    speed_delta = speed_h.apply(_speed_delta)

    df["speed_abs"] = [
        np.sqrt(np.square(h) + np.square(v))
        for h, v in zip(speed_h, speed_v)
    ]
    df["speed_delta"] = speed_delta
    df["abs_speed_delta"] = speed_delta.apply(np.abs)
    # 2.2 m/s is roughly an 8:20 min/km pace. Values below it often behave more
    # like low-intensity jog/walk sessions than steady running.
    df["low_speed_fraction"] = speed_h.apply(lambda h: np.clip((2.2 - h) / 2.2, 0.0, 1.0))
    df["warmup_fraction"] = df["time_grid"].apply(_warmup_fraction)
    df["incline_proxy"] = [
        np.clip(v / np.maximum(h, 0.5), -1.0, 1.0)
        for h, v in zip(speed_h, speed_v)
    ]
    return df


def add_subject_relative_features(
    df,
    population_mask=None,
    subject_id_column="userId",
    time_column="start_dt",
    sport=None,
    prefer_same_sport=True,
):
    """
    Add chronological subject priors and relative-speed channels.

    For each workout w of subject s, priors are computed only from workouts of s
    with start_dt < w.start_dt (never current or future). When prefer_same_sport
    is True and sport is set, same-sport priors are preferred when available,
    otherwise any-sport priors of s are used. Cold-start falls back to
    population stats from population_mask (should be train-only).
    """
    df = df.copy()
    population = compute_population_prior_stats(df, population_mask=population_mask, sport=sport)

    n = len(df)
    prior_values = {key: np.zeros(n, dtype=float) for key in SUBJECT_PRIOR_SCALAR_COLUMNS}
    rel_p50 = [None] * n
    rel_p90 = [None] * n
    subj_low = [None] * n

    # Position lookup for writing by original row order.
    positions = {idx: pos for pos, idx in enumerate(df.index)}

    # Accumulate prior HR/speed arrays chronologically per subject.
    # Store list of (start_dt, sport, hr, speed_h) in time order.
    for subject_id, group in df.groupby(subject_id_column, sort=False):
        group = group.sort_values(time_column)
        past_any = []  # list of (hr, speed, sport)
        for idx, row in group.iterrows():
            pos = positions[idx]
            hr = _as_float_array(row["heart_rate"])
            speed = _as_float_array(row["speed_h"])
            length = len(speed)
            row_sport = row["sport"] if "sport" in row.index else None

            # Select prior pool: same sport if available, else any prior.
            prior_hrs, prior_speeds = [], []
            if prefer_same_sport and sport is not None:
                for phr, psp, psp_sport in past_any:
                    if psp_sport == sport:
                        prior_hrs.append(phr)
                        prior_speeds.append(psp)
            if not prior_hrs:
                for phr, psp, _ in past_any:
                    prior_hrs.append(phr)
                    prior_speeds.append(psp)

            stats = _prior_stats_from_arrays(prior_hrs, prior_speeds)
            missing = 0.0 if stats is not None else 1.0
            if stats is None:
                stats = population

            for key in SUBJECT_PRIOR_SCALAR_COLUMNS:
                if key == "prior_missing":
                    prior_values[key][pos] = missing
                else:
                    prior_values[key][pos] = stats[key]

            sp50 = max(stats["prior_speed_p50"], _EPS)
            sp90 = max(stats["prior_speed_p90"], _EPS)
            rel_p50[pos] = speed / sp50
            rel_p90[pos] = speed / sp90
            subj_low[pos] = np.clip((sp50 - speed) / sp50, 0.0, 1.0)

            # Current workout becomes prior for later ones only after stats used.
            past_any.append((hr, speed, row_sport))
            # Keep arrays as references; do not mutate after append.

    for key in SUBJECT_PRIOR_SCALAR_COLUMNS:
        df[key] = [
            _constant_channel(len(_as_float_array(df.iloc[i]["speed_h"])), prior_values[key][i])
            for i in range(n)
        ]
    df["rel_speed_p50"] = rel_p50
    df["rel_speed_p90"] = rel_p90
    df["subject_low_speed_fraction"] = subj_low
    return df


def add_activity_features(
    df,
    feature_set,
    population_mask=None,
    subject_id_column="userId",
    time_column="start_dt",
    sport=None,
):
    """
    Add per-step activity features used by experimental model variants.

    Feature sets:
      - basic: published speed_h, speed_v only
      - run_intensity: absolute intensity / warmup features
      - run_personal: run_intensity + chronological subject priors + relative speed

    For run_personal, population_mask should be the train split (never held-out)
    so cold-start fallbacks do not leak test labels.
    """
    if feature_set == "basic":
        return df, list(BASIC_ACTIVITY_COLUMNS)

    if feature_set == "run_intensity":
        return _add_run_intensity_features(df), list(RUN_INTENSITY_COLUMNS)

    if feature_set == "run_personal":
        out = _add_run_intensity_features(df)
        out = add_subject_relative_features(
            out,
            population_mask=population_mask,
            subject_id_column=subject_id_column,
            time_column=time_column,
            sport=sport,
            prefer_same_sport=True,
        )
        return out, list(RUN_PERSONAL_COLUMNS)

    raise ValueError(f"Unknown feature_set: {feature_set}")
