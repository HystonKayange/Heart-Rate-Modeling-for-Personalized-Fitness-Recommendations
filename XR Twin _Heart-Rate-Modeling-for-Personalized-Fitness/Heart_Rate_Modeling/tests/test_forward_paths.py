"""
Smoke tests for the three model configurations, runnable without the FitRec data.

These check that each component is actually reached and receives gradient - the
failure mode that produced the published results was a component that was
instantiated, saved into the checkpoint, and never executed.
"""
import os
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.data import WorkoutDatasetConfig
from Model.dbn import DBNConfig, DBNModel

BATCH, SEQ, N_SUBJECTS, HISTORY = 4, 64, 6, 128


def build(_n_subjects=N_SUBJECTS, **overrides):
    data_config = WorkoutDatasetConfig(activity_columns=["speed_h", "speed_v"], weather_columns=[])
    config = DBNConfig(data_config=data_config, seq_length=SEQ, device="cpu", **overrides)
    workouts_info = pd.DataFrame(
        {"subject_id": list(range(_n_subjects)), "workout_id": list(range(100, 100 + _n_subjects))}
    )
    return DBNModel(config=config, workouts_info=workouts_info)


def batch():
    return dict(
        activity=torch.randn(BATCH, SEQ, 2),
        times=torch.rand(BATCH, SEQ),
        workout_id=torch.arange(100, 100 + BATCH),
        subject_id=torch.arange(BATCH),
        history=torch.randn(BATCH, HISTORY, 5),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},                                                            # as-published
        {"use_adafs": True},
        {"use_physiological_head": True},
        {"use_adafs": True, "use_physiological_head": True},           # as-described
    ],
    ids=["published", "adafs", "physiological", "full"],
)
def test_shape_and_gradient_flow(overrides):
    model = build(**overrides)
    predictions = model.forecast_batch(**batch())
    assert predictions.shape == (BATCH, SEQ), predictions.shape

    predictions.sum().backward()

    if overrides.get("use_adafs"):
        assert model.adafs_soft.controller.mlp.mlps[0][0].weight.grad is not None, "AdaFS received no gradient"
    if overrides.get("use_physiological_head"):
        for name in ("A", "B", "hr_min", "hr_range", "intensity"):
            grad = getattr(model, name).layers[0].weight.grad
            assert grad is not None and grad.abs().sum() > 0, f"{name} received no gradient"


def test_physiological_head_respects_hr_bounds():
    model = build(use_physiological_head=True)
    lo = model.config.hr_min_bounds[0]
    hi = model.config.hr_max_bounds[1]
    with torch.no_grad():
        predictions = model.forecast_batch(**batch())
    assert predictions.min() >= lo, predictions.min()
    assert predictions.max() <= hi, predictions.max()


def test_published_checkpoint_still_loads_into_default_config():
    """
    The released best_model.pt must still restore the path that actually trained.
    Keys for the unused components changed, so this is a strict=False load - but
    every live parameter must match, and nothing live may be missing.
    """
    checkpoint = os.path.join(os.path.dirname(__file__), "..", "examples", "best_model.pt")
    if not os.path.exists(checkpoint):
        pytest.skip("best_model.pt not present")

    state_dict = torch.load(checkpoint, map_location="cpu")
    n_subjects = state_dict["embedding_store.subject_embeddings.weight"].shape[0]
    model = build(_n_subjects=n_subjects)

    # Drop the components that never executed: their shapes changed when they were
    # wired up, and the saved values are initialization noise regardless.
    dead_prefixes = ("A.", "B.", "alpha.", "beta.", "hr_min.", "hr_max.", "lstm_encoder.", "adafs_soft.")
    live = {k: v for k, v in state_dict.items() if not k.startswith(dead_prefixes)}

    result = model.load_state_dict(live, strict=False)

    live_prefixes = ("embedding_store.", "transition_model.", "emission_model.")
    missed = [k for k in result.missing_keys if k.startswith(live_prefixes)]
    assert not missed, f"live parameters missing from checkpoint: {missed}"
    assert not result.unexpected_keys, f"unexpected: {result.unexpected_keys}"
