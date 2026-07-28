import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.trainer import Trainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)


def make_trainer(**kwargs):
    # Third arg is val_dataloader (checkpoint selection); never held-out test.
    return Trainer(DummyModel(), train_dataloader=[], val_dataloader=[], **kwargs)


def test_mse_sum_preserves_original_loss_scale():
    trainer = make_trainer(loss_type="mse_sum")
    predictions = torch.tensor([[100.0, 120.0]])
    heart_rate = torch.tensor([[110.0, 115.0]])

    assert trainer.training_loss(predictions, heart_rate).item() == 125.0


def test_weighted_mae_prioritizes_warmup_and_hr_extremes():
    trainer = make_trainer(
        loss_type="mae",
        warmup_weight=2.0,
        warmup_steps=1,
        low_hr_weight=3.0,
        high_hr_weight=4.0,
    )
    predictions = torch.tensor([[100.0, 160.0, 190.0]])
    heart_rate = torch.tensor([[110.0, 150.0, 175.0]])

    # weights: first low-HR warmup -> 6, middle -> 1, high-HR -> 4
    expected = (10.0 * 6.0 + 10.0 * 1.0 + 15.0 * 4.0) / (6.0 + 1.0 + 4.0)
    assert torch.isclose(trainer.training_loss(predictions, heart_rate), torch.tensor(expected))


def test_huber_loss_is_mean_reduced_after_alignment():
    trainer = make_trainer(loss_type="huber", huber_delta=5.0)
    predictions = torch.tensor([[0.0, 10.0, 100.0]])
    heart_rate = torch.tensor([[0.0, 0.0]])

    # Aligned errors are [0, 10]. Huber(delta=5): [0, 37.5], mean 18.75.
    assert torch.isclose(trainer.training_loss(predictions, heart_rate), torch.tensor(18.75))


def test_mean_bias_weight_zero_matches_step_loss():
    trainer = make_trainer(loss_type="huber", huber_delta=5.0, mean_bias_weight=0.0)
    predictions = torch.tensor([[100.0, 120.0], [150.0, 160.0]])
    heart_rate = torch.tensor([[110.0, 110.0], [140.0, 140.0]])
    step_only = make_trainer(loss_type="huber", huber_delta=5.0)
    assert torch.isclose(
        trainer.training_loss(predictions, heart_rate),
        step_only.training_loss(predictions, heart_rate),
    )


def test_mean_bias_loss_penalizes_chunk_level_offset():
    # Constant +10 BPM level miss on both chunks; step MAE = 10, mean bias = 10.
    trainer = make_trainer(loss_type="mae", mean_bias_weight=0.5)
    predictions = torch.tensor([[110.0, 120.0], [160.0, 170.0]])
    heart_rate = torch.tensor([[100.0, 110.0], [150.0, 160.0]])
    # step MAE = 10; mean bias term = 0.5 * 10 = 5; total = 15
    assert torch.isclose(trainer.training_loss(predictions, heart_rate), torch.tensor(15.0))
