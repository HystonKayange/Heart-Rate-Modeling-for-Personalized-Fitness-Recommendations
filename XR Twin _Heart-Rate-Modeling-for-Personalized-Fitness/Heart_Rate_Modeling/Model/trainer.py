import torch
import time
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd

def l2_reg(params):
    return sum([(p**2).sum() for p in params])

def l2_error(tensor1, tensor2=0.0):
    return ((tensor1 - tensor2)).pow(2).sum()

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate=1e-3,
        n_epochs=10,
        device='cpu',
        loss_type='mse_sum',
        huber_delta=10.0,
        low_hr_weight=1.0,
        high_hr_weight=1.0,
        warmup_weight=1.0,
        low_hr_threshold=120.0,
        high_hr_threshold=170.0,
        warmup_steps=12,
        weight_decay=0.0,
        # P3: penalize per-chunk mean level bias (targets workouts where bias ≈ MAE).
        mean_bias_weight=0.0,
        # Backward-compatible alias used by older call sites / tests.
        test_dataloader=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        # Checkpointing and LR schedule must use validation only — never held-out test.
        if val_dataloader is None and test_dataloader is not None:
            val_dataloader = test_dataloader
        if val_dataloader is None:
            raise ValueError("Trainer requires a val_dataloader for checkpoint selection")
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.low_hr_weight = low_hr_weight
        self.high_hr_weight = high_hr_weight
        self.warmup_weight = warmup_weight
        self.low_hr_threshold = low_hr_threshold
        self.high_hr_threshold = high_hr_threshold
        self.warmup_steps = warmup_steps
        self.mean_bias_weight = mean_bias_weight
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=4)
        self.best_loss = float('inf')

    def _align_predictions_and_targets(self, predictions, heart_rate):
        if predictions.size(1) > heart_rate.size(1):
            predictions = predictions[:, :heart_rate.size(1)]
        elif predictions.size(1) < heart_rate.size(1):
            heart_rate = heart_rate[:, :predictions.size(1)]
        return predictions, heart_rate

    def _loss_weights(self, heart_rate):
        weights = torch.ones_like(heart_rate)
        if self.low_hr_weight != 1.0:
            weights = torch.where(heart_rate < self.low_hr_threshold, weights * self.low_hr_weight, weights)
        if self.high_hr_weight != 1.0:
            weights = torch.where(heart_rate >= self.high_hr_threshold, weights * self.high_hr_weight, weights)
        if self.warmup_weight != 1.0 and self.warmup_steps > 0:
            n = min(self.warmup_steps, weights.size(1))
            weights[:, :n] = weights[:, :n] * self.warmup_weight
        return weights

    def _step_loss(self, predictions, heart_rate, weights):
        error = predictions - heart_rate
        if self.loss_type == 'mse_sum':
            return (weights * error.pow(2)).sum()
        if self.loss_type == 'mse':
            return (weights * error.pow(2)).sum() / weights.sum().clamp_min(1.0)
        if self.loss_type == 'mae':
            return (weights * error.abs()).sum() / weights.sum().clamp_min(1.0)
        if self.loss_type == 'huber':
            abs_error = error.abs()
            quadratic = torch.minimum(abs_error, torch.tensor(self.huber_delta, device=error.device))
            linear = abs_error - quadratic
            huber = 0.5 * quadratic.pow(2) + self.huber_delta * linear
            return (weights * huber).sum() / weights.sum().clamp_min(1.0)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _mean_bias_loss(self, predictions, heart_rate):
        """
        Per-chunk mean absolute bias: mean_b | mean_t pred - mean_t hr |.

        Penalizes constant level misses that dominate worst-workout MAE without
        reweighting individual HR zones.
        """
        pred_mean = predictions.mean(dim=1)
        true_mean = heart_rate.mean(dim=1)
        return (pred_mean - true_mean).abs().mean()

    def training_loss(self, predictions, heart_rate):
        predictions, heart_rate = self._align_predictions_and_targets(predictions, heart_rate)
        weights = self._loss_weights(heart_rate)
        loss = self._step_loss(predictions, heart_rate, weights)
        if self.mean_bias_weight != 0.0:
            loss = loss + self.mean_bias_weight * self._mean_bias_loss(predictions, heart_rate)
        return loss

    def train(self):
        for epoch in range(self.n_epochs):
            start = time.time()
            epoch_loss = 0
            self.model.train()

            for batch in tqdm(self.train_dataloader):
                activity = torch.as_tensor(batch["activity"]).float().to(self.device)
                times = torch.as_tensor(batch["time"]).float().to(self.device)
                workout_id = torch.as_tensor(batch["workout_id"]).to(self.device)
                subject_id = torch.as_tensor(batch["subject_id"]).to(self.device)
                history = torch.as_tensor(batch["history"]).float().to(self.device) if batch["history"] is not None else None
                history_length = torch.as_tensor(batch["history_length"]).to(self.device) if batch["history_length"] is not None else None
                heart_rate = torch.as_tensor(batch["heart_rate"]).float().to(self.device)

                predictions = self.model.forecast_batch(
                    activity=activity,
                    times=times,
                    workout_id=workout_id,
                    subject_id=subject_id,
                    history=history,
                    history_length=history_length
                )

                loss = self.training_loss(predictions, heart_rate)

                self.optimizer.zero_grad()
                loss.backward()

                if self.model.config.clip_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.config.clip_gradient)

                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            train_l1, train_relative = self.evaluate(self.train_dataloader)
            val_l1, val_relative = self.evaluate(self.val_dataloader)

            self.scheduler.step(val_l1)

            if val_l1 < self.best_loss:
                self.best_loss = val_l1
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_l1:.6f}).  Saving model ...")
                torch.save(self.model.state_dict(), 'best_model.pt')

            print(f"Epoch {epoch} took {time.time() - start:.1f} seconds",
                  f"Train mean l1: {train_l1:.3f} bpm (= {train_relative:.3f} %)",
                  f"Val mean l1: {val_l1:.3f} bpm (= {val_relative:.3f} %)",
                  sep="\n")

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            predicted_hr_all = []
            true_hr_all = []

            for batch in tqdm(dataloader):
                activity = torch.as_tensor(batch["activity"]).float().to(self.device)
                times = torch.as_tensor(batch["time"]).float().to(self.device)
                workout_id = torch.as_tensor(batch["workout_id"]).to(self.device)
                subject_id = torch.as_tensor(batch["subject_id"]).to(self.device)
                history = torch.as_tensor(batch["history"]).float().to(self.device) if batch["history"] is not None else None
                history_length = torch.as_tensor(batch["history_length"]).to(self.device) if batch["history_length"] is not None else None
                heart_rate = torch.as_tensor(batch["heart_rate"]).float().to(self.device)

                predictions = self.model.forecast_batch(
                    activity=activity,
                    times=times,
                    workout_id=workout_id,
                    subject_id=subject_id,
                    history=history,
                    history_length=history_length
                )

                predictions, heart_rate = self._align_predictions_and_targets(predictions, heart_rate)

                predicted_hr_all.extend(predictions.cpu().numpy())
                true_hr_all.extend(heart_rate.cpu().numpy())

            predicted_hr_all = np.concatenate(predicted_hr_all)
            true_hr_all = np.concatenate(true_hr_all)

            l1_error = np.mean(np.abs(predicted_hr_all - true_hr_all))
            relative_error = 100 * l1_error / np.mean(np.abs(true_hr_all))

            return l1_error, relative_error
