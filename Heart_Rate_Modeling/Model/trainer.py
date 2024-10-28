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
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate=1e-3, n_epochs=10, device='cpu'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=4)
        self.best_loss = float('inf')

    def train(self):
        for epoch in range(self.n_epochs):
            start = time.time()
            epoch_loss = 0
            self.model.train()

            for batch in tqdm(self.train_dataloader):
                activity = torch.tensor(batch["activity"]).float().to(self.device)
                times = torch.tensor(batch["time"]).float().to(self.device)
                workout_id = torch.tensor(batch["workout_id"]).to(self.device)
                subject_id = torch.tensor(batch["subject_id"]).to(self.device)
                history = torch.tensor(batch["history"]).float().to(self.device) if batch["history"] is not None else None
                history_length = torch.tensor(batch["history_length"]).to(self.device) if batch["history_length"] is not None else None
                heart_rate = torch.tensor(batch["heart_rate"]).float().to(self.device)

                predictions = self.model.forecast_batch(
                    activity=activity,
                    times=times,
                    workout_id=workout_id,
                    subject_id=subject_id,
                    history=history,
                    history_length=history_length
                )

                if predictions.size(1) > heart_rate.size(1):
                    predictions = predictions[:, :heart_rate.size(1)]
                elif predictions.size(1) < heart_rate.size(1):
                    heart_rate = heart_rate[:, :predictions.size(1)]

                heart_rate_reconstruction_l2 = l2_error(predictions, heart_rate)
                loss = heart_rate_reconstruction_l2

                self.optimizer.zero_grad()
                loss.backward()

                if self.model.config.clip_gradient > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.config.clip_gradient)

                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            train_l1, train_relative = self.evaluate(self.train_dataloader)
            test_l1, test_relative = self.evaluate(self.test_dataloader)

            self.scheduler.step(test_l1)

            if test_l1 < self.best_loss:
                self.best_loss = test_l1
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {test_l1:.6f}).  Saving model ...")
                torch.save(self.model.state_dict(), 'best_model.pt')

            print(f"Epoch {epoch} took {time.time() - start:.1f} seconds",
                  f"Train mean l1: {train_l1:.3f} bpm (= {train_relative:.3f} %)",
                  f"Test mean l1: {test_l1:.3f} bpm (= {test_relative:.3f} %)",
                  sep="\n")

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            predicted_hr_all = []
            true_hr_all = []

            for batch in tqdm(dataloader):
                activity = torch.tensor(batch["activity"]).float().to(self.device)
                times = torch.tensor(batch["time"]).float().to(self.device)
                workout_id = torch.tensor(batch["workout_id"]).to(self.device)
                subject_id = torch.tensor(batch["subject_id"]).to(self.device)
                history = torch.tensor(batch["history"]).float().to(self.device) if batch["history"] is not None else None
                history_length = torch.tensor(batch["history_length"]).to(self.device) if batch["history_length"] is not None else None
                heart_rate = torch.tensor(batch["heart_rate"]).float().to(self.device)

                predictions = self.model.forecast_batch(
                    activity=activity,
                    times=times,
                    workout_id=workout_id,
                    subject_id=subject_id,
                    history=history,
                    history_length=history_length
                )

                if predictions.size(1) > heart_rate.size(1):
                    predictions = predictions[:, :heart_rate.size(1)]
                elif predictions.size(1) < heart_rate.size(1):
                    heart_rate = heart_rate[:, :predictions.size(1)]

                predicted_hr_all.extend(predictions.cpu().numpy())
                true_hr_all.extend(heart_rate.cpu().numpy())

            predicted_hr_all = np.concatenate(predicted_hr_all)
            true_hr_all = np.concatenate(true_hr_all)

            l1_error = np.mean(np.abs(predicted_hr_all - true_hr_all))
            relative_error = 100 * l1_error / np.mean(np.abs(true_hr_all))

            return l1_error, relative_error
