import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model's heart rate predictions.

    Parameters:
    - model: Trained model to evaluate.
    - dataloader: DataLoader for the evaluation dataset.
    - device: Device to run the model on.

    Returns:
    - all_true_hr: List of true heart rate values.
    - all_pred_hr: List of predicted heart rate values.
    """
    model.to(device)  # Move model to the device
    model.eval()
    all_true_hr = []
    all_pred_hr = []

    with torch.no_grad():
        for batch in dataloader:
            activity = torch.tensor(batch["activity"], dtype=torch.float32).to(device).clone().detach()
            times = torch.tensor(batch["time"], dtype=torch.float32).to(device).clone().detach()
            workout_id = torch.tensor(batch["workout_id"], dtype=torch.int64).to(device)
            subject_id = torch.tensor(batch["subject_id"], dtype=torch.int64).to(device)
            history = torch.tensor(batch["history"], dtype=torch.float32).to(device).clone().detach() if batch["history"] is not None else None
            history_length = torch.tensor(batch["history_length"], dtype=torch.int64).to(device) if batch["history_length"] is not None else None
            true_hr = batch["heart_rate"].flatten()
            
            predictions = model.forecast_batch(activity, times, workout_id, subject_id, history, history_length)
            pred_hr = predictions.cpu().numpy().flatten()
            
            min_len = min(len(true_hr), len(pred_hr))
            all_true_hr.extend(true_hr[:min_len])
            all_pred_hr.extend(pred_hr[:min_len])

    return all_true_hr, all_pred_hr

def plot_evaluation_results(true_hr, pred_hr):
    """
    Plot the evaluation results.

    Parameters:
    - true_hr: List of true heart rate values.
    - pred_hr: List of predicted heart rate values.
    """
    plt.figure(figsize=(12, 6))

    # Plot true vs. predicted heart rate
    plt.subplot(1, 2, 1)
    plt.plot(true_hr, label='True HR', color='r', linestyle='--')
    plt.plot(pred_hr, label='Predicted HR', color='b')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.title('True vs. Predicted Heart Rate')

    # Plot error distribution
    errors = np.array(true_hr) - np.array(pred_hr)
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, color='gray')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show()

def compute_metrics(true_hr, pred_hr):
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    true_hr : list
        List of true heart rate values.
    pred_hr : list
        List of predicted heart rate values.
    
    Returns
    -------
    mae : float
        Mean Absolute Error.
    mse : float
        Mean Squared Error.
    rmse : float
        Root Mean Squared Error.
    mape : float
        Mean Absolute Percentage Error.
    correlation : float
        Pearson Correlation Coefficient.
    """
    true_hr = np.array(true_hr)
    pred_hr = np.array(pred_hr)
    
    mae = mean_absolute_error(true_hr, pred_hr)
    mse = mean_squared_error(true_hr, pred_hr)
    rmse = np.sqrt(mse)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((true_hr - pred_hr) / (true_hr + epsilon))) * 100
    
    # Remove NaN values and extremely high values from correlation calculation
    valid_idx = np.isfinite(true_hr) & np.isfinite(pred_hr)
    correlation = np.corrcoef(true_hr[valid_idx], pred_hr[valid_idx])[0, 1] if np.any(valid_idx) else 0.0
    
    return mae, mse, rmse, mape, correlation

def plot_true_vs_predicted(true_hr, pred_hr):
    """
    Plot true vs. predicted heart rate and the error distribution.

    Parameters
    ----------
    true_hr : list
        List of true heart rate values.
    pred_hr : list
        List of predicted heart rate values.
    """
    plt.figure(figsize=(14, 7))

    # Plot true vs. predicted heart rate
    plt.subplot(1, 2, 1)
    plt.plot(true_hr, label='True HR', color='red', linestyle='dotted')
    plt.plot(pred_hr, label='Predicted HR', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate')
    plt.legend()

    # Plot error distribution
    errors = np.array(true_hr) - np.array(pred_hr)
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=50, color='gray', edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.show()
