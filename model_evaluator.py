"""
model_evaluator.py
──────────────────
Evaluation, benchmarking, and auto-improvement for the Stacked LSTM model.

Responsibilities:
  1. Compute MSE, RMSE, MAE, MAPE on inverse-scaled real prices
  2. Grade model quality (Excellent / Good / Poor)
  3. Auto-tune & retrain once if performance is poor
  4. Log improvement results

Integration:
  Called from ModelTrainer.train() after initial evaluation.
  Does NOT touch UI, sentiment, fusion, or real-time inference.
"""

import os
import json
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────────────
# PART 1 — Compute Metrics
# ──────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics on real (inverse-scaled) prices.

    Args:
        y_true: Ground-truth prices, shape (N, 7) or flat.
        y_pred: Predicted prices,   shape (N, 7) or flat.

    Returns:
        dict with mse, rmse, mae, mape keys.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse  = float(mean_squared_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true_flat, y_pred_flat))

    # MAPE — guard against division-by-zero
    nonzero_mask = y_true_flat != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask])
                           / y_true_flat[nonzero_mask])) * 100
        )
    else:
        mape = 0.0

    return {
        "mse":  round(mse, 6),
        "rmse": round(rmse, 6),
        "mae":  round(mae, 6),
        "mape": round(mape, 4),   # percentage
    }


# ──────────────────────────────────────────────────
# PART 2 — Save Metrics JSON
# ──────────────────────────────────────────────────

def save_metrics(ticker: str, metrics: dict, directory: str = "saved_data") -> str:
    """
    Persist metrics to saved_data/{symbol}_metrics.json.

    Returns the file path.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{ticker}_metrics.json")

    payload = {
        **metrics,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"[Evaluator] Metrics saved -> {path}")
    return path


# ──────────────────────────────────────────────────
# PART 2 — Benchmark Check
# ──────────────────────────────────────────────────

# Thresholds
_MAPE_EXCELLENT = 5.0   # %
_MAPE_GOOD      = 10.0  # %
_RMSE_REL_GOOD  = 3.0   # %  (RMSE / mean price)

def benchmark(metrics: dict, mean_price: float = None) -> str:
    """
    Grade model quality.

    Returns one of: 'Excellent', 'Good', 'Poor'.

    Primary criterion  → MAPE
    Secondary criterion → RMSE relative to mean price (if available)
    """
    mape = metrics.get("mape", 100.0)

    # Primary: MAPE-based grading
    if mape < _MAPE_EXCELLENT:
        return "Excellent"
    elif mape < _MAPE_GOOD:
        return "Good"

    # Secondary: RMSE relative check (fallback for edge-case where
    # MAPE is misleading on low-priced stocks)
    if mean_price and mean_price > 0:
        rmse_rel = (metrics.get("rmse", 0) / mean_price) * 100
        if rmse_rel < _RMSE_REL_GOOD:
            return "Good"

    return "Poor"


# ──────────────────────────────────────────────────
# PART 3 — Auto Model Improvement
# ──────────────────────────────────────────────────

def suggest_hyperparams(current_epochs: int, current_units: int, current_lr: float):
    """
    Return improved hyperparameters for a retry.
    """
    new_epochs = current_epochs + 20
    new_units  = current_units # Keep structurally identical
    new_lr     = current_lr * 0.5

    return new_epochs, new_units, new_lr


def auto_improve(trainer, dp, feature_scaler, target_scaler,
                 base_metrics: dict,
                 base_epochs: int  = 50,
                 base_units: int   = 50,
                 base_lr: float    = 0.001):
    """
    One-shot auto-improvement (2-feature return-based architecture).

    1. Compute improved hyper-params
    2. Rebuild & retrain from scratch
    3. Re-evaluate using recursive 7-day forecasting
    4. Keep the better model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    print("[Evaluator] Model quality is POOR -- attempting auto-improvement...")

    new_epochs, new_units, new_lr = suggest_hyperparams(base_epochs, base_units, base_lr)
    print(f"[Evaluator] Retry params -> epochs={new_epochs}, units={new_units}, lr={new_lr}")

    try:
        # Get fresh training data (uses cached CSV, fast)
        result = dp.get_training_data()
        if result[0] is None:
            return base_metrics, "error"
        X_train, y_train, X_test, y_test, feature_scaler, target_scaler, close_prices = result

        input_size = X_train.shape[2]
        from model_trainer import LSTMModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(input_size=input_size, hidden_size=new_units).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=new_lr)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        best_val_loss = float('inf')
        patienceCounter = 0
        patience = 5
        best_model_state = None
        
        train_losses = []
        val_losses = []

        for epoch in range(new_epochs):
            model.train()
            epoch_train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * X_batch.size(0)
                
            epoch_train_loss /= len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_t)
                val_loss = criterion(val_output, y_test_t).item()
                val_losses.append(val_loss)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patienceCounter = 0
            else:
                patienceCounter += 1
                
            if patienceCounter >= patience:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
            
        history = {'loss': train_losses, 'val_loss': val_losses}

        # Evaluate using recursive 7-day forecasting
        eval_indices = list(range(0, len(X_test) - 7, 7))
        if len(eval_indices) > 100:
            eval_indices = eval_indices[:100]

        y_true_all, y_pred_all = [], []
        for idx in eval_indices:
            initial_window = X_test[idx]
            
            pred_prices = trainer.recursive_forecast(
                model, feature_scaler, target_scaler, initial_window, steps=7
            )
            
            actual_scaled = y_test[idx:idx+7]
            actual_prices = target_scaler.inverse_transform(actual_scaled).flatten()
            
            if len(actual_prices) == 7:
                y_true_all.append(actual_prices)
                y_pred_all.append(pred_prices)

        if len(y_true_all) == 0:
            return base_metrics, "error"

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        new_metrics = compute_metrics(y_true_arr, y_pred_arr)

        # Decision: keep improved model only if MAPE decreased
        if new_metrics["mape"] < base_metrics["mape"]:
            print(f"[Evaluator] [OK] Improvement! MAPE {base_metrics['mape']:.2f}% -> {new_metrics['mape']:.2f}%")
            trainer.save_artifacts(model, new_metrics, history, y_true_arr, y_pred_arr, feature_scaler, target_scaler)
            return new_metrics, "improved"
        else:
            print(f"[Evaluator] [X] No improvement (MAPE {new_metrics['mape']:.2f}% >= {base_metrics['mape']:.2f}%). Keeping original model.")
            return base_metrics, "no_improvement"

    except Exception as e:
        print(f"[Evaluator] Auto-improvement failed: {e}")
        return base_metrics, "error"


# ──────────────────────────────────────────────────
# PART 5 — Improvement Log
# ──────────────────────────────────────────────────

def log_improvement(ticker: str,
                    before_metrics: dict,
                    after_metrics: dict,
                    status: str,
                    directory: str = "saved_data"):
    """
    Append an improvement record to saved_data/{symbol}_improvement_log.json.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{ticker}_improvement_log.json")

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "before":    before_metrics,
        "after":     after_metrics,
        "status":    status,
    }

    # Append to existing log (list of entries)
    if os.path.exists(path):
        with open(path, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=4)

    print(f"[Evaluator] Improvement log updated -> {path}")
