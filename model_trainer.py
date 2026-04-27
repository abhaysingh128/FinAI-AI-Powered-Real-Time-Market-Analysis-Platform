import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_processor import DataProcessor
from model_evaluator import (
    compute_metrics,
    save_metrics,
    benchmark,
    auto_improve,
    log_improvement,
)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Apply dropout to the last LSTM layer output before linear layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class ModelTrainer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model_dir = "saved_models"
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model_path = os.path.join(self.model_dir, f"{self.ticker}_model.pth")
        self.scaler_path = os.path.join(self.model_dir, f"{self.ticker}_scaler.pkl")
        self.target_scaler_path = os.path.join(self.model_dir, f"{self.ticker}_target_scaler.pkl")
        self.metrics_path = os.path.join(self.model_dir, f"{self.ticker}_metrics.json")
        self.training_plot_path = os.path.join(self.model_dir, f"{self.ticker}_training_plot.png")
        
        # Hyperparameters
        self.lookback = 60
        self.forecast_horizon = 7

    def build_model(self, input_size=1):
        model = LSTMModel(input_size=input_size)
        model = model.to(device)
        return model

    def recursive_forecast(self, model, feature_scaler, target_scaler, initial_window, steps=7):
        window = initial_window.copy()
        predicted_scaled = []
        
        model.eval()
        for _ in range(steps):
            x_input = window[-self.lookback:].reshape(1, self.lookback, 1)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(x_tensor).cpu().numpy().flatten()[0]
                
            predicted_scaled.append(pred_scaled)
            window = np.vstack([window, [[pred_scaled]]])
            
        return target_scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()

    def returns_to_prices(self, anchor_price, returns):
        # Stub - no longer needed with direct price prediction
        pass

    def train(self, epochs=30, batch_size=32):
        print(f"Preparing data for {self.ticker}...")
        
        dp = DataProcessor(self.ticker)
        result = dp.get_training_data()
        
        if result[0] is None:
            print("Failed to get training data.")
            return False
            
        X_train, y_train, X_test, y_test, feature_scaler, target_scaler, close_prices = result
        print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
        
        input_size = X_train.shape[2]
        model = self.build_model(input_size=input_size)
        
        criterion = nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patienceCounter = 0
        patience = 5
        best_model_state = None
        
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
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
                
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patienceCounter = 0
            else:
                patienceCounter += 1
                
            if patienceCounter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
            
        history = {'loss': train_losses, 'val_loss': val_losses}

        # 4. Evaluation — recursive forecast on test set
        print("Evaluating with recursive 7-day forecasting on test set...")
        
        eval_indices = list(range(0, len(X_test) - 7, 7))
        if len(eval_indices) > 100:
            eval_indices = eval_indices[:100]
            
        y_true_prices_all, y_pred_prices_all = [], []
        
        for idx in eval_indices:
            initial_window = X_test[idx]
            
            pred_prices = self.recursive_forecast(model, feature_scaler, target_scaler, initial_window, steps=7)
            
            actual_scaled = y_test[idx:idx+7]
            actual_prices = target_scaler.inverse_transform(actual_scaled).flatten()
            
            if len(actual_prices) == 7:
                y_true_prices_all.append(actual_prices)
                y_pred_prices_all.append(pred_prices)
            
        if len(y_true_prices_all) == 0:
            print("Warning: No valid evaluation samples. Using single-step evaluation.")
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_t).cpu().numpy()
            y_test_real = target_scaler.inverse_transform(y_test)
            y_pred_real = target_scaler.inverse_transform(y_pred_scaled)
            y_true_arr = y_test_real
            y_pred_arr = y_pred_real
        else:
            y_true_arr = np.array(y_true_prices_all)
            y_pred_arr = np.array(y_pred_prices_all)

        metrics = compute_metrics(y_true_arr, y_pred_arr)
        r2 = r2_score(y_true_arr.flatten(), y_pred_arr.flatten())
        metrics["r2_score"] = round(float(r2), 6)
        
        print(f"[Trainer] Metrics: {metrics}")

        self.save_artifacts(model, metrics, history, y_true_arr, y_pred_arr, feature_scaler, target_scaler)
        save_metrics(self.ticker, metrics, directory="saved_data")

        mean_price = float(np.mean(y_true_arr))
        grade = benchmark(metrics, mean_price=mean_price)
        print(f"[Trainer] Model quality: {grade}  (MAPE={metrics['mape']:.2f}%)")

        if grade == "Poor":
            before_metrics = dict(metrics)
            improved_metrics, status = auto_improve(
                trainer=self, dp=dp, feature_scaler=feature_scaler, target_scaler=target_scaler,
                base_metrics=metrics, base_epochs=epochs, base_units=64, base_lr=0.001
            )
            log_improvement(self.ticker, before_metrics, improved_metrics, status)
            if status == "improved":
                save_metrics(self.ticker, improved_metrics, directory="saved_data")
        else:
            print(f"[Trainer] Model is {grade} — no auto-improvement needed.")

        return True

    def save_artifacts(self, model, metrics, history, y_true, y_pred, feature_scaler, target_scaler):
        torch.save(model.state_dict(), self.model_path)
        joblib.dump(feature_scaler, self.scaler_path)
        joblib.dump(target_scaler, self.target_scaler_path)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss (MSE)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if len(y_true.shape) > 1 and y_true.shape[1] == 7:
            days = np.arange(1, 8)
            plt.plot(days, y_true[0], 'g-', label='Actual (Seq 0)')
            plt.plot(days, y_pred[0], 'r--', label='Pred (Seq 0)')
            plt.title('Sample 7-Day Forecast')
        else:
            plt.plot(y_true.flatten()[:50], 'g-', label='Actual')
            plt.plot(y_pred.flatten()[:50], 'r--', label='Predicted')
            plt.title('Prediction vs Actual')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.training_plot_path)
        plt.close()
        print("Artifacts saved.")

    def incremental_update(self, model, feature_scaler, target_scaler):
        print(f"Checking for incremental update for {self.ticker}...")
        dp = DataProcessor(self.ticker)
        
        has_new_data = dp.update_and_get_new_data()
        if not has_new_data:
            print("No new data found. Skipping incremental update.")
            return model
            
        X, y = dp.get_incremental_data(feature_scaler, target_scaler, recent_window=120)
        
        if X is None or len(X) == 0:
            print("Not enough new sequences for fine-tuning.")
            return model
            
        print(f"Fine-tuning model on {len(X)} recent sequences...")
        
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model.train()
        for epoch in range(2):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), self.model_path)
        print("Incremental update complete. Model over-written.")
        
        return model

if __name__ == "__main__":
    trainer = ModelTrainer("INFY.NS")
    trainer.train(epochs=1)
