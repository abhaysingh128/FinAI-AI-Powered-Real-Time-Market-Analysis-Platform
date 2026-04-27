import os
import numpy as np
import pandas as pd
import torch
from model_trainer import LSTMModel
import joblib 
from datetime import timedelta
from data_processor import DataProcessor
from model_trainer import ModelTrainer

class MLEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model_dir = "saved_models"
        self.model_path = os.path.join(self.model_dir, f"{self.ticker}_model.pth")
        self.scaler_path = os.path.join(self.model_dir, f"{self.ticker}_scaler.pkl")
        self.target_scaler_path = os.path.join(self.model_dir, f"{self.ticker}_target_scaler.pkl")
        self.metrics_path = os.path.join(self.model_dir, f"{self.ticker}_metrics.json")
        
        self.model = None
        self.feature_scaler = None    # 2-col: [return, delta]
        self.target_scaler = None     # 1-col: returns only
        
        self.lookback = 60 

    def model_exists(self):
        """Check if model and both scalers exist."""
        return (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and
                os.path.exists(self.target_scaler_path))

    def load_resources(self):
        """Load model and both scalers."""
        try:
            if self.model_exists():
                if self.model is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model = LSTMModel(input_size=1)
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                    self.model.to(device)
                    self.model.eval()
                
                self.feature_scaler = joblib.load(self.scaler_path)
                self.target_scaler = joblib.load(self.target_scaler_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False

    def train_offline(self, epochs=50):
        """Orchestrates the new offline training pipeline."""
        print(f"Starting refactored offline training for {self.ticker}...")
        
        # 1. Training (ModelTrainer handles DataProcessor internally)
        mt = ModelTrainer(self.ticker)
        success = mt.train(epochs=epochs)
        
        if success:
            # Reload resources immediately
            self.load_resources()
            
        return success

    def load_model(self):
        """Wrapper for load_resources."""
        return self.load_resources()

    def train_if_not_exists(self):
        """Train full model if it doesn't exist."""
        if not self.model_exists():
            print("Model not found. Initiating offline training...")
            self.train_offline(epochs=50) # Fallback to existing offline method
        else:
            print("Model already exists.")

    def incremental_update(self):
        """Orchestrates incremental learning update."""
        if not self.model_exists():
            self.train_if_not_exists()
            return
            
        self.load_model()
        mt = ModelTrainer(self.ticker)
        self.model = mt.incremental_update(self.model, self.feature_scaler, self.target_scaler)

    def _recursive_forecast(self, scaled_window, steps=7):
        """
        Recursive multi-step forecasting using 1-feature Close-based model.
        """
        window = scaled_window.copy()  # shape (N, 1)
        predicted_scaled = []
        
        for _ in range(steps):
            x_input = window[-self.lookback:].reshape(1, self.lookback, 1)
            
            x_tensor = torch.tensor(x_input, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            with torch.no_grad():
                pred_scaled = self.model(x_tensor).cpu().numpy().flatten()[0]
            
            predicted_scaled.append(pred_scaled)
            window = np.vstack([window, [[pred_scaled]]])
        
        return self.target_scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()

    def _returns_to_prices(self, anchor_price, returns):
        # Stub - no longer needed
        pass

    def predict(self):
        """
        Orchestrates lifecycle & generates forecast using recursive return-based prediction.
        Replaces predict_realtime.
        """
        # 1. Existence check & train
        self.train_if_not_exists()
        
        # 2. Incremental update based on latest data
        self.incremental_update()
        
        # Ensure resources loaded
        if self.model is None or self.feature_scaler is None or self.target_scaler is None:
            if not self.load_model():
                return None, None

        # 3. Fetch recent data from local CSV
        dp = DataProcessor(self.ticker)
        raw_df = pd.read_csv(dp.raw_path, index_col=0, parse_dates=True)
        
        if raw_df is None or raw_df.empty:
            return None, None
            
        close_data = raw_df[['Close']].values
        
        if len(close_data) < self.lookback + 10:
            print("Not enough data for inference")
            return None, None

        # Compute 1-feature matrix [Close] from close prices
        features = close_data.reshape(-1, 1)
        
        # Scale all features
        scaled_features = self.feature_scaler.transform(features)
        
        # ── Past 5 Days Prediction ──
        # Use window ending 5 days ago, predict 5 steps forward
        if len(scaled_features) < self.lookback + 5:
            print("Not enough feature data for 10-day past-future prediction window.")
            return None, None
        
        past_window = scaled_features[-(self.lookback + 5):-5]  # shape (60, 1)
        past_pred_prices = self._recursive_forecast(past_window, steps=5)
        
        # ── Future 5 Days Prediction ──
        # Use most recent window, predict 5 steps forward
        future_window = scaled_features[-self.lookback:]  # shape (60, 1)
        future_pred_prices = self._recursive_forecast(future_window, steps=5)
        
        # 3. Combine Predictions
        pred_prices = np.concatenate([past_pred_prices, future_pred_prices])
        
        # 4. Generate Dates (5 past actual dates + 5 future business days)
        last_date = raw_df.index[-1]
        past_dates = raw_df.index[-5:]
        
        future_dates = []
        current_date = last_date
        count = 0
        while count < 5:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5: # Mon-Fri
                future_dates.append(current_date)
                count += 1
                
        all_dates = list(past_dates) + future_dates
        
        future_df = pd.DataFrame({
            'Date': all_dates,
            'Predicted Price': pred_prices
        }).set_index('Date')
        
        return future_df, raw_df

    def predict_realtime(self):
        """Backward compatibility alias for the old UI bindings."""
        return self.predict()

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        engine = MLEngine("INFY.NS")
        pred, raw = engine.predict()
        if pred is not None:
            print("Prediction Success (10-day window):")
            print(pred)
            # Sanity check: compare past predictions to actual
            actual_last_5 = raw['Close'].iloc[-5:].values
            pred_past_5 = pred['Predicted Price'].iloc[:5].values
            print(f"\nActual last 5 prices:    {actual_last_5}")
            print(f"Predicted last 5 prices: {pred_past_5}")
            pct_errors = np.abs((actual_last_5 - pred_past_5) / actual_last_5) * 100
            print(f"Per-day % error:         {pct_errors}")
            print(f"Mean % error (past 5):   {np.mean(pct_errors):.2f}%")
        else:
            print("Prediction Failed")
