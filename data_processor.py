import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.saved_data_dir = "saved_data"
        if not os.path.exists(self.saved_data_dir):
            os.makedirs(self.saved_data_dir)
            
        self.raw_path = os.path.join(self.saved_data_dir, f"{self.ticker}_raw.csv")
        # Removing processed path as per instructions ("Remove processed CSV saving")
        
    def fetch_raw_data(self, period="10y"):
        """
        Step 1: Fetch 5-8 (now 10) years daily stock data.
        Save ONLY raw CSV with: Date, Open, Close, Volume.
        """
        print(f"Fetching raw data for {self.ticker} (Period: {period})...")
        try:
            data = yf.download(self.ticker, period=period, progress=False, auto_adjust=False)
            
            if data.empty:
                print(f"No data found for {self.ticker}")
                return None
                
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Ensure standard columns & Chronological Order
            required_cols = ['Open', 'Close', 'Volume']
            data = data[required_cols]
            data = data.sort_index()
            
            # Save RAW data
            data.to_csv(self.raw_path)
            print(f"Raw data saved to {self.raw_path}")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def update_and_get_new_data(self, period="3mo"):
        """
        Fetches recent data, appends only unseen rows to raw CSV.
        Returns True if new data was found, False otherwise.
        """
        if not os.path.exists(self.raw_path):
            print("No existing raw data to update.")
            return False
            
        print(f"Checking for new data for {self.ticker}...")
        existing_df = pd.read_csv(self.raw_path, index_col=0, parse_dates=True)
        if existing_df.empty:
            return False
            
        last_date = existing_df.index[-1]
        
        try:
            recent_df = yf.download(self.ticker, period=period, progress=False, auto_adjust=False)
            if recent_df.empty:
                return False
                
            if isinstance(recent_df.columns, pd.MultiIndex):
                recent_df.columns = recent_df.columns.droplevel(1)
                
            required_cols = ['Open', 'Close', 'Volume']
            recent_df = recent_df[required_cols].sort_index()
            
            # Filter strictly new dates
            # Ensuring both are tz-naive or tz-aware if needed. Usually yfinance returns tz-aware, 
            # and read_csv might be naive or aware. Let's compare safely or assume same format.
            if recent_df.index.tz is not None and existing_df.index.tz is None:
                recent_df.index = recent_df.index.tz_localize(None)
            elif recent_df.index.tz is None and existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)

            new_data = recent_df[recent_df.index > last_date]
            
            if new_data.empty:
                return False
                
            # Append and save
            updated_df = pd.concat([existing_df, new_data]).sort_index()
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            updated_df.to_csv(self.raw_path)
            
            print(f"Added {len(new_data)} new rows to {self.raw_path}")
            return True
            
        except Exception as e:
            print(f"Error updating data: {e}")
            return False

    @staticmethod
    def compute_features(close_array):
        # Stub left for compatibility, just returns raw close
        return close_array, close_array

    def get_incremental_data(self, scaler, target_scaler, recent_window=120):
        """
        Loads the most recent 'recent_window' days from raw CSV.
        Creates 1-feature sequences [Close] using the provided pre-fit scalers.
        """
        if not os.path.exists(self.raw_path):
            return None, None
            
        df = pd.read_csv(self.raw_path, index_col=0, parse_dates=True)
        df_recent = df.tail(recent_window)
        
        close_data = df_recent[['Close']].values
        
        if len(close_data) < 62:
            print("Not enough data for incremental sequences.")
            return None, None
            
        scaled_data = scaler.transform(close_data)
        
        time_step = 60
        
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i])  # shape (60, 1)
            y.append(scaled_data[i, 0])
            
        if len(X) == 0:
            return None, None
            
        X = np.array(X)  # shape (n, 60, 1)
        y = np.array(y).reshape(-1, 1)
        
        return X, y


    def get_training_data(self):
        """
        Strict implementation of sliding window Close price sequences:
        - Train test split chronologically BEFORE scaling.
        - Fit scaler only on Train.
        """
        if os.path.exists(self.raw_path):
            df = pd.read_csv(self.raw_path, index_col=0, parse_dates=True)
        else:
            df = self.fetch_raw_data()
            
        if df is None or df.empty:
            return None, None, None, None, None, None, None

        close_data = df[['Close']].values
        
        # Split conceptually based on chronological time
        train_len = int(len(close_data) * 0.8)
        
        train_data = close_data[:train_len]
        
        # Setup scalers strictly on TRAIN data
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(train_data)
        
        # Transform full dataset
        scaled_data = feature_scaler.transform(close_data)
        target_scaler = feature_scaler
        
        time_step = 60
        
        X, y = [], []
        
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i])  # shape (60, 1)
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Adjust split for sequences length
        adjusted_train_len = train_len - time_step
        if adjusted_train_len < 0:
            adjusted_train_len = int(len(X) * 0.8)
            
        X_train, X_test = X[:adjusted_train_len], X[adjusted_train_len:]
        y_train, y_test = y[:adjusted_train_len], y[adjusted_train_len:]
        
        # Store close prices properly for test evaluate loop (last items)
        self._test_close_prices = close_data[adjusted_train_len + time_step:].flatten()
        self._all_close_prices = close_data.flatten()
        self._raw_returns = close_data.flatten() # using for raw prices wrapper later
        
        return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, close_data.flatten()

if __name__ == "__main__":
    # Test
    dp = DataProcessor("INFY.NS")
    dp.fetch_raw_data()
    result = dp.get_training_data()
    if result[0] is not None:
        X_tr, y_tr, X_te, y_te, sc, cp = result
        print(f"Train Shape: X={X_tr.shape}, y={y_tr.shape}")
        print(f"Test Shape: X={X_te.shape}, y={y_te.shape}")
        print(f"Close prices available: {len(cp)}")
