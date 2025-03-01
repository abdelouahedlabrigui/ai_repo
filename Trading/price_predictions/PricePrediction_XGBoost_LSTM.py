import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import ReduceLROnPlateau

class PricePrediction_XGBoost_LSTM:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.horizon = 30
        self.data = pd.concat([self.fetch_stock_data(ticker) for ticker in self.tickers])
        self.param_grid = {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }

    def fetch_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")[["Close"]]
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        return df
    
    def create_features(self, df, lag_days=10):
        df = df.copy()
        for lag in range(1, lag_days + 1):
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

        df["Target"] = df["Close"].shift(-self.horizon)
        return df.dropna()
    
    def final_df(self):
        processed_data = []
        for ticker in self.tickers:
            df_ticker = self.data[self.data["Ticker"] == ticker]
            df_ticker = self.create_features(df_ticker)
            processed_data.append(df_ticker)
        df_final = pd.concat(processed_data)
        return df_final
    
    def variables(self):
        df_final = self.final_df()
        features = [col for col in df_final.columns if "lag" in col]
        X = df_final[features]
        y = df_final["Target"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        split = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]               

        return X, y, X_train, X_test, y_train, y_test
    def xgb_variables(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, objective="reg:squarederror")
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test) 
        return y_pred_xgb
    
    def xgb_errors(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        y_pred_xgb = self.xgb_variables()
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        return mae_xgb, rmse_xgb
    
    def lstm_variables(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        lstm_model = Sequential([
            LSTM(100, activation="relu", return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.2), LSTM(50, activation="relu"), Dropout(0.2), Dense(1)
        ])

        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

        y_pred_lstm = lstm_model.predict(X_test_lstm)
        return y_pred_lstm, X_train_lstm, X_test_lstm
        
    def lstm_errors(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        y_pred_lstm, X_train_lstm, X_test_lstm = self.lstm_variables()
        mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
        return mae_lstm, rmse_lstm
    
    def xgb_lstm_stock_prediction(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        y_pred_xgb = self.xgb_variables()
        y_pred_lstm, X_train_lstm, X_test_lstm = self.lstm_variables()

        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Actual Prices", color="black", linestyle="dashed")
        plt.plot(y_test.index, y_pred_xgb, label="XGBoost Predictions", color="blue")
        plt.plot(y_test.index, y_pred_lstm, label="LSTM Predictions", color="red")

        plt.title("Stock Price Prediction - XGBoost vs LSTM")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def xgb_tuned_variables(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
        grid_search = GridSearchCV(xgb_model, self.param_grid, cv=3, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_xgb_model = grid_search.best_estimator_
        y_pred_xgb_tuned = best_xgb_model.predict(X_test)
        return grid_search, y_pred_xgb_tuned
    
    def xgb_tuned_errors(self):
        X, y, X_train, X_test, y_train, y_test = self.variables()
        grid_search, y_pred_xgb_tuned = self.xgb_tuned_variables()
        best_params = grid_search.best_estimator_
        mae = mean_absolute_error(y_test, y_pred_xgb_tuned)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_tuned))
        return best_params, mae, rmse 
    
    def lstm_tuned_variables(self):
        y_pred_lstm, X_train_lstm, X_test_lstm = self.lstm_variables()
        X, y, X_train, X_test, y_train, y_test = self.variables()
        lr_scheduler = ReduceLROnPlateau(monitor="loss", patience=5, factor=0.5, min_lr=0.0001)
        lstm_model = Sequential([
            LSTM(128, activation="relu", return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.3), LSTM(64, activation="relu"), Dropout(0.3), Dense(1)
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[lr_scheduler])
        y_pred_lstm_tunned = lstm_model.predict(X_test_lstm)
        return y_pred_lstm_tunned
    
    def lstm_tunned_errors(self):
        y_pred_lstm_tunned = self.lstm_tuned_variables()
        X, y, X_train, X_test, y_train, y_test = self.variables()
        mae = mean_absolute_error(y_test, y_pred_lstm_tunned)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm_tunned))
        return mae, rmse
    
    def optimized_xgb_lstm_stock_prediction(self):
        y_pred_lstm_tunned = self.lstm_tuned_variables()
        X, y, X_train, X_test, y_train, y_test = self.variables()
        grid_search, y_pred_xgb_tuned = self.xgb_tuned_variables()
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Actual Prices", color="black", linestyle="dashed")
        plt.plot(y_test.index, y_pred_xgb_tuned, label="XGBoost Optimized", color="blue")
        plt.plot(y_test.index, y_pred_lstm_tunned, label="LSTM Optimized", color="red")

        plt.title("Stock Price Prediction - Optimized XGBoost vs LSTM")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    price = PricePrediction_XGBoost_LSTM(tickers, start_date, end_date)
    mae_xgb, rmse_xgb = price.xgb_errors()
    mae_lstm, rmse_lstm = price.lstm_errors()
    best_params, mae_xgb_tunned, rmse_xgb_tunned = price.xgb_tuned_errors()
    grid_search, y_pred_xgb_tuned = price.xgb_tuned_variables()
    mae_lstm_tunned, rmse_lstm_tunned = price.lstm_tunned_errors()
    print("Best XGBoost Params:", grid_search.best_params_)
    print(f"XGBoost - MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
    print(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}")
    print(f"XGBoost Tunned - MAE: {mae_xgb_tunned:.2f}, RMSE: {rmse_xgb_tunned:.2f}")
    print(f"LSTM Tunned - MAE: {mae_lstm_tunned:.2f}, RMSE: {rmse_lstm_tunned:.2f}")
    price.xgb_lstm_stock_prediction()
    price.optimized_xgb_lstm_stock_prediction()