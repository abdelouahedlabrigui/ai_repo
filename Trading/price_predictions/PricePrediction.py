import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PricePrediction:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.horizon = 30
        self.data = pd.concat([self.fetch_stock_data(ticker) for ticker in self.tickers])
    
    def fetch_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")
        df = df[["Close", "Volume"]]
        df["Ticker"] = ticker
        df.reset_index(inplace=True)
        return df
    
    def create_features(self, df, lag_days=5):
        df = df.copy()
        for lag in range(1, lag_days + 1):
            df[f"Close_lag_{lag}"] = df['Close'].shift(lag)
        df['Moving_Avg'] = df['Close'].rolling(window=lag_days).mean()
        df['Target'] = df['Close'].shift(-self.horizon)
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
        features = [col for col in df_final.columns if "lag" in col or "Moving_Avg" in col]
        X = df_final[features]
        y = df_final["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return X, y, X_train, X_test, y_train, y_test, y_pred, model
    
    def errors(self):
        X, y, X_train, X_test, y_train, y_test, y_pred, model = self.variables()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return mae, rmse
    
    def predict_future_prices(self, model, last_known_data, horizon):
        future_prices = []
        current_input = last_known_data.copy()
        for _ in range(self.horizon):
            pred = model.predict(current_input.reshape(1, -1))[0]
            future_prices.append(pred)

            current_input = np.roll(current_input, -1)
            current_input[-1] = pred

        return future_prices
    
    def price_prediction(self):
        X, y, X_train, X_test, y_train, y_test, y_pred, model = self.variables()
        last_known_features = X.iloc[-1].values
        future_predictions = self.predict_future_prices(model, last_known_features, self.horizon)
        future_dates = pd.date_range(start=self.data["Date"].max(), periods=self.horizon + 1, freq="D")[1:]
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.data[self.data["Ticker"] == f"{ticker}"]["Date"], self.data[self.data["Ticker"] == f"{ticker}"]["Close"], label=f"{ticker} Actual Prices")
            plt.plot(future_dates, future_predictions, linestyle="--", label=f"{ticker} Predcited Prices")
        plt.title(f"{ticker} Price Prediction (30 days ahead)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    
# prediction = PricePrediction(["AAPL", ""])
if __name__ == "__main__":
    tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    price = PricePrediction(tickers, start_date, end_date)
    mae, rmse = price.errors()
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    price.price_prediction()