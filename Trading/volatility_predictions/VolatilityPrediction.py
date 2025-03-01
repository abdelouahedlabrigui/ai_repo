import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from arch import arch_model

class VolatilityPrediction:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)["Close"]

    def variables(self):
        returns = np.log(self.data / self.data.shift(1))
        returns.dropna(inplace=True)
        volatility = returns.rolling(window=30).std() * np.sqrt(252)
        features = returns.rolling(window=30).mean()
        features["Volatility"] = volatility
        features.dropna(inplace=True)
        X = features.drop(columns=["Volatility"])
        y = features["Volatility"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        return X, y, X_train, X_test, y_train, y_test, returns, features
    
    def predicting_future_volatility(self, ticker):
        X, y, X_train, X_test, y_train, y_test, returns, features = self.variables()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        garch_model = arch_model(returns[ticker], col="Garch", p=1, q=1)
        garch_results = garch_model.fit(disp="off")

        garch_forecast = garch_results.forecast(horizon=5)
        garch_pred = np.sqrt(garch_forecast.variance.iloc[-1])

        plt.figure(figsize=(12, 6))
        plt.plot(features.index[-len(y_test):], y_test, label="Actual Volatility", color="blue")
        plt.plot(features.index[-len(y_test):], y_pred, label="RF Predicted Volatility", color="red")
        plt.title("Volatility Prediction using Random Forest")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()

        print("\nGARCH Predicted Volatility for Next 5 Days:")
        print(garch_pred)


tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
start_date: str = '2023-01-01'
end_date: str = '2023-12-31'
volatility = VolatilityPrediction(tickers, start_date, end_date)
volatility.predicting_future_volatility("AAPL")