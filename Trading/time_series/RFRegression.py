import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

class RFRegression:
    def __init__(self, filename, ticker, start_date, end_date):
        self.filename = filename
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')    
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)    
        self.features = ['Open', 'High', 'Low', 'Volume']
        self.target = 'Close'
        self.data['Close_lag1'] = self.data['Close'].shift(1)
        self.data['Close_lag2'] = self.data['Close'].shift(2)
        self.data['Close_lag3'] = self.data['Close'].shift(3)

        self.data.dropna(inplace=True)

    def features_target(self):
        X = self.data[self.features + ['Close_lag1', 'Close_lag2', 'Close_lag3']]
        y = self.data[self.target]
        return X, y
    def training_and_testing(self):
        X, y = self.features_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    def train_model_and_predict(self):
        X_train, X_test, y_train, y_test = self.training_and_testing()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred
    
    def model_evaluation(self):
        model, y_pred = self.train_model_and_predict()
        X_train, X_test, y_train, y_test = self.training_and_testing()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'MeanSquaredError': mse, 'RSquared': r2}
    
    def predictions_visualization(self):
        model, y_pred = self.train_model_and_predict()
        X_train, X_test, y_train, y_test = self.training_and_testing()
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='orange', linestyle="--")
        plt.title(f'{self.ticker} Stock Price Predictions - Random Forest Regression')
        plt.xlabel('Date')
        plt.ylabel('Close Stock Price')
        plt.legend()
        plt.show()

    def predict_future(self):
        model, y_pred = self.train_model_and_predict()
        steps = 30
        last_row = self.data.iloc[-1]
        predictions = []

        for _ in range(steps):
            next_input = last_row[self.features + ['Close_lag1', 'Close_lag2', 'Close_lag3']].values.reshape(1, -1)
            next_prediction = model.predict(next_input)[0]
            predictions.append(next_prediction)
            last_row['Close_lag3'] = last_row['Close_lag2']
            last_row['Close_lag2'] = last_row['Close_lag1']
            last_row['Close_lag1'] = next_prediction

        return predictions
    
    def plot_future(self):
        predictions = self.predict_future()
        last_date = self.data.index[-1]
        future_index = pd.date_range(start=last_date, periods=31, freq='D')[1:]
        plt.figure(figsize=(12, 6))
        plt.plot(future_index, predictions, label='Predicted Prices', color='green')
        plt.title(f'{self.ticker} Future Stock Price Predictions - Random Forest Regression')
        plt.xlabel('Date')
        plt.ylabel('Close Stock Price')
        plt.legend()
        plt.show()

def main():
    filename = r"C:\Users\dell\Entrepreneurship\Engineering\Scripts\Trading\yahoo_finance\data\btc_stock_data_2022_01_to_12.csv" 
    ticker = "BTCM"
    start_date = "2022-01-03" 
    end_date = "2022-11-30"
    regression = RFRegression(filename, ticker, start_date, end_date)
    regression.plot_future()
    regression.predictions_visualization()
    indicators = json.dumps(regression.model_evaluation(), indent=4)
    print(indicators)

if __name__ == "__main__":
    main()