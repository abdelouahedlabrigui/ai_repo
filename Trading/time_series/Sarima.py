import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import seaborn as sns
import io
import base64

import json

class SARIMAForecaster:
    def __init__(self, filename, ticker, start_date, end_date):
        self.filename = filename
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        self.data = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')
        self.data['Date'] = pd.to_datetime(self.data['Date'])   
        self.data.set_index('Date', inplace=True)
        self.target_column = 'Close'
        self.time_series = self.data[self.target_column]
        self.seasonal_period = 12

    def fit_sarima_model(self):
        model = auto_arima(self.time_series, seasonal=True, m=self.seasonal_period, trace=True, error_action='ignore', suppress_warnings=True)
        coef_df = pd.DataFrame(model.params(), columns=["Coefficients"])
        coef_df["StandardErrors"] = model.bse()
        coef_df.index.name = "Terms"
        metrics = {"AIC": model.aic(), "BIC": model.bic()}
        return model, coef_df, metrics
    
    def fit_sarima_model_v1(self):
        model = auto_arima(self.time_series, seasonal=True, m=self.seasonal_period, trace=True, error_action='ignore', suppress_warnings=True)
        coef_df = pd.DataFrame(model.params(), columns=["Coefficients"])
        coef_df["StandardErrors"] = model.bse()
        coef_df.index.name = "Terms"
        coef_df.reset_index(inplace=True)
        data = []
        for term, coefficient, standard_error in zip(
            list(coef_df['Terms']),
            list(coef_df['Coefficients']),
            list(coef_df['StandardErrors']),

        ):
            data.append({
                "Ticker": self.ticker,
                "StartDate": self.start_date,
                "EndDate": self.end_date,
                "Terms": term,
                "Coefficients": coefficient,
                "StandardErrors": standard_error,
                "CreatedAT": str(datetime.datetime.now()),  
            })

        coef_df = pd.DataFrame(data)
        # Extract AIC and BIC
        metrics = {
            "Ticker": self.ticker,
            "StartDate": self.start_date,
            "EndDate": self.end_date,
            "AIC": model.aic(), 
            "BIC": model.bic(),
            "CreatedAT": str(datetime.datetime.now()),  
        }
        
        return coef_df, metrics
    
    def prepare_time_series(self):
        if not isinstance(self.time_series.index, pd.DatetimeIndex):
            raise ValueError("Time series index must be a DatetimeIndex")
        if not hasattr(self.time_series.index, 'freq') or self.time_series.index.freq is None:
            inferred_freq = pd.infer_freq(self.time_series.index)
            if inferred_freq is not None:
                self.time_series.index = pd.date_range(
                    start=self.time_series.index[0], 
                    periods=len(self.time_series), 
                    freq=inferred_freq
                )
            else:
                print("Frequence couldn't be inferred. Filling missing dates with interpolation.")
                self.time_series = self.time_series.asfreq('D')
                self.time_series.interpolate(method='time', inplace=True)
    
    def forecast(self, steps):
        self.prepare_time_series()
        model_, _, _ = self.fit_sarima_model()
        forecast = model_.predict(n_periods=steps)
        last_date = self.time_series.index[-1]
        future_index = pd.date_range(start=last_date, periods=steps + 1, freq=self.time_series.index.freq)[1:]
        
        return pd.Series(forecast, index=future_index, name="Forecast")
    
    def plot_forecast(self, steps):
        forecast = self.forecast(steps)
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_series, label="Actual Data", color="blue")
        plt.plot(forecast, label="Forecast", linestyle="--", color="orange")
        plt.title(f"{self.ticker} Stock Price SARIMA Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        # plt.show()
        # Save the plot to a PNG image in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)

        # Encode the image in base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the buffer and plot
        buffer.close()
        plt.close()
        return encoded_image

    def plot_summary(self):
        model, coef_df, metrics = self.fit_sarima_model()
        coef_df.index = coef_df.index.astype(str)
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(x=coef_df.index, y="Coefficients", data=coef_df, palette="viridis")
        plt.errorbar(coef_df.index, coef_df["Coefficients"], yerr=coef_df["StandardErrors"], fmt='o', color='red', capsize=5)
        plt.title("SARIMA Model Coefficients with Errors")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45)

        metrics = {key: float(value) for key, value in metrics.items()}
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm")
        plt.title("Model Selection Metrics")
        plt.ylabel("Metric Value")

        plt.tight_layout()
        # plt.show()
        # Save the plot to a PNG image in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)

        # Encode the image in base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the buffer and plot
        buffer.close()
        plt.close()
        return encoded_image


def main():
    filename = r"C:\Users\dell\Entrepreneurship\Engineering\Scripts\Trading\yahoo_finance\data\btc_stock_data_2022_01_to_12.csv" 
    ticker = "APPL"
    start_date = "2023-01-01" 
    end_date = "2023-10-27"
    sarima = SARIMAForecaster(filename, ticker, start_date, end_date)
    print(sarima.time_series.index)
    steps = 30
    model, coef_df, metrics = sarima.fit_sarima_model()
    print(coef_df)
    print()
    print(json.dumps(metrics, indent=4))
    sarima.plot_summary()
    sarima.plot_forecast(steps)

if __name__ == "__main__":
    main()