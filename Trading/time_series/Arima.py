import base64
import datetime
import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimeSeriesAnalysis:
    def __init__(self, filename, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')
        self.data['Date'] = pd.to_datetime(self.data['Date'])   
        self.data.set_index('Date', inplace=True)
        self.target_column = 'Close'
        self.time_series = self.data[self.target_column]
        self.time_series_diff = self.time_series.diff().dropna()
        self.train = self.time_series[:int(0.8 * len(self.time_series))]
        self.test = self.time_series[int(0.8 * len(self.time_series)):]
        self.forecast, self.fitted_model = self.forecast_arima()
    
    def plot_time_series(self):
        plt.figure(figsize=(12, 6))
        self.time_series.plot(title=f'{self.target_column} Over Time')
        plt.xlabel('Date')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    
    # Check for Stationarity
    def adf_test(self, series, stationarity):
        result = adfuller(series)
        adfTestResult = {'Ticker': f'{self.ticker}', 'StartDate': f'{self.start_date}', 'EndDate': f'{self.end_date}', "Stationarity": f"{stationarity}",
                         'ADFStatistic': result[0], 'PValue': result[1], "CreatedAT": str(datetime.datetime.now())}
        critical_value = []
        for key, value in result[4].items():
            critical_value.append({'Ticker': f'{self.ticker}', 'StartDate': f'{self.start_date}', 'EndDate': f'{self.end_date}', 
                                   "Stationarity": f"{stationarity}", "CriticalValues": f"Key: ({key}) - Value: {value}", 
                                   "CreatedAT": str(datetime.datetime.now())})
        return critical_value, adfTestResult
    
    # Make the series Stationary (id needed)
    def differencing_the_series(self):
        self.time_series_diff.plot(title='Differenced Time Series', figsize=(12, 6))
        plt.xlabel('Date')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image

    # Fit ARIMA Model
    # def fit_arima_model(self):
    #     model = auto_arima(self.time_series, seasonal=False, trace=True, error_action='ignore', supress_warnings=True)
    #     return model.summary()
    
    def fit_arima_model(self):

        # Fit ARIMA model
        model = auto_arima(self.time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        
        # Extract summary
        summary = model.summary()

        # Extract coefficients and statistics from the summary
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
    
    def fit_arima_model_v1(self):

        # Fit ARIMA model
        model = auto_arima(self.time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        
        # Extract summary
        summary = model.summary()

        # Extract coefficients and statistics from the summary
        coef_df = pd.DataFrame(model.params(), columns=["Coefficients"])
        coef_df["StandardErrors"] = model.bse()
        coef_df.index.name = "Terms"

        # Extract AIC and BIC
        metrics = {"AIC": model.aic(), "BIC": model.bic()}

        return coef_df, metrics

    def plot_summary(self):
        # Get model results
        coef_df, metrics = self.fit_arima_model_v1()

        # Plot coefficients and standard errors
        plt.figure(figsize=(14, 6))

        # Bar plot of coefficients
        plt.subplot(1, 2, 1)
        sns.barplot(x=coef_df.index, y="Coefficients", data=coef_df, palette="viridis")
        plt.errorbar(coef_df.index, coef_df["Coefficients"], yerr=coef_df["StandardErrors"], fmt='o', color='red', capsize=5)
        plt.title("ARIMA Model Coefficients with Errors")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45)

        # Metrics visualization
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm")
        plt.title("Model Selection Metrics")
        plt.ylabel("Metric Value")
        
        plt.tight_layout()

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


    
    # Forecasting with ARIMA
    def forecast_arima(self):
        model = auto_arima(self.time_series, seasonal=False, trace=True, error_action='ignore', supress_warnings=True)
        order = model.order
        arima_model = ARIMA(self.train, order=order)  
        fitted_model = arima_model.fit()
        forecast = fitted_model.forecast(steps=len(self.test))
        return forecast, fitted_model
    
    def calculate_rmse(self):
        rmse = np.sqrt(mean_squared_error(self.test, self.forecast))
        return {'Ticker': f'{self.ticker}', 'StartDate': f'{self.start_date}', 'EndDate': f'{self.end_date}', "RMSE": rmse}

    def plot_arima_model_evaluation(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train, label='Training Data')
        plt.plot(self.test, label='Actual Data')
        plt.plot(self.test.index, self.forecast, label='Forecast', color="red")
        plt.legend()
        plt.title('ARIMA Model Evaluation')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image

    # Forecast Future Values
    def forecast_future_values(self):
        future_steps = 30
        future_forecast = self.fitted_model.get_forecast(steps=future_steps)
        forecast_index = pd.date_range(start=self.test.index[-1], periods=future_steps + 1, freq='B')[1:] 

        plt.figure(figsize=(12, 6))
        plt.plot(self.time_series, label="Historical Data")
        plt.plot(forecast_index, future_forecast.predicted_mean, label="Future Forecast", color="green")
        plt.fill_between(
            forecast_index, future_forecast.conf_int().iloc[:, 0], future_forecast.conf_int().iloc[:, 1], color="pink",
            alpha=0.3, label="Confidence Interval"
        )
        plt.legend()
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image

def main():
    filename = r"C:\Users\dell\Entrepreneurship\Engineering\Scripts\Trading\yahoo_finance\data\btc_stock_data_2022_01_to_12.csv" 
    ticker = "APPL"
    start_date = "2023-01-01" 
    end_date = "2023-10-27"
    arima = TimeSeriesAnalysis(filename, ticker, start_date, end_date)
        
    plot_time_series = arima.plot_time_series()[:20]
    differencing_the_series = arima.differencing_the_series()[:20]
    plot_arima_model_evaluation = arima.plot_arima_model_evaluation()[:20]
    forecast_future_values = arima.forecast_future_values()[:20]
    plot_summary = arima.plot_summary()[:20]
    
    print(plot_time_series)
    print("===========================")
    print(differencing_the_series)
    print("===========================")
    print(plot_arima_model_evaluation)
    print("===========================")
    print(forecast_future_values)
    print("===========================")
    print(plot_summary)
    print("===========================")

if __name__ == "__main__":
    main()