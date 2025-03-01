from mistralai import Mistral
import base64
import io
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig

class Prompts:
    def __init__(self, prompt):
        self.prompt = prompt
        self.key = r'C:\Users\dell\Entrepreneurship\Engineering\Scripts\GenerativeAI\Mistral\private-key.txt'
        self.read_key = str(open(f'{self.key}', encoding='utf-8').read())
        self.api_key = f"{self.read_key}"
        self.client = Mistral(api_key=self.api_key)
    def ask_mistral(self):
        try:
            prompt = f"May you interpret this result: {self.prompt}"
            chat_response = self.client.chat.complete(
                model = 'mistral-small-latest',
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ]
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}" 

class VolumeBasedFeatures:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        self.volume_features = {}
        self.ma_window = 20
        self.indicators = {}
        self.volume_data_v1 = self.data[(self.tickers[0], "Volume")].dropna()
        self.volume_data_v2 = {ticker: self.data[(ticker, "Volume")].dropna() for ticker in self.tickers}
        self.features_volume = self.compute_volume_features()

    def compute_volume_features(self):
        """Compute volume-based indicators for each stock."""
        volume_features = {}
        for ticker in self.tickers:
            volume = self.data[(ticker, "Volume")]
            ma_volume = volume.rolling(window=self.ma_window).mean()
            volume_change = volume.pct_change() * 100  # Convert to percentage
            volume_features[ticker] = {
                "Moving Average Volume": ma_volume,
                "Volume Change (%)": volume_change
            }
        return volume_features
    
    def generate_paragraph_stock_volume_vs_moving_average(self, ticker):
        max_volume = self.data[(ticker, 'Volume')].max()
        min_volume = self.data[(ticker, 'Volume')].min()
        avg_volume = self.features_volume[ticker]['Moving Average Volume'].mean()
        current_volume = self.data[(ticker, 'Volume')].iloc[-1]
        volume_change = (current_volume - avg_volume) / avg_volume * 100
        volume_rank = self.data[(ticker, "Volume")].rank(pct=True).iloc[-1] * 100
        ma5_volume = self.data[(ticker, "Volume")].rolling(window=5).mean().iloc[-1]
        ma5_volume_change = (current_volume - ma5_volume) / ma5_volume * 100

        return (f"The {ticker} stock's trading volume has fluctuated significantly, ranging from a peak of {max_volume:.0f} to a trough of {min_volume:.0f}. "
            f"The average moving volume is {avg_volume:.0f}, providing a baseline for comparison.  Currently, the volume stands at {current_volume:.0f}, "
            f"representing a {volume_change:.0f}% change from the average. This places the current volume at the {volume_rank:.0f}th percentile, "
            f"indicating where it stands relative to all historical volume data.  A shorter-term moving average, such as the 5-day MA at {ma5_volume:.0f}, "
            f"shows a {ma5_volume_change:.0f}% change, offering a more recent perspective on volume trends.  When the actual volume surpasses the moving average, "
            f"especially significantly as indicated by the percentile and percentage change, it often suggests heightened market activity and can be a precursor to potential price movements.")
    
    def generate_paragraph_daily_volume_change(self, ticker):
        """Generate a paragraph for daily volume changes."""
        volume_changes = self.features_volume[ticker]["Volume Change (%)"]
        avg_change = volume_changes.mean()
        max_change = volume_changes.max()
        min_change = volume_changes.min()
        current_change = volume_changes.iloc[-1]
        std_dev = volume_changes.std()
        volatility = std_dev / abs(avg_change) if avg_change != 0 else float('inf')
        positive_days = volume_changes[volume_changes > 0].count()
        negative_days = volume_changes[volume_changes < 0].count()
        total_days = len(volume_changes)
        positive_percentage = (positive_days / total_days) * 100 if total_days > 0 else 0
        recent_changes = volume_changes.tail(10)
        avg_recent_change = recent_changes.mean()

        return (f"The daily trading volume for {ticker} shows considerable fluctuation, with an average percentage change of {avg_change:.2f}%.  "
            f"Peak increases have reached {max_change:.2f}%, while the most significant drops have been {min_change:.2f}%. "
            f"Currently, the volume change is {current_change:.2f}%.  The standard deviation of these changes is {std_dev:.2f}%, indicating a volatility of {volatility:.2f}. "
            f"Over the observed period, positive volume changes occurred on {positive_percentage:.0f}% of trading days, suggesting a general tendency towards increased volume. "
            f"Looking at the most recent 10 days, the average volume change is {avg_recent_change:.2f}%, providing a more current perspective.  "
            f"Large positive changes often signal heightened investor interest and potential price volatility, whereas sharp declines can reflect market corrections or decreased participation.  "
            f"The volatility metric helps to quantify the overall risk associated with volume fluctuations.")
    
    def generate_paragraph_volume_surges_vs_price(self, ticker):
        """Generate a paragraph analyzing volume surges vs. price changes."""
        close_prices = self.data[(ticker, "Close")]
        volume = self.data[(ticker, "Volume")]

        price_change = close_prices.pct_change().mean() * 100  # Convert to percentage
        volume_surge = volume.pct_change().max() * 100

        # Calculate correlation between volume change and price change
        correlation = volume.pct_change().corr(close_prices.pct_change())

        # Calculate average volume during price increases vs. decreases (example)
        price_increases = close_prices.pct_change() > 0
        price_decreases = close_prices.pct_change() < 0
        avg_volume_increase = volume[price_increases].mean()
        avg_volume_decrease = volume[price_decreases].mean()

        # Calculate the number of times a volume surge coincided with a significant price change (e.g. > 2%)
        significant_price_change_threshold = 2  # Adjust as needed
        volume_change = volume.pct_change() * 100
        significant_price_changes = abs(close_prices.pct_change() * 100) > significant_price_change_threshold
        coinciding_surges = volume_change[significant_price_changes & (volume_change > volume.quantile(0.90))].count() # Number of times volume is in top 10% when a big price move happens. You can change the 0.9 to a different percentile.
        total_significant_changes = significant_price_changes.sum()
        surge_coincidence_rate = (coinciding_surges / total_significant_changes) * 100 if total_significant_changes > 0 else 0

        return (f"For {ticker}, the relationship between price fluctuations and trading volume surges is complex but often informative. "
                f"On average, price changes {price_change:.2f}% daily, while extreme volume spikes can reach {volume_surge:.2f}%. "
                f"The correlation between daily volume change and price change is {correlation:.2f}, indicating the strength of their linear relationship. "
                f"During periods of price increases, the average volume was {avg_volume_increase:.0f}, compared to {avg_volume_decrease:.0f} during price decreases. "
                f"Furthermore, volume surges (defined as being in the top 10% of volume changes) coincided with significant price movements (greater than {significant_price_change_threshold}%) {surge_coincidence_rate:.0f}% of the time. "
                f"These observations suggest that sudden increases in volume can indeed precede or accompany price movements, potentially offering insights into market sentiment and future price direction. However, it's crucial to note that correlation does not imply causation, and other factors can also influence price changes.")
    
    def volume_features_dataset(self):
        for ticker in self.tickers:
            volume = self.data[(ticker, "Volume")]
            avg_volume = volume.mean()
            volume_change = volume.pct_change() * 100
            ma_volume = volume.rolling(window=self.ma_window).mean()

            self.volume_features[ticker] = {
                "Average Volume": avg_volume, "Volume Change (%)": volume_change, "Moving Average Volume": ma_volume,
            }
        df_features = pd.DataFrame({ticker: features['Average Volume'] for ticker, features in self.volume_features.items()},
                                   index=['Average Volume'])
        df_features = df_features.T
        return df_features
    
    def plot_stock_volume_vs_moving_average(self):
        info = []
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.data.index, self.data[(ticker, "Volume")], label=f"{ticker} Volume", alpha=0.6)
            plt.plot(self.data.index, self.volume_features[ticker]['Moving Average Volume'], linestyle="--", label=f"{ticker} MA {self.ma_window}")
            info.append({
                f'{ticker} Stock Volume vs. Moving Average': self.generate_paragraph_stock_volume_vs_moving_average(ticker),
            })
        plt.title("Stock Volume vs. Moving Average")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {info}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, info, str(promptResult)
        

    def plot_daily_volume_change(self):
        plt.figure(figsize=(12, 6))
        info = []
        for ticker in self.tickers:
            plt.plot(self.data.index, self.volume_features[ticker]['Volume Change (%)'], label=f"{ticker} Volume Change")
            info.append({
                f'{ticker} Daily Volume Change (%)': self.generate_paragraph_daily_volume_change(ticker),
            })
        plt.title("Daily Volume Change (%)")
        plt.xlabel("Date")
        plt.ylabel("Percentage Change")
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {info}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, info, str(promptResult)

    def compute_RSI(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_MACD(self, series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal
    
    def compute_indicators_for_each_stock(self):
        for ticker in self.tickers:
            close_price = self.data[(ticker, "Close")]
            volume = self.data[(ticker, "Volume")]

            ma_volume = volume.rolling(window=self.ma_window).mean()
            rsi = self.compute_RSI(close_price)
            macd, signal = self.compute_MACD(close_price)
            self.indicators[ticker] = {
                "Close": close_price, "Volume": volume, "MA_Volume": ma_volume, "RSI": rsi, "MACD": macd, "Signal": signal
            }
        return self.indicators
    
    def plot_volume_surges_vs_price(self):
        info = []
        indicators = self.compute_indicators_for_each_stock()
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.data.index, indicators[ticker]['Close'], label=f"{ticker} Price")
            plt.scatter(self.data.index, indicators[ticker]['Volume'], s=10, label=f"{ticker} Volume", alpha=0.5)
            info.append({
                f'{ticker} Stock Price vs. Volume Surges': self.generate_paragraph_volume_surges_vs_price(ticker),
            })
        plt.title("Stock Price vs. Volume Surges")
        plt.xlabel("Date")
        plt.ylabel("Price & Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {info}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, info, str(promptResult)
        

    def plot_rsi(self):
        indicators = self.compute_indicators_for_each_stock()
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.data.index, indicators[ticker]['RSI'], label=f"{ticker} RSI")
        plt.axhline(70, linestyle="--", color="red", alpha=0.7)
        plt.axhline(30, linestyle="--", color="green", alpha=0.7)
        plt.title("RSI for Stocks")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        

        recent_rsi = {ticker: indicators[ticker]['RSI'].iloc[-1] for ticker in self.tickers}
        overbought = [ticker for ticker, rsi in recent_rsi.items() if rsi > 70]
        oversold = [ticker for ticker, rsi in recent_rsi.items() if rsi < 30]

        # Calculate RSI change over a period (e.g., last 14 days)
        rsi_change = {}
        for ticker in self.tickers:
            rsi_values = indicators[ticker]['RSI']
            rsi_change[ticker] = rsi_values.iloc[-1] - rsi_values.iloc[-14]  # Change over 14 days. Adjust as needed

        # Calculate the average RSI for each ticker
        average_rsi = {ticker: indicators[ticker]['RSI'].mean() for ticker in self.tickers}

        # Calculate the number of times the RSI has crossed above 70 or below 30 in a given period (e.g., last year)
        overbought_crossings = {}
        oversold_crossings = {}
        for ticker in self.tickers:
            rsi_values = indicators[ticker]['RSI']
            overbought_crossings[ticker] = ((rsi_values > 70).rolling(window=2).sum() == 2).sum() # Number of times it goes above 70, then stays above 70
            oversold_crossings[ticker] = ((rsi_values < 30).rolling(window=2).sum() == 2).sum() # Number of times it goes below 30, then stays below 30

        paragraph = f"The Relative Strength Index (RSI) highlights the momentum of each stock. "

        if overbought:
            overbought_str = ", ".join(overbought)
            paragraph += f"Currently, {overbought_str} are in the overbought region (>70), signaling potential pullbacks. "
            for ticker in overbought:
                paragraph += f"{ticker}: RSI = {recent_rsi[ticker]:.2f}, Change (14 days) = {rsi_change[ticker]:.2f}, Average RSI = {average_rsi[ticker]:.2f}, Overbought Crossings (Last Year) = {overbought_crossings[ticker]}. "

        if oversold:
            oversold_str = ", ".join(oversold)
            paragraph += f"While {oversold_str} are in the oversold region (<30), indicating possible buying opportunities. "
            for ticker in oversold:
                paragraph += f"{ticker}: RSI = {recent_rsi[ticker]:.2f}, Change (14 days) = {rsi_change[ticker]:.2f}, Average RSI = {average_rsi[ticker]:.2f}, Oversold Crossings (Last Year) = {oversold_crossings[ticker]}. "

        if not overbought and not oversold:
            paragraph += "Currently, no stocks are in the overbought or oversold regions. "
            for ticker in self.tickers:
                paragraph += f"{ticker}: RSI = {recent_rsi[ticker]:.2f}, Change (14 days) = {rsi_change[ticker]:.2f}, Average RSI = {average_rsi[ticker]:.2f}, Overbought Crossings (Last Year) = {overbought_crossings[ticker]}, Oversold Crossings (Last Year) = {oversold_crossings[ticker]}. "
        
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {paragraph}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, paragraph, str(promptResult)
        

    def plot_macd(self):
        indicators = self.compute_indicators_for_each_stock()
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.data.index, indicators[ticker]['MACD'], label=f"{ticker} MACD")
            plt.plot(self.data.index, indicators[ticker]['Signal'], linestyle="--", label=f"{ticker} Signal")
        plt.axhline(0, linestyle="--", color="red", alpha=0.7)
        plt.title("MACD for Stocks")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        recent_macd = {ticker: indicators[ticker]['MACD'].iloc[-1] for ticker in self.tickers}
        recent_signal = {ticker: indicators[ticker]['Signal'].iloc[-1] for ticker in self.tickers}

        bullish = [ticker for ticker in self.tickers if recent_macd[ticker] > recent_signal[ticker]]
        bearish = [ticker for ticker in self.tickers if recent_macd[ticker] < recent_signal[ticker]]

        # Calculate MACD histogram
        macd_histogram = {}
        for ticker in self.tickers:
            macd_histogram[ticker] = indicators[ticker]['MACD'] - indicators[ticker]['Signal']

        # Calculate MACD crossover history (how many times MACD crossed above/below signal line)
        bullish_crossovers = {}
        bearish_crossovers = {}
        for ticker in self.tickers:
            macd = indicators[ticker]['MACD']
            signal = indicators[ticker]['Signal']
            bullish_crossovers[ticker] = ((macd > signal).rolling(window=2).sum() == 2).sum() # Count when MACD crosses above signal and stays above
            bearish_crossovers[ticker] = ((macd < signal).rolling(window=2).sum() == 2).sum() # Count when MACD crosses below signal and stays below

        # Calculate the distance between MACD and signal line
        macd_distance = {ticker: abs(recent_macd[ticker] - recent_signal[ticker]) for ticker in self.tickers}

        paragraph = f"The Moving Average Convergence Divergence (MACD) indicator offers insights into trend direction and momentum. "

        if bullish:
            bullish_str = ", ".join(bullish)
            paragraph += f"Currently, {bullish_str} show bullish signals as their MACD line is above the signal line. "
            for ticker in bullish:
                paragraph += (f"{ticker}: MACD = {recent_macd[ticker]:.2f}, Signal = {recent_signal[ticker]:.2f}, "
                            f"Histogram = {macd_histogram[ticker].iloc[-1]:.2f}, Bullish Crossovers (Last Year) = {bullish_crossovers[ticker]}, "
                            f"MACD-Signal Distance = {macd_distance[ticker]:.2f}. ")

        if bearish:
            bearish_str = ", ".join(bearish)
            paragraph += f"While {bearish_str} exhibit bearish trends with MACD below the signal line. "
            for ticker in bearish:
                paragraph += (f"{ticker}: MACD = {recent_macd[ticker]:.2f}, Signal = {recent_signal[ticker]:.2f}, "
                            f"Histogram = {macd_histogram[ticker].iloc[-1]:.2f}, Bearish Crossovers (Last Year) = {bearish_crossovers[ticker]}, "
                            f"MACD-Signal Distance = {macd_distance[ticker]:.2f}. ")

        if not bullish and not bearish:
            paragraph += "Currently, no stocks exhibit clear bullish or bearish MACD signals. "
            for ticker in self.tickers:
                paragraph += (f"{ticker}: MACD = {recent_macd[ticker]:.2f}, Signal = {recent_signal[ticker]:.2f}, "
                            f"Histogram = {macd_histogram[ticker].iloc[-1]:.2f}, Bullish Crossovers (Last Year) = {bullish_crossovers[ticker]}, Bearish Crossovers (Last Year) = {bearish_crossovers[ticker]}, "
                            f"MACD-Signal Distance = {macd_distance[ticker]:.2f}. ")
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {paragraph}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, paragraph, str(promptResult)
        

    def forecast_arima(self, series, steps=30):
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    
    def forecast_lstm(self, series, steps=30):
        series = series.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        series_scaled = scaler.fit_transform(series)

        X, y = [], []
        lookback = 30
        for i in range(len(series_scaled) - lookback):
            X.append(series_scaled[i:i+lookback])
            y.append(series_scaled[i+lookback])

        X, y = np.array(X), np.array(y)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(50), Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=1)

        future_inputs = series_scaled[-lookback:].reshape(1, lookback, 1)
        future_forecast = []
        for _ in range(steps):
            pred = model.predict(future_inputs)[0,0]
            future_forecast.append(pred)
            future_inputs = np.roll(future_inputs, -1)
            future_inputs[0, -1, 0] = pred

        future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1)).flatten()
        return future_forecast
    
    def plot_volume_prediction(self):
        arima_forecast = self.forecast_arima(self.volume_data_v1, steps=30)
        lstm_forecast = self.forecast_lstm(self.volume_data_v1, steps=30)

        plt.figure(figsize=(12, 6))
        plt.plot(self.volume_data_v1[-300:], label="Actual Volume")
        plt.plot(pd.date_range(self.volume_data_v1.index[-1], periods=30, freq='D'), arima_forecast, label="ARIMA Forecast", linestyle="--")
        plt.plot(pd.date_range(self.volume_data_v1.index[-1], periods=30, freq="D"), lstm_forecast, label="LSTM Forecast", linestyle="--")

        plt.title(f"Volume Prediction for {self.tickers[0]}")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        last_actual_volume = self.volume_data_v1.iloc[-1]
        last_arima_pred = arima_forecast.iloc[-1] if isinstance(arima_forecast, pd.Series) else arima_forecast[-1]
        last_lstm_pred = lstm_forecast.iloc[-1] if isinstance(lstm_forecast, pd.Series) else lstm_forecast[-1]

        trend = "increasing" if last_arima_pred > last_actual_volume else "decreasing"
        consistency = "consistent" if abs(last_arima_pred - last_lstm_pred) < 0.05 * last_actual_volume else "divergent"

        # Calculate percentage change predictions
        arima_pct_change = (last_arima_pred - last_actual_volume) / last_actual_volume * 100 if last_actual_volume != 0 else 0
        lstm_pct_change = (last_lstm_pred - last_actual_volume) / last_actual_volume * 100 if last_actual_volume != 0 else 0

        # Calculate the difference between the two predictions (absolute and percentage)
        prediction_diff = abs(last_arima_pred - last_lstm_pred)
        prediction_diff_pct = (prediction_diff / last_actual_volume) * 100 if last_actual_volume != 0 else 0

        # Calculate the historical average volume (e.g., over the last year)
        historical_avg_volume = self.volume_data_v1.mean()  # Or a more specific period

        # Calculate the ratio of the last actual volume to the historical average
        volume_ratio = last_actual_volume / historical_avg_volume if historical_avg_volume != 0 else 0


        paragraph = f"The volume prediction analysis uses ARIMA and LSTM models to forecast future trading volume trends. "

        paragraph += (f"The last actual volume was {last_actual_volume:.0f}. "
                    f"The ARIMA model predicts {last_arima_pred:.0f} ({arima_pct_change:.2f}% change), "
                    f"while the LSTM model forecasts {last_lstm_pred:.0f} ({lstm_pct_change:.2f}% change). ")


        paragraph += f"Both models indicate a {trend} trend. "

        paragraph += f"Their forecasts are {consistency}, with a difference of {prediction_diff:.0f} ({prediction_diff_pct:.2f}%). "

        paragraph += (f"The current volume represents {volume_ratio:.2f} times the historical average volume of {historical_avg_volume:.0f}. "
                    f"This gives context to the magnitude of the current volume. ")

        paragraph += ("This level of agreement/disagreement suggests either strong confidence or differing outlooks "
                    "on volume movement in the upcoming period.  A larger divergence could imply greater uncertainty in the predictions. "
                    "The percentage change values provide a clearer picture of the predicted volume's relative movement.  "
                    "Comparing the current volume to its historical average provides a longer-term perspective.")
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {paragraph}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, paragraph, str(promptResult)


    def dataset_v2(self):
        df = pd.DataFrame(self.volume_data_v2)
        df.index.name = "Date"
        df.fillna(method="ffill")
        scaler = MinMaxScaler(feature_range=(0,1))
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=self.tickers, index=df.index)        
        return df, scaler, df_scaled
    
    def plot_original_volume_data(self):
        df, scaler, df_scaled = self.dataset_v2()
        df.plot(figsize=(12, 6), title='Stock Volume Data')
        plt.ylabel("Volume")
        plt.show()

    def create_lstm_data(self, series, lookback=30):
        X, y = [], []
        for i in range(len(series) - lookback):
            X.append(series[i:i+lookback])
            y.append(series[i+lookback])
        return np.array(X), np.array(y)
    
    def build_lstm(self):
        lookback= 30
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)), LSTM(50), Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model
    
    def plot_lstm_volume_predictions_for_multiple_stocks(self):
        df, scaler, df_scaled = self.dataset_v2()
        lookback = 30
        X_train, y_train, X_test, y_test = {}, {}, {}, {}
        for ticker in self.tickers:
            series = df_scaled[ticker].values.reshape(-1, 1)
            X, y = self.create_lstm_data(series, lookback)
            X_train[ticker], X_test[ticker] = X[:-30], X[-30:]
            y_train[ticker], y_test[ticker] = y[:-30], y[-30:]

        forecasts = {}
        for ticker in self.tickers:
            model = self.build_lstm()
            model.fit(X_train[ticker], y_train[ticker], epochs=20, batch_size=16, verbose=1)
            forecast = model.predict(X_test[ticker])
            forecasts[ticker] = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(df.index[-30:], forecast[ticker], label=f"{ticker} Forecast")
        
        plt.title("LSTM Volume Predictions for Multiple Stocks")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def prepare_prophet_data(self, series):
        df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values})
        return df_prophet
    
    def generate_prophet_summary(self, prophet_forecasts):
        summary = ""
        for ticker, forecast in prophet_forecasts.items():

            if isinstance(forecast, pd.DataFrame):
                first_pred = forecast['yhat'].iloc[0]
                last_pred = forecast['yhat'].iloc[-1]
                average_forecast = forecast['yhat'].mean()
                std_dev_forecast = forecast['yhat'].std()
                median_forecast = forecast['yhat'].median()
                max_forecast = forecast['yhat'].max()
                min_forecast = forecast['yhat'].min()

            elif isinstance(forecast, np.ndarray):
                first_pred = forecast[0]
                last_pred = forecast[-1]
                average_forecast = forecast.mean()
                std_dev_forecast = forecast.std()
                median_forecast = np.median(forecast)  # Use np.median() for NumPy
                max_forecast = forecast.max()
                min_forecast = forecast.min()

            else:
                raise TypeError(f"Unexpected forecast type: {type(forecast)}. Expecting pandas.DataFrame or numpy.ndarray")

            trend = "increasing" if last_pred > first_pred else "decreasing"
            percentage_change = ((last_pred - first_pred) / first_pred) * 100 if first_pred != 0 else 0

            summary += (
                f"For {ticker}, the forecasted trading volume over the next 30 days shows an {trend} trend. "
                f"The expected volume ranges from a low of {min_forecast:,.0f} to a high of {max_forecast:,.0f}, "
                f"starting at approximately {first_pred:,.0f} and ending at around {last_pred:,.0f} ({percentage_change:,.2f}% change). "
                f"The average forecasted volume is {average_forecast:,.0f}, with a standard deviation of {std_dev_forecast:,.0f} and a median of {median_forecast:,.0f}. "
                f"This suggests potential market activity shifts, which investors should monitor closely.  "
                f"The range and standard deviation give an indication of the forecast's uncertainty. "
                f"Comparing the average and median can highlight any skew in the forecasted distribution.\n\n"
            )
        return summary.strip()

    def plot_prophet_volume_prediction_for_multiple_stocks(self):
        df, scaler, df_scaled = self.dataset_v2()
        prophet_forecasts = {}
        future_dates = pd.date_range(df.index[-1], periods=30, freq='D')

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            prophet_df = self.prepare_prophet_data(df[ticker])
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            prophet_forecasts[ticker] = forecast['yhat'][-30:].values
            plt.plot(future_dates, forecast['yhat'][-30:], label=f"{ticker} Prophet Forecast")

        plt.title("Prophet Volume Predictions for Multiple Stocks")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        summary = self.generate_prophet_summary(prophet_forecasts=prophet_forecasts)
        promptStr = f"From these tickers: {self.tickers}, based on this start: {self.start_date} and end: {self.end_date} dates; may interpret this result: {summary}."
        promptResult = Prompts(promptStr).ask_mistral()
        return encoded_image, summary, str(promptResult)
        

    def convert_to_torch(self, series):
        tensor = torch.tensor(series.values, dtype=torch.float32).view(1, -1, 1)
        return tensor
    
    def plot_transformer_volume_predictions_for_multiple_stocks(self):
        df, scaler, df_scaled = self.dataset_v2()
        future_dates = pd.date_range(df.index[-1], periods=30, freq='D')
        lookback = 30
        config = TimeSeriesTransformerConfig(
            input_size=1, num_attention_heads=2, num_hidden_layers=2, hidden_size=32
        )
        model = TimeSeriesTransformerModel(config)
        transformer_forecasts = {}
        for ticker in self.tickers:
            input_tensor = self.convert_to_torch(df[ticker][-lookback:])
            with torch.no_grad():
                forecast = model(input_tensor).squeeze().numpy()
            transformer_forecasts[ticker] = forecast[-30:]
        
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(future_dates, transformer_forecasts[ticker], label=f"{ticker} Transformer Forecast")
        
        plt.title("Transformer Volume Predictions for Multiple Stocks")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    volume = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
    df = volume.volume_features_dataset()
    # print(df.head())
    # volume.plot_stock_volume_vs_moving_average()
    # volume.plot_daily_volume_change()
    # print(volume.compute_indicators_for_each_stock())
    # volume.plot_volume_surges_vs_price()
    # volume.plot_rsi()
    # volume.plot_macd()
    # volume.plot_volume_prediction()
    # volume.plot_original_volume_data()
    # volume.plot_lstm_volume_predictions_for_multiple_stocks()
    # volume.plot_prophet_volume_prediction_for_multiple_stocks()
    # volume.plot_transformer_volume_predictions_for_multiple_stocks()