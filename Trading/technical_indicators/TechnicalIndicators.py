import base64
import io
from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
import datetime

class TechnicalIndicators:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.abs_path = r"C:\Users\dell\Entrepreneurship\Engineering\Scripts\Trading\yahoo_finance\data"
    def download(self):
        data_dict = {}
        for ticker in self.tickers:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            data_dict[ticker] = data
        return data_dict
    
    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window).mean()

    def calculate_ema(self, data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, data, window):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data):
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, data, window):
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        return upper_band, lower_band

    def calculate_stochastic_oscillator(self, data, window):
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        return k

    def save_technical_indicators(self):
        paths = []
        for ticker, data in self.download().items():
            data['Ticker'] = ticker
            data['SMA_20'] = self.calculate_sma(data, 20)
            data['EMA_20'] = self.calculate_ema(data, 20)
            data['RSI_14'] = self.calculate_rsi(data, 14)
            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data)
            data['Upper_BB'], data['Lower_BB'] = self.calculate_bollinger_bands(data, 20)
            data['Stochastic_K'] = self.calculate_stochastic_oscillator(data, 14)

            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            # now = str(datetime.datetime.now()).replace(":", "_").replace(" ", "_")
            path = f"{self.abs_path}\{ticker}_technical_indicators.csv"
            data = data.iloc[1:]
            data.to_csv(path)
            paths.append({'Path': path})
        return paths

    def plot_close_price_moving_avg_rsi_macd(self, dataset, ticker):
        data = pd.read_csv(dataset, encoding='utf-8')
        plt.figure(figsize=(14, 8))

        # Plot Close price and Moving Averages
        plt.subplot(2, 1, 1)
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['SMA_20'], label='SMA (20)', color='green')
        plt.plot(data['EMA_20'], label='EMA (20)', color='red')
        plt.title(f'{ticker} - Close Price & Moving Averages')
        plt.legend()

        # Plot RSI
        plt.subplot(2, 2, 3)
        plt.plot(data['RSI_14'], label='RSI (14)', color='purple')
        plt.axhline(70, color='red', linestyle='--', linewidth=1)
        plt.axhline(30, color='green', linestyle='--', linewidth=1)
        plt.title('RSI')
        plt.legend()

        # Plot MACD
        plt.subplot(2, 2, 4)
        plt.plot(data['MACD'], label='MACD', color='blue')
        plt.plot(data['MACD_Signal'], label='Signal Line', color='orange')
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title('MACD')
        plt.legend()

        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image

if __name__ == "__main__":
    tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    indictors = TechnicalIndicators(tickers, start_date, end_date)

    indictors.save_technical_indicators()    
    # indictors.plot_close_price_moving_avg_rsi_macd(csv, "AAPL")