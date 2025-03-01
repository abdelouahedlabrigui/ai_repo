import base64
import datetime
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zipline.data.data_portal import DataPortal
from zipline.data import bundles
from zipline.utils.calendar_utils import get_calendar
from utils import *
from transformers import pipeline, set_seed
set_seed(42)
# from zipline.data.data_portal import OHLCV_FIELDS
# from matplotlib.dates import MonthLocator, date2num, DateFormatter
# from mpl_finance import candlestick2_ohlc

class DailyEquity:
    def __init__(self, ticker: str):
        self.bundle_data = bundles.load("quandl")
        self.end_data = pd.Timestamp("2014-01-01", tz="utc")
        self.ticker = ticker

    def equity_dataset(self):
        self.bundle_data.equity_daily_bar_reader.first_trading_day
        data_por = DataPortal(
            asset_finder=self.bundle_data.asset_finder,
            trading_calendar=get_calendar("NYSE"),
            first_trading_day=self.bundle_data.equity_daily_bar_reader.first_trading_day,
            equity_daily_reader=self.bundle_data.equity_daily_bar_reader
        )
        ticker_name = data_por.asset_finder.lookup_symbol(f"{self.ticker}", as_of_date=None)
        df = data_por.get_history_window(
            assets=[ticker_name], 
            end_dt=self.end_data, 
            bar_count=31 * 12, 
            frequency='1d', 
            data_frequency='daily', 
            field="open"
        )
        df['open'] = df[list(df.columns)[0]]
        df['close'] = data_por.get_history_window(
            assets=[ticker_name],
            end_dt=self.end_data,
            bar_count=31 * 12,
            frequency='1d',
            data_frequency='daily',
            field='close'
        )
        df['low'] = data_por.get_history_window(
            assets=[ticker_name],
            end_dt=self.end_data,
            bar_count=31 * 12,
            frequency='1d',
            data_frequency='daily',
            field='low'
        )
        df['high'] = data_por.get_history_window(
            assets=[ticker_name],
            end_dt=self.end_data,
            bar_count=31 * 12,
            frequency='1d',
            data_frequency='daily',
            field='high'
        )
        
        df.columns = ["Equity", 'open', 'close', 'low', 'high']
        # 1. Daily Range
        df['dailyRange'] = df['high'] - df['low']

        # 2. Midpoint
        df['midpoint'] = (df['high'] + df['low']) / 2

        # 3. Simple Daily Return
        df['dailyReturn'] = (df['close'] - df['open']) / df['open']

        # 4. Intraday Volatility (as a percentage)
        df['intradayVolatility'] = ((df['high'] - df['low']) / df['open']) * 100 
        
        describe_df = df.describe().T

        return df, describe_df
    
    def plot_price_equity(self):
        df, describe_df = self.equity_dataset()
        fig, ax1 = plt.subplots(figsize=(16, 8))
        fig.subplots_adjust(bottom=0.3)

        ax1.plot(df.index, df["open"], label="Open", color="blue", alpha=0.7, linewidth=1.5)
        ax1.plot(df.index, df["close"], label="Close", color="red", alpha=0.7, linewidth=1.5)
        ax1.plot(df.index, df["low"], label="Low", color="green", linestyle="dashed", alpha=0.7, linewidth=1.5)
        ax1.plot(df.index, df["high"], label="High", color="orange", linestyle="dashed", alpha=0.7, linewidth=1.5)

        
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.xticks(rotation=70)

        ax2 = ax1.twinx()
        ax2.plot(df.index, df["Equity"], label="Equity", color="purple", linestyle="dotted", alpha=0.8, linewidth=1.5)

        ax1.set_ylabel("Price")
        ax2.set_ylabel("Equity")
        ax1.legend(loc="upper left", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        first_plot_summary = (
            f"End date: {self.end_date}"
            f"Over the observed period, the **Equity {self.ticker}** price fluctuated between "  # Use equity_name
            f"{describe_df.loc['Open', 'min']:.2f} (Open), {describe_df.loc['High', 'max']:.2f} (High), "  # Include Open and High
            f"{describe_df.loc['Low', 'min']:.2f} (Low), and {describe_df.loc['Close', 'max']:.2f} (Close), "  # Include Low and Close
            f"with an average closing price of {describe_df.loc['Close', 'mean']:.2f}. "  # Focus on closing price
            f"The average open price was {describe_df.loc['Open', 'mean']:.2f}, the highest price reached was {describe_df.loc['High', 'max']:.2f}, "
            f"and the lowest price touched was {describe_df.loc['Low', 'min']:.2f}. "
            f"This indicates periods of notable price volatility, with significant swings between highs and lows."  # Stronger concluding statement
        )
        
        text_generation = TextGeneration(first_plot_summary).text_generation()
        translation = Translation(text_generation).translation_generator()

        data = {
            "Plot": encoded_image, "ESTranslation": translation, "Description": text_generation, "CreatedAT": str(datetime.datetime.now())
        }

        return data

    def plot_v1_technical_indicators(self):
        df, describe_df = self.equity_dataset()

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.subplots_adjust(bottom=0.3)

        ax.plot(df.index, df["dailyRange"], label="Daily Range", color="cyan", linewidth=1.5)
        ax.plot(df.index, df["midpoint"], label="Midpoint", linestyle="dashed", color="magenta", linewidth=1.5)
        ax.plot(df.index, df["dailyReturn"], label="Daily Return", linestyle="dotted", color="brown", linewidth=1.5)
        ax.plot(df.index, df["intradayVolatility"], label="Intraday Volatility", linestyle="dashdot", color="black", linewidth=1.5)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.xticks(rotation=70)

        ax.set_ylabel("Statistical Metrics")
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        second_plot_summary = (
            f"End date: {self.end_date}, Ticker: {self.ticker}"
            f"The **Daily Range** (difference between high and low prices) averaged {describe_df.loc['dailyRange', 'mean']:.2f}, "
            f"indicating periods of **price swings**. The **Midpoint** price, averaging {describe_df.loc['midpoint', 'mean']:.2f}, "
            f"helps visualize the central tendency of the stock price. **Daily Returns** fluctuated between "
            f"{describe_df.loc['dailyReturn', 'min']:.2%} and {describe_df.loc['dailyReturn', 'max']:.2%}, "
            f"highlighting gains and losses per day. Lastly, **Intraday Volatility**, with an average of {describe_df.loc['intradayVolatility', 'mean']:.2%}, "
            f"demonstrates how much price movement occurred within trading sessions."
        )

        text_generation = TextGeneration(second_plot_summary).text_generation()
        translation = Translation(text_generation).translation_generator()

        data = {
            "Plot": encoded_image, "ESTranslation": translation, "Description": text_generation, "CreatedAT": str(datetime.datetime.now())
        }

        return data

# equity = DailyEquity("2014-01-01", "TSLA")

class Translation:
    def __init__(self, text: str):
        self.text = text
    def translation_generator(self):
        try:
            if (not self.text.strip()) or (not self.title.strip()):
                raise ValueError("Input text is empty!")
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")
            outputs = translator(self.text, clean_up_tokenization_spaces=True, min_length=140)
            data = []
            translation_text = outputs[0]['translation_text']
            return translation_text
        except Exception as e:
            raise ValueError(f"{e}")

class TextGeneration:
    def __init__(self, text: str):
        self.text = text
        self.actor = "Trader"
        self.response = " so it means that for "
    def text_generation(self):
        try:
            if (not self.text.strip()) or (not self.title.strip()) or (not self.actor.strip()) or (not self.response.strip()):
                raise ValueError("Input text is empty!")
            generator = pipeline('text-generation')
            prompt = self.text + f"\n\n based on perspective of: {self.actor}:\n" + self.response
            outputs = generator(prompt, max_length=400)
            generated_text = outputs[0]["generated_text"]
            return generated_text
        except Exception as e:
            raise ValueError(f"{e}")