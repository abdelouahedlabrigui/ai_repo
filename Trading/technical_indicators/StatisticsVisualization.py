import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

class StatisticsVisualization:
    def __init__(self):
        pass

    def fetch_summary_stats(self, ticker, year):
        url = f"http://localhost:5082/api/StockPricesApi/summary-stats-using-sql-client?ticker={ticker}&year={year}"
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            print(data.head())
            return data
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None
    
    def plot_close_summary(self, ticker, year):
        data = self.fetch_summary_stats(ticker, year)
        close_stats = ["avgClose", "minClose", "maxClose", "medianClose", "percentile25Close", "percentile75Close"]
        values = data[close_stats].iloc[0]

        plt.figure(figsize=(10,6))
        sns.barplot(x=close_stats, y=values, palette='Blues_d')
        plt.title(f"{ticker} Closing Price Statistics ({year})", fontsize=14)
        plt.ylabel("Close Price ($)")
        plt.xlabel("Statistics")
        plt.xticks(rotation=45)
        plt.show()

    def plot_volume_summary(self, ticker, year):
        data = self.fetch_summary_stats(ticker, year)
        close_stats = ["minVolume", "maxVolume", "medianVolume", "percentile25Volume", "percentile75Volume"]
        values = data[close_stats].iloc[0]

        plt.figure(figsize=(10,6))
        sns.barplot(x=close_stats, y=values, palette='Blues_d')
        plt.title(f"{ticker} Volume Price Statistics ({year})", fontsize=14)
        plt.ylabel("Volume")
        plt.xlabel("Statistics")
        plt.xticks(rotation=45)
        plt.show()

    def plot_close_percentiles(self, ticker, year):
        data = self.fetch_summary_stats(ticker, year)
        percentiles = ['percentile25Close', "medianClose", 'percentile75Close']
        values = data[percentiles].iloc[0]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=percentiles, y=values, palette="Oranges_d")
        plt.title(f"{ticker} Closing Price Percentiles ({year})", fontsize=14)
        plt.ylabel("Close Price ($)")
        plt.xlabel("Percentiles")
        plt.show()



    
plot = StatisticsVisualization()
plot.plot_close_summary("AAPL", "2023")
plot.plot_volume_summary("AAPL", "2023")
plot.plot_close_percentiles("AAPL", "2023")