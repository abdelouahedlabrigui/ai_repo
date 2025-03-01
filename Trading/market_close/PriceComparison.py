import base64
import datetime
import io
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

class PriceComparison:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
    
    def get_yahoo_finance_data(self, company, start_date, end_date):
        """
        Fetches historical stock data from Yahoo Finance and saves it to a CSV file.

        Args:
            company: The stock ticker symbol (e.g., "AAPL", "MSFT").
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.
            filename: The name of the CSV file to save the data to. Defaults to "stock_data.csv".
        """

        try:
            # Validate date format
            datetime.datetime.strptime(start_date, '%Y-%m-%d')
            datetime.datetime.strptime(end_date, '%Y-%m-%d')

            # Download the data
            data = yf.download(company, start=start_date, end=end_date)

            # Check if data was successfully downloaded
            if data.empty:
                print(f"No data found for {company} between {start_date} and {end_date}.")
                return
            return data
        except ValueError as e:
            print(f"Invalid date format: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def plot_close_tickers_comparison(self):
        tickers = self.tickers
        
        plt.figure(figsize=(12, 6))
        
        for ticker in tickers:
            data = self.get_yahoo_finance_data(str(ticker).strip(), self.start_date, self.end_date)
            df = pd.DataFrame(data)
            
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            plt.plot(df.index, df["Close"], label=f"{ticker} - Close")


        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")
        plt.title("Stock Price Comparison (2019)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    def plot_Open_tickers_comparison(self):
        tickers = self.tickers
        
        plt.figure(figsize=(12, 6))
        
        for ticker in tickers:
            data = self.get_yahoo_finance_data(ticker, self.start_date, self.end_date)
            df = pd.DataFrame(data)
            
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            plt.plot(df.index, df["Open"], label=f"{ticker} - Open")


        plt.xlabel("Date")
        plt.ylabel("Open Price (USD)")
        plt.title("Stock Price Comparison (2019)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    def plot_Low_tickers_comparison(self):
        tickers = self.tickers
        
        plt.figure(figsize=(12, 6))
        
        for ticker in tickers:
            data = self.get_yahoo_finance_data(ticker, self.start_date, self.end_date)
            df = pd.DataFrame(data)
            
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            plt.plot(df.index, df["Low"], label=f"{ticker} - Low")


        plt.xlabel("Date")
        plt.ylabel("Low Price (USD)")
        plt.title("Stock Price Comparison (2019)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    def plot_high_tickers_comparison(self):
        tickers = self.tickers
        
        plt.figure(figsize=(12, 6))
        
        for ticker in tickers:
            data = self.get_yahoo_finance_data(ticker, self.start_date, self.end_date)
            df = pd.DataFrame(data)
            
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            plt.plot(df.index, df["High"], label=f"{ticker} - High")


        plt.xlabel("Date")
        plt.ylabel("High Price (USD)")
        plt.title("Stock Price Comparison (2019)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    def plot_volume_tickers_comparison(self):
        tickers = self.tickers
        
        plt.figure(figsize=(12, 6))
        
        for ticker in tickers:
            data = self.get_yahoo_finance_data(ticker, self.start_date, self.end_date)
            df = pd.DataFrame(data)
            
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            plt.plot(df.index, df["Volume"], label=f"{ticker} - Volume")


        plt.xlabel("Date")
        plt.ylabel("Volume Price (USD)")
        plt.title("Stock Price Comparison (2019)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')    
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image