import yfinance as yf
import pandas as pd
import datetime

class DownloadDataset:
    def __init__(self, company, start_date, end_date, filename):
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.filename = filename
    
    def get_yahoo_finance_data(self):
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
            datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.datetime.strptime(self.end_date, '%Y-%m-%d')

            # Download the data
            data = yf.download(self.company, start=self.start_date, end=self.end_date)

            # Check if data was successfully downloaded
            if data.empty:
                print(f"No data found for {self.company} between {self.start_date} and {self.end_date}.")
                return

            # Save to CSV
            data.to_csv(self.filename)
            return f"Data for {self.company} saved to {self.filename}"

        except ValueError as e:
            print(f"Invalid date format: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


# Example usage:
# company_symbol = "AAPL"  # Replace with the desired company symbol
# start_date = "2023-01-01"  # Replace with the desired start date
# end_date = "2023-10-27"  # Replace with the desired end date
# output_filename = "apple_stock_data.csv" #Optional filename

# get_yahoo_finance_data(company_symbol, start_date, end_date, output_filename)