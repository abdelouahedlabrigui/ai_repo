from matplotlib import pyplot as plt
import seaborn as sns
import requests
from utils import *
import pandas as pd

class Classification:
    def __init__(self, text_file: str, start_str: str, end_str: str):
        self.text_file = text_file
        self.start_str = start_str
        self.end_str = end_str
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    def request_dataframe(self,):
        endpoint = f"http://127.0.0.1:5005/topics/extract_text_between?text_file={self.text_file}&start_str={self.start_str}&end_str={self.end_str}"
        response = requests.get(endpoint, headers=self.headers)

        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return pd.DataFrame()
        
        
    def plot_class_distribution(self):
        df = self.request_dataframe()
        if df.empty:
            print("No data available for plotting.")
            return
        
        plt.figure(figsize=(8, 5))
        sns.barplot(
            y=df["label_name"].value_counts(ascending=True).index,
            x=df["label_name"].value_counts(ascending=True).values,
            palette="viridis"
        )
        plt.title("Frequency of Classes", fontsize=14, fontweight="bold")
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Topic", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.show()

    def plot_words_per_text(self):
        df = self.request_dataframe()

        if df.empty:
            print("No data available for plotting.")
            return
        
        df["Words Per Text"] = df["text"].str.split().apply(len)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="label_name", y="Words Per Text", data=df, palette="coolwarm", showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Words Per Text by Topic", fontsize=14, fontweight="bold")
        plt.xlabel("Topic", fontsize=12)
        plt.ylabel("Words Per Text", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()
    

text_file = "C:/Users/dell/Entrepreneurship/Books/NewsPapers/Biosafety_in_Microbiological_and_Biomedical_Laboratories_6th_Edition.txt"
start_str = "The ongoing practice of biological risk assessment is the foundation of safe"
end_str = "knowledge and experience may justify altering these safeguards."

# Example Usage
if __name__ == "__main__":
    classifier = Classification(text_file=text_file, start_str=start_str, end_str=end_str)
    df = classifier.request_dataframe()
    print(df.head())
