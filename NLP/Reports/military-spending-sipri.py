import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils import *
from transformers import pipeline, set_seed


class TextGeneration:
    def __init__(self, text: str, actor: str, response: str):
        self.text = text
        self.actor = actor
        self.response = response
    def text_generation(self):
        try:
            generator = pipeline('text-generation')
            prompt = self.text + f"\n\n based on perspective of: {self.actor}:\n" + self.response
            outputs = generator(prompt, max_length=300)
            generated_text = outputs[0]["generated_text"]
            return str(generated_text).split(f"based on perspective of: {self.actor}")[1]
        except Exception as e:
            raise ValueError(f"{e}")

df = pd.read_csv(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\Data\military-spending-sipri.csv")
# List of countries to analyze
countries = [
    "Canada", "China", "European Union (SIPRI)", "France", "Germany",
    "Japan", "Israel", "Italy", "Spain", "United Kingdom", "United States"
]

sequence_length = 5  # Number of past years to use for prediction
future_years = 10  # Number of years to predict

# Create a figure for plotting
fig = plt.figure(figsize=(15, 10))

sequence_length = 5  # Number of past years to use for prediction
future_years = 10  # Number of years to predict

for country in countries:
    # Filter data for the country
    country_df = df[df["Entity"] == country].sort_values(by="Year")
    
    # Ensure we have enough data
    if len(country_df) < sequence_length:
        print(f"Skipping {country}: Not enough data ({len(country_df)} years).")
        continue
    
    # Normalize the military expenditure
    scaler = MinMaxScaler()
    country_df["Military expenditure (normalized)"] = scaler.fit_transform(
        country_df[["Military expenditure (constant US$)"]]
    )
    
    # Prepare sequences
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        return np.array(sequences), np.array(labels)
    
    country_data = country_df["Military expenditure (normalized)"].values
    X, y = create_sequences(country_data, sequence_length)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Train LSTM model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    
    # Predict future values
    last_sequence = X[-1]  # Last known sequence
    predictions = []
    
    for _ in range(future_years):
        next_value = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
        predictions.append(next_value[0][0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_value
    
    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Get years for plotting
    past_years = country_df["Year"].values
    future_years_range = np.arange(past_years[-1] + 1, past_years[-1] + 1 + future_years)
    
    # Enhanced Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(past_years, country_df["Military expenditure (constant US$)"], label="Actual", marker="o", linewidth=2)
    plt.plot(future_years_range, predictions, linestyle="--", label="Predicted", marker="x", color="red", linewidth=2)
    
    # Add shaded confidence interval (simplified, assuming constant error for demonstration)
    pred_mean = predictions.flatten()
    pred_std = np.std(country_df["Military expenditure (constant US$)"])
    plt.fill_between(future_years_range, pred_mean - pred_std, pred_mean + pred_std, color='red', alpha=0.2, label='Confidence Interval (approx.)')
    
    # Customize plot
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Military Expenditure (constant US$)", fontsize=12)
    plt.title(f"Military Expenditure Prediction - {country}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Dynamic Interpretation with Date Precision
    last_actual = country_df["Military expenditure (constant US$)"].iloc[-1]
    last_actual_year = country_df["Year"].iloc[-1]
    first_predicted = predictions[0][0]
    first_predicted_year = future_years_range[0]
    
    change = first_predicted - last_actual
    percentage_change = (change / last_actual) * 100 if last_actual != 0 else 0
    
    interpretation = f"For {country}, the last recorded expenditure in {last_actual_year} was ${last_actual:,.2f}. The model predicts an expenditure of ${first_predicted:,.2f} for {first_predicted_year}, representing a change of ${change:,.2f} ({percentage_change:,.2f}%)."
    
    if percentage_change > 0:
        interpretation += " This suggests a potential increase in military expenditure."
    elif percentage_change < 0:
        interpretation += " This suggests a potential decrease in military expenditure."
    else:
        interpretation += " This suggests the military expenditure is expected to remain relatively stable."
    
    with open(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\interpretations\notes.txt", 'a', encoding='utf-8') as f:
        f.writelines(f"Country: {country}\n")
        f.writelines(f"{interpretation}\n\n")
    # plt.savefig(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\plots\expenditure_" + f"{country}.png", dpi=fig.dpi)
