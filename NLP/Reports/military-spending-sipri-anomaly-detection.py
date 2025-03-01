import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

data = pd.read_csv(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\Data\military-spending-sipri.csv")
entity_data = data.groupby("Entity")["Military expenditure (constant US$)"].apply(list)

# Parameters
timesteps = 5  # Window size
scaler = MinMaxScaler()

# Prepare sequences
sequences = []
for entity, expenditures in entity_data.items():
    exp = np.array(expenditures).reshape(-1, 1)
    scaled_exp = scaler.fit_transform(exp)
    for i in range(len(scaled_exp) - timesteps + 1):
        sequences.append(scaled_exp[i:i + timesteps])
sequences = np.array(sequences)

# Define model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(timesteps, 1), return_sequences=False),
    RepeatVector(timesteps),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(sequences, sequences, epochs=50, batch_size=32, validation_split=0.1)

# Reconstruction and anomaly detection
reconstructed = model.predict(sequences)
errors = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
threshold = np.mean(errors) + 2 * np.std(errors)
anomalies = errors > threshold

X = data[["Military expenditure (constant US$)"]].values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.05)  # nu controls the fraction of outliers
predictions = ocsvm.fit_predict(X_scaled)

# Anomalies
data["Anomaly"] = predictions == -1
anomalies = data[data["Anomaly"]]
#print(anomalies[["Entity", "Year", "Military expenditure (constant US$)"]])

countries = [
    "China", "United States", "Russia"
]

fig = plt.figure(figsize=(12, 8))  # Adjust figure size for better clarity
for country in countries:
    entity_df = data[data["Entity"] == country]

    # Plot military spending
    plt.plot(entity_df["Year"], entity_df["Military expenditure (constant US$)"], label=f"{country} Spending", alpha=0.7, linewidth=1.5)

    # Plot anomalies
    anomaly_years = entity_df[entity_df["Anomaly"]]["Year"]
    anomaly_values = entity_df[entity_df["Anomaly"]]["Military expenditure (constant US$)"]
    plt.scatter(anomaly_years, anomaly_values, color='red', label=f"{country} Anomalies" if not anomaly_years.empty else "") # only label anomalies if there are any.

plt.xlabel("Year", fontsize=14)
plt.ylabel("Military Expenditure (constant US$)", fontsize=14)
plt.title("Military Spending with Anomalies for Selected Countries", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6) #Add grid for improved readability.
plt.tight_layout() #Improve plot layout.
plt.savefig(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\plots\anomalies\anomalies_by_selected_countries.png", dpi=fig.dpi)

# Generate Dynamic Interpretation
interpretation_text = "Dynamic Interpretation:\n"
for country in countries:
    entity_df = data[data["Entity"] == country]
    anomaly_count = entity_df["Anomaly"].sum()
    latest_year = entity_df["Year"].max()
    latest_spending = entity_df[entity_df["Year"] == latest_year]["Military expenditure (constant US$)"].values[0] if not entity_df[entity_df["Year"] == latest_year].empty else "N/A"

    interpretation_text += f"- {country}:\n"
    if anomaly_count > 0:
        interpretation_text += f"  - Experienced {anomaly_count} anomaly/anomalies in military spending.\n"
    else:
        interpretation_text += f"  - No significant anomalies detected in military spending.\n"

    interpretation_text += f"  - Latest recorded military spending ({latest_year}): ${latest_spending:,.2f} (constant US$).\n"

    # Basic Trend Analysis (Optional, can be improved)
    if len(entity_df) > 2:
        last_two_years = entity_df.tail(2)["Military expenditure (constant US$)"].values
        if len(last_two_years) == 2:
            change = last_two_years[1] - last_two_years[0]
            if change > 0:
                interpretation_text += "  - Recent trend: Increasing military spending.\n"
            elif change < 0:
                interpretation_text += "  - Recent trend: Decreasing military spending.\n"
            else:
                interpretation_text += "  - Recent trend: Stable military spending.\n"
    else:
        interpretation_text += "  - Insufficient data for recent trend analysis.\n"

# Final Print Statement with Interpretation
print(interpretation_text)
with open(r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\interpretations\notes.txt', "a", encoding='utf-8') as f:
    f.writelines("Military Spending with Anomalies for Selected Countries\n")
    f.writelines(interpretation_text)

countries = [
    "Canada", "China", "European Union (SIPRI)", "France", "Germany",
    "Japan", "Israel", "Italy", "Spain", "United Kingdom", "United States"
]

year = 2020

year_df = data[data["Year"] == year]

# Calculate the mean military expenditure for the given year
mean_expenditure = year_df["Military expenditure (constant US$)"].mean()

# Filter countries with expenditure above the mean
high_spending_countries = year_df[
    (year_df["Military expenditure (constant US$)"] > mean_expenditure)
]

# Sort the countries by expenditure for better visualization
high_spending_countries = high_spending_countries.sort_values(by="Military expenditure (constant US$)", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=high_spending_countries["Entity"], y=high_spending_countries["Military expenditure (constant US$)"], hue=high_spending_countries["Entity"],
            palette="viridis", legend=False) # remove legend.
plt.xticks(rotation=90)
plt.xlabel("Entity", fontsize=14)
plt.ylabel("Military Expenditure (constant US$)", fontsize=14)
plt.title(f"Military Spending Above Mean in {year}", fontsize=16, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\plots\anomalies\military_spending_above_mean.png", dpi=fig.dpi)

# Generate Interpretation
interpretation_text = f"Interpretation of Military Spending Above Mean in {year}:\n\n"
if high_spending_countries.empty:
    interpretation_text += f"No countries exceeded the mean military expenditure (${mean_expenditure:,.2f}) in {year}.\n"
else:
    interpretation_text += f"In {year}, the following countries had military expenditures exceeding the mean (${mean_expenditure:,.2f}):\n\n"
    for index, row in high_spending_countries.iterrows():
        country = row["Entity"]
        expenditure = row["Military expenditure (constant US$)"]
        interpretation_text += f"- {country}: ${expenditure:,.2f}\n"

    # Find the highest spender
    highest_spender = high_spending_countries.iloc[0]
    highest_country = highest_spender["Entity"]
    highest_expenditure = highest_spender["Military expenditure (constant US$)"]
    interpretation_text += f"\n{highest_country} had the highest military expenditure at ${highest_expenditure:,.2f}.\n"

# Print Interpretation
print(interpretation_text)
with open(r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Reports\military-spending-sipri\interpretations\notes.txt', "a", encoding='utf-8') as f:
    f.writelines(f"Military Spending Above Mean in {year}")
    f.writelines(interpretation_text)