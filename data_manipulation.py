import pandas as pd
import numpy as np
import random

# Load the pre-filtered Germany-only data
file_path = "germany_epidemiology_data.csv"
df = pd.read_csv(file_path)

# Set seeds for reproducibility across runs
random.seed(42)
np.random.seed(42)

# Work on a copy to avoid side effects or unintended overwrites
df_noisy = df.copy()

# Inject missing values in ~25% of the dataframe to simulate data loss
missing_mask = np.random.rand(*df_noisy.shape) < 0.25
df_noisy = df_noisy.mask(missing_mask)

# Add high-variance noise to numerical columns to simulate corrupted measurements
numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    noise = np.random.normal(loc=0, scale=0.5, size=df_noisy[col].shape)
    df_noisy[col] = df_noisy[col] + noise

# Randomly drop 20% of the rows to mimic partial data ingestion
df_noisy = df_noisy.sample(frac=0.8, random_state=42).reset_index(drop=True)

# Persist the manipulated dataset for downstream testing
df_noisy.to_csv("germany_epidemiology_data_noisy_incomplete.csv", index=False)

print("Saved: 'germany_epidemiology_data_noisy_incomplete.csv'")