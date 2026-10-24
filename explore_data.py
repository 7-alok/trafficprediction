
import pandas as pd
from pathlib import Path

file_path = Path("data/Metro_Interstate_Traffic_Volume.csv")

df = pd.read_csv(file_path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\nDataset loaded successfully!")
