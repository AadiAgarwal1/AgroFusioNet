import pandas as pd

df = pd.read_csv("data/agri_multimodal_dataset_engineered.csv")

print("Dataset shape:")
print(df.shape)

print("\nAll columns:")
for col in df.columns:
    print(col)

print("\nColumn types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst rows:")
print(df.head())