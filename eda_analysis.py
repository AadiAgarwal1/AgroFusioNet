import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\nLoading dataset...")

df = pd.read_csv("data/agri_multimodal_dataset_engineered.csv")

print("\nDataset shape:", df.shape)

# ------------------------------------------------
# 🔥 FIX YIELD UNITS (SAFE + CONTROLLED)
# ------------------------------------------------

def fix_yield_units(df):
    df = df.copy()

    yield_cols = ["yield", "yield_last_year", "yield_3yr_avg"]

    for col in yield_cols:
        if col in df.columns:
            max_val = df[col].max()

            # realistic yield should be < 20–30 ton/hectare
            if max_val > 100:
                print(f"Scaling {col} (max={max_val}) → dividing by 1000")
                df[col] = df[col] / 1000

    return df


df = fix_yield_units(df)

# ------------------------------------------------
# BASIC INFO
# ------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())

print("\nColumn types:")
print(df.dtypes)

# ------------------------------------------------
# MISSING VALUES
# ------------------------------------------------

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing Values:")
print(missing)

# ------------------------------------------------
# BASIC STATS
# ------------------------------------------------

print("\nYield Stats:\n", df["yield"].describe())
print("\nRainfall Stats:\n", df["annual_rainfall"].describe())
print("\nNDVI Stats:\n", df["ndvi"].describe())
print("\nAESI Stats:\n", df["aesi"].describe())

# ------------------------------------------------
# UNIQUE COUNTS
# ------------------------------------------------

print("\nUnique districts:", df["district"].nunique())
print("Unique crops:", df["crop"].nunique())
print("Year range:", df["year"].min(), "-", df["year"].max())

# ------------------------------------------------
# CREATE FIGURE FOLDER
# ------------------------------------------------

os.makedirs("figures", exist_ok=True)

# ------------------------------------------------
# 1. TEMPORAL TREND
# ------------------------------------------------

plt.figure(figsize=(10,5))
df.groupby("year")["yield"].mean().plot()
plt.title("Average Yield Over Time")
plt.ylabel("Yield (ton/hectare)")
plt.xlabel("Year")
plt.savefig("figures/yield_trend.png")
plt.close()

# ------------------------------------------------
# 2. NDVI vs YIELD
# ------------------------------------------------

plt.figure()
sns.scatterplot(x=df["ndvi"], y=df["yield"], alpha=0.3)
plt.title("NDVI vs Yield")
plt.xlabel("NDVI")
plt.ylabel("Yield")
plt.savefig("figures/ndvi_vs_yield.png")
plt.close()

# ------------------------------------------------
# 3. CORRELATION MATRIX
# ------------------------------------------------

important_features = [
    "yield", "annual_rainfall", "ndvi",
    "n", "p", "k",
    "aesi", "snii",
    "rainfall_variability", "ndvi_trend"
]

plt.figure(figsize=(10,8))
corr = df[important_features].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Important Feature Correlation")
plt.savefig("figures/correlation_matrix.png")
plt.close()

# ------------------------------------------------
# 4. CROP DISTRIBUTION
# ------------------------------------------------

plt.figure(figsize=(12,5))
df["crop"].value_counts().head(20).plot(kind="bar")
plt.title("Top 20 Crops")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.savefig("figures/crop_distribution.png")
plt.close()

# ------------------------------------------------
# 5. RAINFALL vs YIELD
# ------------------------------------------------

plt.figure()
sns.scatterplot(x=df["annual_rainfall"], y=df["yield"], alpha=0.3)
plt.title("Rainfall vs Yield")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Yield")
plt.savefig("figures/rainfall_vs_yield.png")
plt.close()

print("\n✅ EDA completed successfully")
print("Figures saved in /figures/")