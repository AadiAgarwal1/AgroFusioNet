import pandas as pd
import numpy as np


# ------------------------------------------------
# Load dataset
# ------------------------------------------------

DATA_PATH = "data/agri_multimodal_dataset_final.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# ------------------------------------------------
# Soil balance ratios
# ------------------------------------------------

print("Adding soil balance features...")

df["np_ratio"] = df["n"] / (df["p"] + 1e-6)
df["nk_ratio"] = df["n"] / (df["k"] + 1e-6)
df["pk_ratio"] = df["p"] / (df["k"] + 1e-6)

df["npk_balance"] = (
    (df["n"] / df["n"].mean()) +
    (df["p"] / df["p"].mean()) +
    (df["k"] / df["k"].mean())
) / 3


# ------------------------------------------------
# Rainfall variability
# ------------------------------------------------

print("Adding rainfall variability...")

rain_cols = [
"rain_jan","rain_feb","rain_mar","rain_apr",
"rain_may","rain_jun","rain_jul","rain_aug",
"rain_sep","rain_oct","rain_nov","rain_dec"
]

df["rainfall_variability"] = df[rain_cols].std(axis=1)


# ------------------------------------------------
# Crop-season rainfall
# ------------------------------------------------

print("Adding crop season rainfall...")

# Kharif (Monsoon crops)

df["kharif_rainfall"] = (
    df["rain_jun"] +
    df["rain_jul"] +
    df["rain_aug"] +
    df["rain_sep"] +
    df["rain_oct"]
)

# Rabi (Winter crops)

df["rabi_rainfall"] = (
    df["rain_oct"] +
    df["rain_nov"] +
    df["rain_dec"] +
    df["rain_jan"] +
    df["rain_feb"] +
    df["rain_mar"]
)

# Zaid (Summer crops)

df["zaid_rainfall"] = (
    df["rain_mar"] +
    df["rain_apr"] +
    df["rain_may"] +
    df["rain_jun"]
)


# ------------------------------------------------
# Seasonal rainfall ratios
# ------------------------------------------------

df["kharif_ratio"] = df["kharif_rainfall"] / df["annual_rainfall"]
df["rabi_ratio"] = df["rabi_rainfall"] / df["annual_rainfall"]
df["zaid_ratio"] = df["zaid_rainfall"] / df["annual_rainfall"]


# ------------------------------------------------
# NDVI temporal features
# ------------------------------------------------

print("Adding NDVI trend features...")

df["ndvi_trend"] = df["ndvi"] - df["ndvi_lag1"]

df["ndvi_acceleration"] = (
    df["ndvi"] - 2*df["ndvi_lag1"] + df["ndvi_lag2"]
)


# ------------------------------------------------
# Rainfall shock indicator
# ------------------------------------------------

print("Adding rainfall shock feature...")

threshold = df["rainfall_deviation"].std()

df["rainfall_shock"] = (
    abs(df["rainfall_deviation"]) > threshold
).astype(int)


# ------------------------------------------------
# Yield stability metric
# ------------------------------------------------

print("Adding yield stability...")

df = df.sort_values(["district","crop","year"])

df["yield_variance"] = (
    df.groupby(["district","crop"])["yield"]
    .transform(lambda x: x.rolling(3, min_periods=1).std())
)

df["yield_stability"] = 1 / (df["yield_variance"] + 1e-6)


# ------------------------------------------------
# Save enhanced dataset
# ------------------------------------------------

OUTPUT_PATH = "data/agri_multimodal_dataset_engineered.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("Feature engineering complete")
print("New dataset shape:", df.shape)
print("Saved to:", OUTPUT_PATH)