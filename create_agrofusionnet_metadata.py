import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

DATA_PATH = "data/agri_multimodal_dataset_engineered.csv"

os.makedirs("saved_models", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ------------------------------------------------
# Feature Groups (must match training file)
# ------------------------------------------------

RAIN_COLS = [
    "rain_jan","rain_feb","rain_mar","rain_apr","rain_may","rain_jun",
    "rain_jul","rain_aug","rain_sep","rain_oct","rain_nov","rain_dec"
]

NDVI_COLS = [
    "ndvi","ndvi_lag1","ndvi_lag2"
]

CLIMATE = [
    "annual_rainfall","rainfall_avg","rainfall_deviation",
    "rainfall_variability","rainfall_shock",
    "kharif_rainfall","rabi_rainfall","zaid_rainfall",
    "kharif_ratio","rabi_ratio","zaid_ratio"
]

SOIL = [
    "n","p","k","ph",
    "nitrogen_surplus","nutrient_balance",
    "np_ratio","nk_ratio","pk_ratio","npk_balance"
]

INTERACTION = [
    "rainfall_n_interaction",
    "rainfall_p_interaction",
    "rainfall_k_interaction"
]

STRESS = [
    "snii","aesi"
]

TEMPORAL = [
    "yield_last_year",
    "yield_3yr_avg",
    "yield_variance",
    "yield_stability"
]

# ------------------------------------------------
# All numerical features
# ------------------------------------------------

ALL_FEATURES = (
    RAIN_COLS +
    NDVI_COLS +
    CLIMATE +
    SOIL +
    INTERACTION +
    STRESS +
    TEMPORAL
)

# ------------------------------------------------
# Create and save scaler
# ------------------------------------------------

print("Fitting scaler...")

scaler = StandardScaler()

df[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])

joblib.dump(
    scaler,
    "saved_models/agrofusionnet_scaler.pkl"
)

print("Saved scaler")

# ------------------------------------------------
# Save feature metadata
# ------------------------------------------------

metadata = {
    "RAIN_COLS": RAIN_COLS,
    "NDVI_COLS": NDVI_COLS,
    "CLIMATE": CLIMATE,
    "SOIL": SOIL,
    "INTERACTION": INTERACTION,
    "STRESS": STRESS,
    "TEMPORAL": TEMPORAL,
    "ALL_FEATURES": ALL_FEATURES
}

joblib.dump(
    metadata,
    "saved_models/agrofusionnet_metadata.pkl"
)

print("Saved feature metadata")

# ------------------------------------------------
# Save district mapping
# ------------------------------------------------

print("Creating district mapping...")

district_map = (
    df[["district","district_id"]]
    .drop_duplicates()
    .sort_values("district")
)

district_map.to_csv(
    "saved_models/district_mapping.csv",
    index=False
)

print("Saved district mapping")

# ------------------------------------------------
# Save stress thresholds
# ------------------------------------------------

print("Calculating stress thresholds...")

q50 = df["aesi"].quantile(0.50)
q75 = df["aesi"].quantile(0.75)

stress_thresholds = {
    "q50": float(q50),
    "q75": float(q75)
}

joblib.dump(
    stress_thresholds,
    "saved_models/stress_thresholds.pkl"
)

print("Saved stress thresholds")

# ------------------------------------------------
# Done
# ------------------------------------------------

print("\nAll metadata files created successfully!")

print("\nFiles created in saved_models/:")

print("""
agrofusionnet_metadata.pkl
agrofusionnet_scaler.pkl
district_mapping.csv
stress_thresholds.pkl
""")
