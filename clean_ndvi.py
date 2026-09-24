import pandas as pd

print("NDVI cleaning started")

# -----------------------------------
# File paths
# -----------------------------------

INPUT_FILE = "data/raw/india_district_ndvi.csv"
OUTPUT_FILE = "data/raw/district_ndvi_clean.csv"


# -----------------------------------
# Load dataset
# -----------------------------------

ndvi = pd.read_csv(INPUT_FILE)

print("Raw dataset shape:", ndvi.shape)
print("Columns detected:", list(ndvi.columns))


# -----------------------------------
# Detect district column
# -----------------------------------

district_candidates = [
    "adm2_name",
    "district",
    "district_name",
    "dist_name",
    "name"
]

district_col = None

for col in ndvi.columns:
    if col.lower() in district_candidates:
        district_col = col
        break

if district_col is None:
    raise ValueError("District column not found")


# -----------------------------------
# Detect NDVI column
# -----------------------------------

ndvi_candidates = [
    "mean",
    "ndvi",
    "value",
    "avg"
]

ndvi_col = None

for col in ndvi.columns:
    if col.lower() in ndvi_candidates:
        ndvi_col = col
        break

if ndvi_col is None:
    raise ValueError("NDVI column not found")


# -----------------------------------
# Keep required columns
# -----------------------------------

ndvi = ndvi[[district_col, "year", ndvi_col]]


# -----------------------------------
# Rename columns
# -----------------------------------

ndvi = ndvi.rename(columns={
    district_col: "district",
    ndvi_col: "ndvi"
})


# -----------------------------------
# Clean district names
# -----------------------------------

ndvi["district"] = (
    ndvi["district"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# -----------------------------------
# Convert NDVI scale if needed
# -----------------------------------

if ndvi["ndvi"].max() > 1:
    print("Converting NDVI scale from 0-10000 → 0-1")
    ndvi["ndvi"] = ndvi["ndvi"] / 10000


# -----------------------------------
# Remove invalid NDVI values
# -----------------------------------

ndvi = ndvi[
    (ndvi["ndvi"] >= -1) &
    (ndvi["ndvi"] <= 1)
]


# -----------------------------------
# Remove duplicates
# -----------------------------------

ndvi = ndvi.drop_duplicates(
    subset=["district", "year"]
)


# -----------------------------------
# Sort dataset
# -----------------------------------

ndvi = ndvi.sort_values(
    ["district", "year"]
)

ndvi["year"] = ndvi["year"].astype(int)
# -----------------------------------
# Save cleaned dataset
# -----------------------------------

ndvi.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nNDVI cleaning finished")

print("Clean dataset shape:", ndvi.shape)

print("\nExample rows:")
print(ndvi.head())


print("\nSaved to:", OUTPUT_FILE)

import pandas as pd

ndvi = pd.read_csv("data/district_ndvi_clean.csv")

print("Shape:", ndvi.shape)
print("\nColumns:", ndvi.columns)

print("\nSample rows:")
print(ndvi.head())

print("\nYear range:")
print(ndvi["year"].min(), "-", ndvi["year"].max())

print("\nNDVI statistics:")
print(ndvi["ndvi"].describe())

print("\nUnique districts:", ndvi["district"].nunique())