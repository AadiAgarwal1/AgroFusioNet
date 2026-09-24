# ------------------------------------------------
# Target column
# ------------------------------------------------

TARGET = "yield"


# ------------------------------------------------
# Time column
# ------------------------------------------------

TIME_COLUMN = "year"


# ------------------------------------------------
# Location columns
# ------------------------------------------------

STATE_COLUMN = "state_clean"
DISTRICT_COLUMN = "district_clean"


# ------------------------------------------------
# Core agricultural features
# ------------------------------------------------

CORE_FEATURES = [
    "area",
    "yield_last_year",
    "yield_3yr_avg",
]


# ------------------------------------------------
# Climate features
# ------------------------------------------------

CLIMATE_FEATURES = [
    "annual_rainfall",
    "rainfall_avg",
    "rainfall_deviation",
    "rainfall_variability",
    "kharif_rainfall",
    "rabi_rainfall",
    "zaid_rainfall",
    "kharif_ratio",
    "rabi_ratio",
    "zaid_ratio",
]


# ------------------------------------------------
# Soil features
# ------------------------------------------------

SOIL_FEATURES = [
    "n",
    "p",
    "k",
    "ph",
    "nitrogen_surplus",
    "nutrient_balance",
]


# ------------------------------------------------
# Soil balance & stress features
# ------------------------------------------------

SOIL_ENGINEERED = [
    "n_norm",
    "p_norm",
    "k_norm",
    "np_ratio",
    "nk_ratio",
    "pk_ratio",
    "npk_balance",
    "snii",
]


# ------------------------------------------------
# Vegetation (NDVI) features
# ------------------------------------------------

VEGETATION_FEATURES = [
    "ndvi",
    "ndvi_lag1",
    "ndvi_lag2",
    "ndvi_trend",
    "ndvi_acceleration",
    "ndvi_deficit",
]


# ------------------------------------------------
# Climate–soil interaction features
# ------------------------------------------------

INTERACTION_FEATURES = [
    "rainfall_n_interaction",
    "rainfall_p_interaction",
    "rainfall_k_interaction",
]


# ------------------------------------------------
# Stress indicators
# ------------------------------------------------

STRESS_FEATURES = [
    "aesi",
    "rainfall_shock",
]


# ------------------------------------------------
# Temporal stability features
# ------------------------------------------------

TEMPORAL_FEATURES = [
    "yield_variance",
    "yield_stability",
]


# ------------------------------------------------
# Final model feature list
# ------------------------------------------------

FEATURES = (
    CORE_FEATURES
    + CLIMATE_FEATURES
    + SOIL_FEATURES
    + SOIL_ENGINEERED
    + VEGETATION_FEATURES
    + INTERACTION_FEATURES
    + STRESS_FEATURES
    + TEMPORAL_FEATURES
)