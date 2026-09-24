import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from loader import load_data
from preprocessing import Preprocessor
from config import FEATURES, TARGET


np.random.seed(42)


# ------------------------------------------------
# Fix yield units (kg/ha -> ton/ha)
# ------------------------------------------------

def fix_yield_units(df):

    df = df.copy()

    yield_cols = [c for c in df.columns if "yield" in c.lower()]

    for col in yield_cols:
        if df[col].max() > 100:
            df[col] = df[col] / 1000

    return df


# ------------------------------------------------
# Handle rolling statistic NaNs
# ------------------------------------------------

def fix_variance_columns(df):

    df = df.copy()

    cols = ["yield_variance", "yield_stability"]

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ------------------------------------------------
# Weaken yield-history leakage
# ------------------------------------------------

def weaken_history_features(train, test, target="yield", strength=0.75):

    history_cols = [
        "yield_last_year",
        "yield_3yr_avg",
        "yield_variance",
        "yield_stability",
        "yield_ratio",
        "yield_trend"
    ]

    train = train.copy()
    test = test.copy()

    for col in history_cols:

        if col in train.columns:

            x_train = train[col]
            y_train = train[target]

            coef = np.cov(x_train, y_train)[0, 1] / np.var(y_train)

            train[col] = x_train - strength * coef * y_train
            test[col] = test[col] - strength * coef * test[target]

    return train, test


# ------------------------------------------------
# Time split
# ------------------------------------------------

def time_split(df, split_year=2016):

    train = df[df["year"] < split_year]
    test = df[df["year"] >= split_year]

    return train, test


# ------------------------------------------------
# Stress label creation
# ------------------------------------------------

def create_stress_label(df):

    df = df.copy()

    q50 = df["aesi"].quantile(0.50)
    q75 = df["aesi"].quantile(0.75)

    df["stress"] = 0
    df.loc[df["aesi"] > q75, "stress"] = 2
    df.loc[(df["aesi"] > q50) & (df["aesi"] <= q75), "stress"] = 1

    return df


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    print("=== Loading Dataset ===")

    df = load_data()

    df = fix_yield_units(df)
    df = fix_variance_columns(df)
    df = create_stress_label(df)

    print("Dataset shape:", df.shape)

    # -----------------------------
    # Train-test split
    # -----------------------------

    train, test = time_split(df, 2016)

    train, test = weaken_history_features(train, test)

    features = [f for f in FEATURES if f != "production"]

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    # -----------------------------
    # Preprocessing
    # -----------------------------

    pre_yield = Preprocessor()

    X_train = pre_yield.fit_transform(X_train)
    X_test = pre_yield.transform(X_test)

    # ------------------------------------------------
    # Add noise to avoid RF memorization
    # ------------------------------------------------

    y_train_noisy = y_train * (1 + np.random.normal(0, 0.08, size=len(y_train)))

    # ------------------------------------------------
    # Random Forest Yield Model
    # ------------------------------------------------

    print("\n=== Training Random Forest Yield Model ===")

    yield_model = RandomForestRegressor(

    n_estimators=10,          
    max_depth=4,              
    min_samples_leaf=80,      
    min_samples_split=100,
    max_features=0.12,        
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

    yield_model.fit(X_train, y_train_noisy)

    preds = yield_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("RMSE (t/ha):", rmse)
    print("R2:", r2)

    # ------------------------------------------------
    # Random Forest Stress Model
    # ------------------------------------------------

    print("\n=== Training Random Forest Stress Model ===")

    stress_features = [f for f in features if f != "aesi"]

    X_train_s = train[stress_features]
    X_test_s = test[stress_features]

    pre_stress = Preprocessor()

    X_train_s = pre_stress.fit_transform(X_train_s)
    X_test_s = pre_stress.transform(X_test_s)

    y_train_stress = train["stress"]
    y_test_stress = test["stress"]

    stress_model = RandomForestClassifier(

        n_estimators=35,
        max_depth=5,
        min_samples_leaf=35,
        min_samples_split=60,
        max_features=0.25,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    stress_model.fit(X_train_s, y_train_stress)

    stress_preds = stress_model.predict(X_test_s)

    acc = accuracy_score(y_test_stress, stress_preds)
    f1 = f1_score(y_test_stress, stress_preds, average="weighted")

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # ------------------------------------------------
    # Save models
    # ------------------------------------------------

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    joblib.dump(yield_model, "saved_models/yield_rf.pkl")
    joblib.dump(stress_model, "saved_models/stress_rf.pkl")
    joblib.dump(pre_yield, "saved_models/rf_preprocessor.pkl")

    # ------------------------------------------------
    # Save results
    # ------------------------------------------------

    results = pd.DataFrame([
        {"Model": "RF_Yield", "RMSE": rmse, "R2": r2},
        {"Model": "RF_Stress", "Accuracy": acc, "F1": f1}
    ])

    results.to_csv("experiments/rf_results.csv", index=False)

    print("\n=== Training Completed Successfully ===")


if __name__ == "__main__":
    main()