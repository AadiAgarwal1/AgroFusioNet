import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from loader import load_data
from preprocessing import Preprocessor
from config import FEATURES, TARGET


np.random.seed(42)


# ------------------------------------------------
# Fix yield units across dataset
# ------------------------------------------------

def fix_yield_units(df):

    df = df.copy()

    yield_cols = [c for c in df.columns if "yield" in c.lower()]

    for col in yield_cols:
        if df[col].max() > 100:
            df[col] = df[col] / 1000

    return df


# ------------------------------------------------
# Time-based split
# ------------------------------------------------

def time_split(df, split_year=2016):

    train = df[df["year"] < split_year]
    test = df[df["year"] >= split_year]

    return train, test


# ------------------------------------------------
# Create stress labels
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
    df = create_stress_label(df)

    print("Dataset shape:", df.shape)

    # -----------------------------
    # Train-test split
    # -----------------------------

    train, test = time_split(df, 2016)

    # remove leakage feature
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

    # -----------------------------
    # Log-transform target
    # -----------------------------

    y_train_log = np.log1p(y_train)

    # -----------------------------
    # Yield model
    # -----------------------------

    print("\n=== Training Yield Model (XGBoost) ===")

    yield_model = XGBRegressor(

        n_estimators=80,
        learning_rate=0.1,
        max_depth=2,
        subsample=0.5,
        colsample_bytree=0.35,
        colsample_bylevel=0.35,
        reg_lambda=5,
        reg_alpha=2,
        min_child_weight=20,
        gamma=2,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror"
    )

    yield_model.fit(X_train, y_train_log)

    preds_log = yield_model.predict(X_test)

    preds = np.expm1(preds_log)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("RMSE (t/ha):", rmse)
    print("R2:", r2)

    # -----------------------------
    # Stress model
    # -----------------------------

    print("\n=== Training Stress Model (XGBoost) ===")

    stress_features = [f for f in features if f != "aesi"]

    X_train_s = train[stress_features]
    X_test_s = test[stress_features]

    pre_stress = Preprocessor()

    X_train_s = pre_stress.fit_transform(X_train_s)
    X_test_s = pre_stress.transform(X_test_s)

    y_train_stress = train["stress"]
    y_test_stress = test["stress"]

    stress_model = XGBClassifier(

        n_estimators=120,
        learning_rate=0.09,
        max_depth=3,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_lambda=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    stress_model.fit(X_train_s, y_train_stress)

    stress_preds = stress_model.predict(X_test_s)

    acc = accuracy_score(y_test_stress, stress_preds)
    f1 = f1_score(y_test_stress, stress_preds, average="weighted")

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # -----------------------------
    # Save models
    # -----------------------------

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    joblib.dump(yield_model, "saved_models/yield_xgb.pkl")
    joblib.dump(stress_model, "saved_models/stress_xgb.pkl")
    joblib.dump(pre_yield, "saved_models/xgb_preprocessor.pkl")

    results = pd.DataFrame([
        {"Model": "XGB_Yield", "RMSE": rmse, "R2": r2},
        {"Model": "XGB_Stress", "Accuracy": acc, "F1": f1}
    ])

    results.to_csv("experiments/xgb_results.csv", index=False)

    print("\n=== Training Completed Successfully ===")


if __name__ == "__main__":
    main()