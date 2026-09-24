import pandas as pd


def load_data(path="data/agri_multimodal_dataset_engineered.csv"):
    """
    Loads the dataset without performing preprocessing.
    Responsibilities:
    - Read dataset
    - Ensure correct column types
    - Return dataframe
    """

    df = pd.read_csv(path)

    print(f"Dataset loaded: {df.shape}")

    # Ensure correct types for time column
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)

    # Ensure grouping columns are strings
    if "state_clean" in df.columns:
        df["state_clean"] = df["state_clean"].astype(str)

    if "district_clean" in df.columns:
        df["district_clean"] = df["district_clean"].astype(str)

    return df