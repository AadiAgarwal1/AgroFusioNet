from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd


class Preprocessor:
    """
    Handles:
    - Missing value imputation
    - Feature scaling
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_columns = None

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit preprocessing pipeline and transform training data
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")

        # Store column order
        self.feature_columns = X.columns.tolist()

        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Return DataFrame to preserve column names
        return pd.DataFrame(X_scaled, columns=self.feature_columns)

    def transform(self, X: pd.DataFrame):
        """
        Apply preprocessing to new data (prediction)
        """

        if self.feature_columns is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")

        # Ensure same column order
        X = X[self.feature_columns]

        # Impute missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.feature_columns)