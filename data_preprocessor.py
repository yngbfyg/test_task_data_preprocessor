import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()

        self.removed_columns = []
        self.fill_values = {}
        self.encoded_columns = []
        self.numeric_stats = {}

    def remove_missing(self, threshold: float = 0.5):
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1")

        missing_ratio = self.df.isna().mean()
        self.removed_columns = missing_ratio[missing_ratio > threshold].index.tolist()
        self.df = self.df.drop(columns=self.removed_columns)

        for col in self.df.columns:
            if self.df[col].isna().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    value = self.df[col].mean()
                else:
                    value = self.df[col].mode().iloc[0]
                self.fill_values[col] = value
                self.df[col] = self.df[col].fillna(value)

        return self

    def encode_categorical(self):
        categorical_cols = self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if categorical_cols:
            dummies = pd.get_dummies(self.df[categorical_cols], drop_first=False)
            self.encoded_columns = dummies.columns.tolist()
            self.df = pd.concat([self.df.drop(columns=categorical_cols), dummies], axis=1)
        return self

    def normalize_numeric(self, method: str = "minmax"):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if method == "minmax":
                mn = self.df[col].min()
                mx = self.df[col].max()
                self.numeric_stats[col] = ("minmax", float(mn), float(mx))
                if mx != mn:
                    self.df[col] = (self.df[col] - mn) / (mx - mn)
                else:
                    self.df[col] = 0.0

            elif method == "std":
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.numeric_stats[col] = ("std", float(mean), float(std))
                if std != 0:
                    self.df[col] = (self.df[col] - mean) / std
                else:
                    self.df[col] = 0.0
            else:
                raise ValueError("method must be 'minmax' or 'std'")

        return self

    def fit_transform(self, threshold: float = 0.5, method: str = "minmax") -> pd.DataFrame:
        return (
            self.remove_missing(threshold)
            .encode_categorical()
            .normalize_numeric(method)
            .df
        )
