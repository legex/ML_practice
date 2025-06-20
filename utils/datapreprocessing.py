"""
Module for data preprocessing tasks including:
- Handling missing values
- Categorical encoding (label and one-hot)
- Target encoding
- Scaling features
"""
import os
from random import randint
from dataclasses import dataclass, field
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer


class DataProcessing:
    """
    Handles preprocessing: 
    missing value imputation, categorical encoding, target encoding, and scaling.
    """

    @dataclass
    class _Config:
        target_col: str
        encoder_dir: str = "encoders"
        is_inference: bool = False
        random_int: int = field(default_factory=lambda: randint(15, 50))

        def target_encoder_path(self):
            """
            Returns the path for the target encoder.
            """
            return os.path.join(self.encoder_dir, f"label_{self.target_col}.pkl")

        def ensure_encoder_dir(self):
            """
            Ensures the encoder directory exists.
            """
            os.makedirs(self.encoder_dir, exist_ok=True)

    def __init__(self, dataframe, target_col, encoder_dir='encoders', is_inference=False):
        self.df = dataframe.copy()
        self.config = self._Config(target_col, encoder_dir, is_inference)
        self.config.ensure_encoder_dir()

        self.target_col = self.config.target_col
        self.low_cardinality = []
        self.high_cardinality = []
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()

        if self.config.is_inference and self.target_col in self.df.columns:
            self.df.drop(columns=[self.target_col], inplace=True)

    def _load_encoder(self, encoder_type, col):
        """
        Loads a pre-trained encoder from the specified directory.
        Args:
            encoder_type (str): Type of encoder ('label' or 'onehot').
            col (str): Column name for which the encoder is saved.
        Returns:
            Loaded encoder object.
        """
        path = os.path.join(self.config.encoder_dir, f"{encoder_type}_{col}.pkl")
        return joblib.load(path)

    def _save_encoder(self, encoder, encoder_type, col):
        """ Saves the encoder to the specified directory.
        Args:
            encoder: Encoder object to save.
            encoder_type (str): Type of encoder ('label' or 'onehot').
            col (str): Column name for which the encoder is saved.
        returns:
            None
        """
        path = os.path.join(self.config.encoder_dir, f"{encoder_type}_{col}.pkl")
        joblib.dump(encoder, path)

    def handle_missing(self):
        """ 
        Imputes missing values in the DataFrame.
        Uses median for numerical columns and most frequent for categorical columns.
        """
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                strategy = "most_frequent" if self.df[col].dtype == 'object' else "median"
                imputer = SimpleImputer(strategy=strategy)
                self.df[col] = imputer.fit_transform(self.df[[col]]).ravel()
        return self.df

    def handle_categorical(self):
        """ 
        Handles categorical encoding for the DataFrame.
        Encodes low cardinality columns with Label Encoding,
        and high cardinality columns with One-Hot Encoding.
        Returns:
        DataFrame with encoded categorical features.
        """
        self.df = self.handle_missing()
        all_cols = list(self.df.columns)

        for col in all_cols:
            if col == self.target_col:
                continue

            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 3:
                    self.low_cardinality.append(col)
                    if self.config.is_inference:
                        le = self._load_encoder('label', col)
                        self.df[col] = le.transform(self.df[col].astype(str))
                    else:
                        le = LabelEncoder()
                        self.df[col] = le.fit_transform(self.df[col].astype(str))
                        self._save_encoder(le, 'label', col)
                else:
                    self.high_cardinality.append(col)
                    if self.config.is_inference:
                        oe = self._load_encoder('onehot', col)
                        encoded = oe.transform(self.df[[col]].astype(str))
                        categories = oe.categories_[0]
                    else:
                        oe = OneHotEncoder(handle_unknown='ignore')
                        encoded = oe.fit_transform(self.df[[col]].astype(str))
                        self._save_encoder(oe, 'onehot', col)
                        categories = oe.categories_[0]

                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in categories],
                        index=self.df.index
                    )
                    self.df.drop(columns=[col], inplace=True)
                    self.df = pd.concat([self.df, encoded_df], axis=1)

        return self.df

    def split(self):
        """
        Splits the data into training and testing sets, handling categorical encoding and scaling.
        Returns:
            If inference: Scaled features.
            If training: Tuple of (X_train, X_test, y_train, y_test).
        """
        self.df = self.handle_categorical()
        x_data = self.df.drop(self.target_col, axis=1) if not self.config.is_inference else self.df

        # Encode target
        if not self.config.is_inference:
            y = self.target_encoder.fit_transform(self.df[self.target_col].astype(str))
            joblib.dump(self.target_encoder, self.config.target_encoder_path())
        else:
            y = None
            if os.path.exists(self.config.target_encoder_path()):
                self.target_encoder = joblib.load(self.config.target_encoder_path())

        # Scale features
        if self.config.is_inference:
            self.scaler = joblib.load(os.path.join(self.config.encoder_dir, 'scaler.pkl'))
            x_scaled = self.scaler.transform(x_data)
            return x_scaled
        else:
            x_scaled = self.scaler.fit_transform(x_data)
            joblib.dump(self.scaler, os.path.join(self.config.encoder_dir, 'scaler.pkl'))
            return train_test_split(
                x_scaled, y, test_size=0.2, random_state=self.config.random_int
            )
