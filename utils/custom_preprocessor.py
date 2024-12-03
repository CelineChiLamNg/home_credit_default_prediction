from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class OneHotEncoderWithKeys(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode, key_column, prefix):
        self.column_to_encode = column_to_encode
        self.key_column = key_column
        self.prefix = prefix
        self.encoder = OneHotEncoder(sparse_output=False,
                                     handle_unknown='ignore')

    def fit(self, X, y=None):
        # Fit the encoder on the specified column
        self.encoder.fit(X[[self.column_to_encode]])
        return self

    def transform(self, X):
        # Extract the keys and column to encode
        keys = X[self.key_column]
        column = X[[self.column_to_encode]]

        # Transform the column using the fitted encoder
        encoded_array = self.encoder.transform(column)

        # Convert to a DataFrame with appropriate column names
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=[f"{self.prefix}_{cat}" for cat in
                     self.encoder.categories_[0]]
        )

        # Drop the original column to encode
        X_dropped = X.drop(columns=[self.column_to_encode])

        # Concatenate the rest of the columns with the encoded columns (key column already exists)
        result = pd.concat([X_dropped.reset_index(drop=True), encoded_df],
                           axis=1)

        return result


