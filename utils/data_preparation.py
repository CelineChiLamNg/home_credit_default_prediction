import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def split_related_table(df, train_ids, test_ids, id_column):
    train_split = df[df[id_column].isin(train_ids)]
    test_split = df[df[id_column].isin(test_ids)]
    return train_split, test_split


def encode_object_columns_with_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes object-type columns using OrdinalEncoder while preserving NaN values.
    """
    df = df.copy()

    # Identify object columns
    object_columns = df.select_dtypes(include='object').columns.tolist()

    if object_columns:
        # Initialize the OrdinalEncoder without modifying NaN
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

        # Encode only the object columns
        df[object_columns] = encoder.fit_transform(df[object_columns])

    return df


