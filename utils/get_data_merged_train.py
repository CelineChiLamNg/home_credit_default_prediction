from typing import List, Dict
import pandas as pd
import numpy as np

import pandas as pd

from utils.custom_preprocessor import OneHotEncoderWithKeys
from sklearn.preprocessing import OrdinalEncoder

def encode_object_columns_with_ordinal_optimized(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """
    Optimized encoding for all object-type columns in the DataFrame using OrdinalEncoder.
    Encodes all columns in one step to avoid fragmentation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        key_column (str): The key column to retain during encoding.

    Returns:
        pd.DataFrame: The DataFrame with object columns ordinally encoded.
    """
    encoded_df = df.copy()

    # Identify object-type columns to encode
    object_columns = encoded_df.select_dtypes(include='object').columns.tolist()

    if object_columns:
        # Extract the key column separately to avoid encoding it
        non_encoded = encoded_df.drop(columns=object_columns)
        object_data = encoded_df[object_columns]

        # Initialize OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Perform encoding
        encoded_values = encoder.fit_transform(object_data)

        # Create a DataFrame for encoded values
        encoded_columns = pd.DataFrame(
            encoded_values,
            columns=object_columns,
            index=encoded_df.index
        )

        # Concatenate back non-encoded and encoded columns
        encoded_df = pd.concat([non_encoded, encoded_columns], axis=1)

    return encoded_df





def process_bureau(bureau_transformed: pd.DataFrame) -> pd.DataFrame:



    # Encode object columns
    bureau_encoded = encode_object_columns_with_ordinal_optimized(bureau_transformed,
                                                     key_column='SK_ID_BUREAU')

# Step 1: Add counts for SK_ID_BUREAU
    bureau_counts: pd.DataFrame = bureau_encoded.groupby('SK_ID_CURR').size().reset_index(name='SK_ID_BUREAU_count')

    # Step 2: Drop unnecessary columns
    bureau_cleaned: pd.DataFrame = bureau_encoded.drop(columns=['SK_ID_BUREAU'], errors='ignore')

    # Identify numeric columns for aggregation
    numeric_cols = bureau_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude key and count columns from aggregation
    excluded_cols = ['SK_ID_CURR', 'SK_ID_BUREAU_count']
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

    # Step 3: Aggregate by SK_ID_CURR
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    bureau_aggregated = bureau_cleaned.groupby('SK_ID_CURR').agg(
        {col: agg_funcs for col in bureau_cleaned.columns if col not in ['SK_ID_CURR', 'SK_ID_BUREAU_count']}
    ).reset_index()

    # Step 4: Rename columns efficiently
    bureau_aggregated.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in bureau_aggregated.columns
    ]

    # Step 5: Merge counts into transformed bureau table
    # Check if SK_ID_CURR exists in bureau_counts
    if 'SK_ID_CURR' not in bureau_counts.columns:
        raise KeyError("SK_ID_CURR is missing from bureau_counts. Ensure the counting step includes it.")
    # Check if SK_ID_CURR exists in bureau_aggregated
    if 'SK_ID_CURR_' not in bureau_aggregated.columns:
        raise KeyError("SK_ID_CURR_ is missing from bureau_aggregated. "
                       "Ensure the grouping step includes it.")


    print("äbout to run merge")
    # Merge the two DataFrames
    bureau_transformed_final: pd.DataFrame = bureau_aggregated.merge(
        bureau_counts, left_on='SK_ID_CURR_', right_on='SK_ID_CURR',
        how='left')

    print("merge ran successfully")
    return bureau_transformed_final


def process_previous_application(previous_application_transformed: pd.DataFrame) -> pd.DataFrame:

    # Encode object columns
    previous_encoded = encode_object_columns_with_ordinal_optimized(
        previous_application_transformed, key_column='SK_ID_PREV')


# Step 1: Add counts for SK_ID_PREV
    previous_counts: pd.DataFrame = previous_encoded.groupby('SK_ID_CURR').size().reset_index(name='SK_ID_PREV_count')

    # Step 2: Drop unnecessary columns
    previous_cleaned: pd.DataFrame = previous_encoded.drop(columns=['SK_ID_PREV'], errors='ignore')

    # Identify numeric columns for aggregation
    numeric_cols = previous_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude key and count columns from aggregation
    excluded_cols = ['SK_ID_CURR', 'SK_ID_PREV_count']
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

    # Step 3: Aggregate by SK_ID_CURR only on numeric columns
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    previous_aggregated: pd.DataFrame = previous_cleaned.groupby('SK_ID_CURR')[numeric_cols].agg(agg_funcs)
    previous_aggregated.columns = [f"{col[0]}_{col[1]}" for col in previous_aggregated.columns]
    previous_aggregated.reset_index(inplace=True)

    # Step 3: Aggregate by SK_ID_CURR
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    previous_aggregated = previous_cleaned.groupby('SK_ID_CURR').agg(
        {col: agg_funcs for col in previous_cleaned.columns if col not in ['SK_ID_CURR', 'SK_ID_BUREAU_count']}
    ).reset_index()

    # Step 4: Rename columns efficiently
    previous_aggregated.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in previous_aggregated.columns
    ]


    # Step 5: Merge counts into transformed previous application table
    previous_transformed_final: pd.DataFrame = previous_aggregated.merge(previous_counts, on='SK_ID_CURR', how='left')
    return previous_transformed_final

def merge_with_main_table(
        main_table: pd.DataFrame,
        bureau_transformed: pd.DataFrame,
        previous_application_transformed: pd.DataFrame
) -> pd.DataFrame:
    # Step 1: Merge bureau_transformed_final into main table
    merged_with_bureau: pd.DataFrame = main_table.merge(bureau_transformed, on='SK_ID_CURR', how='left')

    # Step 2: Merge previous_transformed_final into the updated main table
    final_table: pd.DataFrame = merged_with_bureau.merge(previous_application_transformed, on='SK_ID_CURR', how='left')
    return final_table
def get_data_merged_train(
        bureau_transformed: pd.DataFrame,
        previous_application_transformed: pd.DataFrame,
        train_splits: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    bureau_final: pd.DataFrame = process_bureau(bureau_transformed)
    previous_final: pd.DataFrame = process_previous_application(previous_application_transformed)
    merged_train: pd.DataFrame = merge_with_main_table(train_splits['data_train'], bureau_final, previous_final)
    return merged_train



