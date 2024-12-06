from typing import List, Dict
import pandas as pd
import numpy as np
import datetime

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
    print(f"[{datetime.datetime.now()}] Starting ordinal encoding...")
    encoded_df = df.copy()

    # Identify object-type columns to encode
    object_columns = encoded_df.select_dtypes(include='object').columns.tolist()
    print(f"[{datetime.datetime.now()}] Found {len(object_columns)} object columns to encode")

    if object_columns:
        print(f"[{datetime.datetime.now()}] Processing object columns...")
        # Extract the key column separately to avoid encoding it
        non_encoded = encoded_df.drop(columns=object_columns)
        object_data = encoded_df[object_columns]

        # Initialize OrdinalEncoder
        print(f"[{datetime.datetime.now()}] Initializing OrdinalEncoder...")
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Perform encoding
        print(f"[{datetime.datetime.now()}] Performing encoding transformation...")
        encoded_values = encoder.fit_transform(object_data)

        # Create a DataFrame for encoded values
        print(f"[{datetime.datetime.now()}] Creating DataFrame with encoded values...")
        encoded_columns = pd.DataFrame(
            encoded_values,
            columns=object_columns,
            index=encoded_df.index
        )

        # Concatenate back non-encoded and encoded columns
        print(f"[{datetime.datetime.now()}] Concatenating encoded and non-encoded columns...")
        encoded_df = pd.concat([non_encoded, encoded_columns], axis=1)

    print(f"[{datetime.datetime.now()}] Ordinal encoding completed successfully")
    return encoded_df


def process_bureau(bureau_transformed: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[{datetime.datetime.now()}] === Starting bureau processing ===")
    print(f"[{datetime.datetime.now()}] Initial bureau shape: {bureau_transformed.shape}")

    # Encode object columns
    print(f"[{datetime.datetime.now()}] Encoding object columns in bureau data...")
    bureau_encoded = encode_object_columns_with_ordinal_optimized(bureau_transformed,
                                                     key_column='SK_ID_BUREAU')
    print(f"[{datetime.datetime.now()}] Bureau encoding completed")

    # Step 1: Add counts for SK_ID_BUREAU
    print(f"[{datetime.datetime.now()}] Calculating bureau counts...")
    bureau_counts: pd.DataFrame = bureau_encoded.groupby('SK_ID_CURR').size().reset_index(name='SK_ID_BUREAU_count')
    print(f"[{datetime.datetime.now()}] Bureau counts shape: {bureau_counts.shape}")

    # Step 2: Drop unnecessary columns
    print(f"[{datetime.datetime.now()}] Cleaning bureau data...")
    bureau_cleaned: pd.DataFrame = bureau_encoded.drop(columns=['SK_ID_BUREAU'], errors='ignore')
    print(f"[{datetime.datetime.now()}] Cleaned bureau shape: {bureau_cleaned.shape}")

    # Identify numeric columns for aggregation
    print(f"[{datetime.datetime.now()}] Identifying numeric columns...")
    numeric_cols = bureau_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude key and count columns from aggregation
    excluded_cols = ['SK_ID_CURR', 'SK_ID_BUREAU_count']
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    print(f"[{datetime.datetime.now()}] Found {len(numeric_cols)} numeric columns for aggregation")

    # Step 3: Aggregate by SK_ID_CURR
    print(f"[{datetime.datetime.now()}] Performing bureau aggregation...")
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    bureau_aggregated = bureau_cleaned.groupby('SK_ID_CURR').agg(
        {col: agg_funcs for col in bureau_cleaned.columns if col not in ['SK_ID_CURR', 'SK_ID_BUREAU_count']}
    ).reset_index()
    print(f"[{datetime.datetime.now()}] Aggregated bureau shape: {bureau_aggregated.shape}")

    # Step 4: Rename columns efficiently
    print(f"[{datetime.datetime.now()}] Renaming columns...")
    bureau_aggregated.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in bureau_aggregated.columns
    ]

    # Step 5: Merge counts into transformed bureau table
    print(f"[{datetime.datetime.now()}] Performing final bureau merge...")
    # Check if SK_ID_CURR exists in bureau_counts
    if 'SK_ID_CURR' not in bureau_counts.columns:
        raise KeyError("SK_ID_CURR is missing from bureau_counts. Ensure the counting step includes it.")
    # Check if SK_ID_CURR exists in bureau_aggregated
    if 'SK_ID_CURR_' not in bureau_aggregated.columns:
        raise KeyError("SK_ID_CURR_ is missing from bureau_aggregated. "
                       "Ensure the grouping step includes it.")

    print(f"[{datetime.datetime.now()}] About to run merge...")
    # Merge the two DataFrames
    bureau_transformed_final: pd.DataFrame = bureau_aggregated.merge(
        bureau_counts, left_on='SK_ID_CURR_', right_on='SK_ID_CURR',
        how='left')

    print(f"[{datetime.datetime.now()}] Bureau processing completed. Final shape: {bureau_transformed_final.shape}")
    return bureau_transformed_final


def process_previous_application(previous_application_transformed: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[{datetime.datetime.now()}] === Starting previous application processing ===")
    print(f"[{datetime.datetime.now()}] Initial previous application shape: {previous_application_transformed.shape}")

    # Encode object columns
    print(f"[{datetime.datetime.now()}] Encoding object columns in previous application data...")
    previous_encoded = encode_object_columns_with_ordinal_optimized(
        previous_application_transformed, key_column='SK_ID_PREV')
    print(f"[{datetime.datetime.now()}] Previous application encoding completed")

    # Step 1: Add counts for SK_ID_PREV
    print(f"[{datetime.datetime.now()}] Calculating previous application counts...")
    previous_counts: pd.DataFrame = previous_encoded.groupby('SK_ID_CURR').size().reset_index(name='SK_ID_PREV_count')
    print(f"[{datetime.datetime.now()}] Previous counts shape: {previous_counts.shape}")

    # Step 2: Drop unnecessary columns
    print(f"[{datetime.datetime.now()}] Cleaning previous application data...")
    previous_cleaned: pd.DataFrame = previous_encoded.drop(columns=['SK_ID_PREV'], errors='ignore')
    print(f"[{datetime.datetime.now()}] Cleaned previous application shape: {previous_cleaned.shape}")

    # Identify numeric columns for aggregation
    print(f"[{datetime.datetime.now()}] Identifying numeric columns...")
    numeric_cols = previous_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude key and count columns from aggregation
    excluded_cols = ['SK_ID_CURR', 'SK_ID_PREV_count']
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    print(f"[{datetime.datetime.now()}] Found {len(numeric_cols)} numeric columns for aggregation")

    # Step 3: Aggregate by SK_ID_CURR only on numeric columns
    print(f"[{datetime.datetime.now()}] Performing initial aggregation...")
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    previous_aggregated: pd.DataFrame = previous_cleaned.groupby('SK_ID_CURR')[numeric_cols].agg(agg_funcs)
    previous_aggregated.columns = [f"{col[0]}_{col[1]}" for col in previous_aggregated.columns]
    previous_aggregated.reset_index(inplace=True)
    print(f"[{datetime.datetime.now()}] Initial aggregation shape: {previous_aggregated.shape}")

    # Step 3: Aggregate by SK_ID_CURR
    print(f"[{datetime.datetime.now()}] Performing full aggregation...")
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    previous_aggregated = previous_cleaned.groupby('SK_ID_CURR').agg(
        {col: agg_funcs for col in previous_cleaned.columns if col not in ['SK_ID_CURR', 'SK_ID_BUREAU_count']}
    ).reset_index()
    print(f"[{datetime.datetime.now()}] Full aggregation shape: {previous_aggregated.shape}")

    # Step 4: Rename columns efficiently
    print(f"[{datetime.datetime.now()}] Renaming columns...")
    previous_aggregated.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in previous_aggregated.columns
    ]

    # Step 5: Merge counts into transformed previous application table
    print(f"[{datetime.datetime.now()}] Performing final previous application merge...")

    # Verify SK_ID_CURR exists in both DataFrames
    if 'SK_ID_CURR_' not in previous_aggregated.columns:
        raise ValueError("SK_ID_CURR_ missing from previous_aggregated DataFrame")
    if 'SK_ID_CURR' not in previous_counts.columns:
        raise ValueError("SK_ID_CURR missing from previous_counts DataFrame")
    previous_transformed_final: pd.DataFrame = previous_aggregated.merge(previous_counts, left_on='SK_ID_CURR_', right_on='SK_ID_CURR', how='left')
    print(f"[{datetime.datetime.now()}] Previous application processing completed. Final shape: {previous_transformed_final.shape}")
    return previous_transformed_final

def merge_with_main_table(
        main_table: pd.DataFrame,
        bureau_transformed: pd.DataFrame,
        previous_application_transformed: pd.DataFrame
) -> pd.DataFrame:
    print(f"\n[{datetime.datetime.now()}] === Starting final merging process ===")
    print(f"[{datetime.datetime.now()}] Main table shape: {main_table.shape}")
    print(f"[{datetime.datetime.now()}] Bureau transformed shape: {bureau_transformed.shape}")
    print(f"[{datetime.datetime.now()}] Previous application transformed shape: {previous_application_transformed.shape}")

    # Step 1: Merge bureau_transformed_final into main table
    print(f"[{datetime.datetime.now()}] Merging bureau data with main table...")
    merged_with_bureau: pd.DataFrame = main_table.merge(bureau_transformed, on='SK_ID_CURR', how='left')
    print(f"[{datetime.datetime.now()}] Shape after bureau merge: {merged_with_bureau.shape}")

    # Step 2: Merge previous_transformed_final into the updated main table
    print(f"[{datetime.datetime.now()}] Merging previous application data...")
    final_table: pd.DataFrame = merged_with_bureau.merge(previous_application_transformed, on='SK_ID_CURR', how='left')
    print(f"[{datetime.datetime.now()}] Final merged shape: {final_table.shape}")
    return final_table

def get_data_merged_train(
        bureau_transformed: pd.DataFrame,
        previous_application_transformed: pd.DataFrame,
        train_splits: Dict[str, pd.DataFrame],
        n_rows: int = None
) -> pd.DataFrame:
    print(f"\n[{datetime.datetime.now()}] === Starting data merge train process ===")
    
    # If n_rows is specified, subset the main training data and filter related records
    if n_rows is not None:
        print(f"[{datetime.datetime.now()}] Subsetting data to first {n_rows} rows...")
        train_data_small = train_splits['data_train'].head(n_rows)
        valid_ids = train_data_small['SK_ID_CURR'].unique()
        
        bureau_transformed = bureau_transformed[
            bureau_transformed['SK_ID_CURR'].isin(valid_ids)
        ]
        previous_application_transformed = previous_application_transformed[
            previous_application_transformed['SK_ID_CURR'].isin(valid_ids)
        ]
        train_splits = {'data_train': train_data_small}
        
        print(f"[{datetime.datetime.now()}] Subset shapes:")
        print(f"Training data: {train_data_small.shape}")
        print(f"Bureau data: {bureau_transformed.shape}")
        print(f"Previous application data: {previous_application_transformed.shape}")
    
    print(f"[{datetime.datetime.now()}] Processing bureau data...")
    bureau_final: pd.DataFrame = process_bureau(bureau_transformed)
    
    print(f"\n[{datetime.datetime.now()}] Processing previous application data...")
    previous_final: pd.DataFrame = process_previous_application(previous_application_transformed)
    
    print(f"\n[{datetime.datetime.now()}] Performing final merge with main table...")
    merged_train: pd.DataFrame = merge_with_main_table(train_splits['data_train'], bureau_final, previous_final)
    print(f"[{datetime.datetime.now()}] Data merge train process completed successfully!")
    return merged_train

