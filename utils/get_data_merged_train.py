from typing import List, Dict
import pandas as pd
import numpy as np

import pandas as pd

from utils.custom_preprocessor import OneHotEncoderWithKeys


def process_bureau(bureau_transformed: pd.DataFrame) -> pd.DataFrame:



    status_encoder = OneHotEncoderWithKeys(
        column_to_encode='STATUS', key_column='SK_ID_BUREAU', prefix='STATUS'
    )

    bureau_encoded = status_encoder.fit_transform(bureau_transformed)


    # Step 1: Add counts for SK_ID_BUREAU
    bureau_counts: pd.DataFrame = bureau_encoded.groupby('SK_ID_CURR').size().reset_index(name='SK_ID_BUREAU_count')

    # Step 2: Drop unnecessary columns
    bureau_cleaned: pd.DataFrame = bureau_encoded.drop(columns=['SK_ID_BUREAU'], errors='ignore')

    # Identify numeric columns for aggregation
    numeric_cols = bureau_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude key and count columns from aggregation
    excluded_cols = ['SK_ID_CURR', 'SK_ID_BUREAU_count']
    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

    # Step 3: Aggregate by SK_ID_CURR only on numeric columns
    agg_funcs: List[str] = ['mean', 'sum', 'min', 'max', 'count']
    bureau_aggregated: pd.DataFrame = bureau_cleaned.groupby('SK_ID_CURR')[numeric_cols].agg(agg_funcs)
    bureau_aggregated.columns = [f"{col[0]}_{col[1]}" for col in bureau_aggregated.columns]
    bureau_aggregated.reset_index(inplace=True)

    # Step 4: Merge counts into transformed bureau table
    bureau_transformed_final: pd.DataFrame = bureau_aggregated.merge(bureau_counts, on='SK_ID_CURR', how='left')
    return bureau_transformed_final


def process_previous_application(previous_application_transformed: pd.DataFrame) -> pd.DataFrame:
    # Encode object columns




    status_encoder = OneHotEncoderWithKeys(
        column_to_encode='STATUS', key_column='SK_ID_BUREAU', prefix='STATUS'
    )

    previous_encoded = status_encoder.fit_transform(previous_application_transformed)


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

    # Step 4: Merge counts into transformed previous application table
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



