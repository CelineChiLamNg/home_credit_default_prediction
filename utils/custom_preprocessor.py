from typing import Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

# Encoding
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



# Aggregating to bureau and previous_application
class BureauBalanceAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, agg_funcs=['mean', 'sum', 'min', 'max', 'count']):
        self.agg_funcs = agg_funcs

    def fit(self, X, y=None):
        return self  # No fitting required, so just return self

    def transform(self, X):
        # X should be a tuple: (bureau, bureau_balance)
        bureau, bureau_balance = X

        bureau = bureau.rename(
            columns={col: f"bureau_{col}" for col in bureau.columns if
                     col not in ['SK_ID_BUREAU', 'SK_ID_CURR']})

        # Perform aggregation
        bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(
            {col: self.agg_funcs for col in bureau_balance.columns if col != 'SK_ID_BUREAU'}
        )

        bureau_balance_agg.columns = [f"bureau_balance_{col[0]}_{col[1]}" for
                                      col in bureau_balance_agg.columns]
        bureau_balance_agg.reset_index(inplace=True)

        # Merge aggregated data back to bureau
        bureau = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')
        return bureau


class PreviousApplicationAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, agg_funcs=['mean', 'sum', 'min', 'max', 'count']):
        self.agg_funcs = agg_funcs

    def fit(self, X, y=None):
        return self  # No fitting required, so just return self

    def transform(self, X):
        # X should be a tuple: (previous_application, credit_card_balance, installments_payments, pos_cash_balance)
        previous_application, credit_card_balance, installments_payments, pos_cash_balance = X

        previous_application = previous_application.rename(
            columns={col: f"previous_application_{col}" for col in
                     previous_application.columns if col not in [
                         'SK_ID_PREV', 'SK_ID_CURR']}
        )

        # Aggregate related tables
        credit_card_agg = self._aggregate_table_by_prev(credit_card_balance, 'credit_card')
        installments_agg = self._aggregate_table_by_prev(installments_payments, 'installments')
        pos_cash_agg = self._aggregate_table_by_prev(pos_cash_balance, 'pos_cash')

        # Merge all aggregations back into previous_application
        previous_application = previous_application.merge(credit_card_agg, on='SK_ID_PREV', how='left')
        previous_application = previous_application.merge(installments_agg, on='SK_ID_PREV', how='left')
        previous_application = previous_application.merge(pos_cash_agg, on='SK_ID_PREV', how='left')

        return previous_application

    def _aggregate_table_by_prev(self, table, table_name):
        agg = table.groupby('SK_ID_PREV').agg(
            {col: self.agg_funcs for col in table.columns if col not in ['SK_ID_PREV', 'SK_ID_CURR']}
        )
        agg.columns = [f"{table_name}_{col[0]}_{col[1]}" for col in agg.columns]
        agg.reset_index(inplace=True)
        return agg

class AggregateByKey(BaseEstimator, TransformerMixin):
    def __init__(self, key_column, agg_funcs=['mean', 'sum', 'min', 'max', 'count']):
        self.key_column = key_column
        self.agg_funcs = agg_funcs

    def fit(self, X, y=None):
        return self  # No fitting required, so just return self

    def transform(self, X):
        # Perform aggregation by key_column
        agg_df = X.groupby(self.key_column).agg(
            {col: self.agg_funcs for col in X.columns if col != self.key_column}
        )

        # Flatten column names
        agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
        agg_df.reset_index(inplace=True)

        return agg_df


class MultiInputWrapper(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Accept two DataFrames
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Input to MultiInputWrapper must be a tuple of two DataFrames.")
        self.bureau_df, self.prev_app_df = X
        return self

    def transform(self, X):
        return [self.bureau_df,  self.prev_app_df]


#Count SK_ID_BUREAU and SK_ID_PREV
class SKIDCounter(BaseEstimator, TransformerMixin):
    def __init__(self, id_column, count_column_name):
        self.id_column = id_column
        self.count_column_name = count_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        count_df = X.groupby('SK_ID_CURR').size().reset_index(name=self.count_column_name)
        return count_df



#Drop Unnecessary Columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


# Merge Count Columns into Transformed Tables
class AddCountsToTransformed(BaseEstimator, TransformerMixin):
    def __init__(self, count_column_name):
        self.count_column_name = count_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, transformed_table, count_df):
        return transformed_table.merge(count_df, on='SK_ID_CURR', how='left')


# Custom Transformer for Merging
class MergePipelineOutputWithMainTable(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline, main_table, key_column='SK_ID_CURR'):
        self.pipeline = pipeline
        self.main_table = main_table
        self.key_column = key_column

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        pipeline_output = self.pipeline.transform(X)
        merged_table = self.main_table.merge(pipeline_output, on=self.key_column, how='left')
        return merged_table
