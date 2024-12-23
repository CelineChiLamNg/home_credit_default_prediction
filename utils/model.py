import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

def prep_train_eval(
    preprocessor: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    """splits the data_csv into train/eval and preprocesses.

    Parameters
    ----------
    preprocessor : sklearn.pipeline.Pipeline
        Preprocessor pipeline to transform the dataset.
    X : pd.DataFrame
        predictor set.
    y : pd.DataFrame
        target column.
    """

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    X_train = pd.DataFrame(
        preprocessor.fit_transform(X_train, y_train),
        columns=preprocessor.get_feature_names_out(),
    )
    X_eval = pd.DataFrame(
        preprocessor.transform(X_eval), columns=preprocessor.get_feature_names_out()
    )
    return X_train, y_train, X_eval, y_eval

class AgeBinnerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, labels=None):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Age_binned'] = pd.cut(X['Age'], bins=self.bins, labels=self.labels)
        return X

