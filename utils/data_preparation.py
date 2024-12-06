import pandas as pd


def split_related_table(df, train_ids, test_ids, id_column):
    train_split = df[df[id_column].isin(train_ids)]
    test_split = df[df[id_column].isin(test_ids)]
    return train_split, test_split

