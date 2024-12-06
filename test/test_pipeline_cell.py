import pickle
import os

from utils.custom_preprocessor import get_data_merged_train

bureau_file_path = "/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/aggregated_data/bureau_transformed.pkl"
previous_application_file_path = "/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/aggregated_data/previous_application_transformed.pkl"
data_train_file_path ="/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/data_train_test_split/train_split/data_train.pkl"

# Load the DataFrame
with open(data_train_file_path, "rb") as data_train_file:
    data_train = pickle.load(data_train_file)

# Load the DataFrames
with open(bureau_file_path, "rb") as bureau_file:
    bureau_transformed = pickle.load(bureau_file)

with open(previous_application_file_path, "rb") as previous_application_file:
    previous_application_transformed = pickle.load(previous_application_file)

data_merged_train = get_data_merged_train(bureau_transformed,
                                          previous_application_transformed,
                                          {'data_train': data_train})