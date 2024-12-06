import pickle
import pandas as pd

from utils.get_data_merged_train import get_data_merged_train

def run_test_small():
    # Define number of rows to use for testing
    N_ROWS = 1000  # Adjust this number as needed

    # File paths
    bureau_file_path = "/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/aggregated_data/bureau_transformed.pkl"
    previous_application_file_path = "/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/aggregated_data/previous_application_transformed.pkl"
    data_train_file_path = "/Users/celineng/PycharmProjects/chilng-DS.v2.5.3.4.1/data_train_test_split/train_split/data_train.pkl"

    # Load the DataFrames
    with open(data_train_file_path, "rb") as data_train_file:
        data_train = pickle.load(data_train_file)

    with open(bureau_file_path, "rb") as bureau_file:
        bureau_transformed = pickle.load(bureau_file)

    with open(previous_application_file_path, "rb") as previous_application_file:
        previous_application_transformed = pickle.load(previous_application_file)

    print(f"Original shapes:")
    print(f"data_train: {data_train.shape}")
    print(f"bureau_transformed: {bureau_transformed.shape}")
    print(f"previous_application_transformed: {previous_application_transformed.shape}")

    # Run the merge function with n_rows parameter
    data_merged_train_small = get_data_merged_train(
        bureau_transformed,
        previous_application_transformed,
        {'data_train': data_train},
        n_rows=N_ROWS
    )

    print("\nFinal merged shape:", data_merged_train_small.shape)
    print("\nFirst few rows of merged data:")
    print(data_merged_train_small.head())

if __name__ == "__main__":
    run_test_small() 