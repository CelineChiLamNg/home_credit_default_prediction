import os
import pandas as pd

whole_data_folder = "data_pkl"

train_folder = "data_train"
test_folder = "data_test"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

application_train = pd.read_pickle(
    os.path.join(whole_data_folder, 'application_train.pkl'))
application_test = pd.read_pickle(
    os.path.join(whole_data_folder, 'application_test.pkl'))

sk_id_curr_train = application_train['SK_ID_CURR']
sk_id_curr_test = application_test['SK_ID_CURR']

for filename in os.listdir(whole_data_folder):
    if filename.endswith(".pkl"):
        file_path = os.path.join(whole_data_folder, filename)
        df = pd.read_pickle(file_path)

        # Skip files without SK_ID_CURR column
        if 'SK_ID_CURR' not in df.columns:
            print(f"Skipping {filename} (no SK_ID_CURR column)")
            continue

        # Filter for train and test
        filtered_train = df[df['SK_ID_CURR'].isin(sk_id_curr_train)]
        filtered_test = df[df['SK_ID_CURR'].isin(sk_id_curr_test)]

        # Save filtered train data
        train_pkl_path = os.path.join(train_folder,
                                      f"{filename.replace('.pkl', '_train.pkl')}")
        filtered_train.to_pickle(train_pkl_path)
        print(f"Saved train data for {filename} to {train_pkl_path}")

        # Save filtered test data
        test_pkl_path = os.path.join(test_folder,
                                     f"{filename.replace('.pkl', '_test.pkl')}")
        filtered_test.to_pickle(test_pkl_path)
        print(f"Saved test data for {filename} to {test_pkl_path}")

print("\nAll done! Filtered data has been saved.")
