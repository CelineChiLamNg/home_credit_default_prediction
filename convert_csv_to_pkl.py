import os
import pandas as pd

data_folder = "data_csv"
pkl_folder = "data_pkl"

os.makedirs(pkl_folder, exist_ok=True)
files = os.listdir(data_folder)

excluded_files = {'sample_submission.csv', 'HomeCredit_columns_description.csv'}

print(f"Found {len(files)} files in the folder: {data_folder}")

for filename in files:
    if filename.endswith(".csv") and filename not in excluded_files:
        csv_path = os.path.join(data_folder, filename)
        pickle_path = os.path.join(pkl_folder, filename.replace(".csv", ".pkl"))

        print(f"\nProcessing: {filename}")

        try:
            df = pd.read_csv(csv_path)
            print(f"Read {filename} successfully. Shape: {df.shape}")

            df.to_pickle(pickle_path)
            print(f"Saved as pickle: {pickle_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nAll done!")

