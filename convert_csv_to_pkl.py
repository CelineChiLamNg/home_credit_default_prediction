import os
import pandas as pd

# Path to your data_csv folder
data_folder = "data_csv"

# Path to the data_pkl folder
pkl_folder = "data_pkl"

# Create the new folder for pickle files
os.makedirs(pkl_folder, exist_ok=True)

# Get all files in the data_csv folder
files = os.listdir(data_folder)

# Track progress
print(f"Found {len(files)} files in the folder: {data_folder}")

# Iterate through all files in the data_csv folder
for filename in files:
    # Check if the file is a CSV
    if filename.endswith(".csv"):
        csv_path = os.path.join(data_folder, filename)  # Full path to the CSV
        pickle_path = os.path.join(pkl_folder, filename.replace(".csv", ".pkl"))  # Save in the new folder

        print(f"\nProcessing: {filename}")

        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            print(f"Read {filename} successfully. Shape: {df.shape}")

            # Save as pickle in the new folder
            df.to_pickle(pickle_path)
            print(f"Saved as pickle: {pickle_path}")

        except Exception as e:
            # Catch and print any errors
            print(f"Error processing {filename}: {e}")

print("\nAll done!")

