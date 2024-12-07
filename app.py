from flask import Flask, request, jsonify, Response
import pandas as pd
import pickle
from utils.custom_preprocessor import BureauBalanceAggregator, \
    PreviousApplicationAggregator, FeatureCreation2, FrequencyEncoder
from utils.data_preparation import encode_object_columns_with_ordinal
from utils.get_data_merged_train import get_data_merged_test

# Initialize Flask app
app = Flask(__name__)

@app.route('/upload_csvs', methods=['POST'])
def upload_csvs():
    # Get all the files from the request under one key
    files = request.files.getlist('files[]')

    # Check if exactly 7 files are uploaded
    if len(files) != 7:
        return jsonify({"error": "You must upload exactly 7 CSV files."}), 400

    # Initialize variables to store each DataFrame
    POS_CASH_balance_test = None
    bureau_test = None
    data_test = None
    previous_application_test = None
    bureau_balance_test = None
    credit_card_balance_test = None
    installments_payments_test = None

    # Loop through the files and read them into pandas dataframes
    for file in files:
        if file.filename.endswith('.csv'):
            # Read CSV file into a DataFrame
            dataframe = pd.read_csv(file)

            # Store it in the corresponding variable based on the file name
            if file.filename == 'POS_CASH_balance_test.csv':
                POS_CASH_balance_test = dataframe
            elif file.filename == 'bureau_test.csv':
                bureau_test = dataframe
            elif file.filename == 'data_test.csv':
                data_test = dataframe
            elif file.filename == 'previous_application_test.csv':
                previous_application_test = dataframe
            elif file.filename == 'bureau_balance_test.csv':
                bureau_balance_test = dataframe
            elif file.filename == 'credit_card_balance_test.csv':
                credit_card_balance_test = dataframe
            elif file.filename == 'installments_payments_test.csv':
                installments_payments_test = dataframe
            else:
                return jsonify({"error": f"Unexpected file name: {file.filename}"}), 400
        else:
            return jsonify({"error": f"{file.filename} is not a CSV file."}), 400

    test_splits = {
        "POS_CASH_balance_test": POS_CASH_balance_test,
        "bureau_test": bureau_test,
        "data_test": data_test,
        "previous_application_test": previous_application_test,
        "bureau_balance_test": bureau_balance_test,
        "credit_card_balance_test": credit_card_balance_test,
        "installments_payments_test": installments_payments_test
    }

    # Encode for aggregation
    bureau_balance_encoded = encode_object_columns_with_ordinal(
        test_splits['bureau_balance_test']
    )
    credit_card_balance_encoded = encode_object_columns_with_ordinal(
        test_splits['credit_card_balance_test']
    )
    POS_CASH_balance_encoded = encode_object_columns_with_ordinal(
        test_splits['POS_CASH_balance_test']
    )
    previous_application_encoded = encode_object_columns_with_ordinal(
        test_splits['previous_application_test']
    )

    bureau_aggregator = BureauBalanceAggregator(agg_funcs=['mean', 'sum', 'min', 'max', 'count'])
    bureau_transformed = bureau_aggregator.transform(
        (test_splits['bureau_test'], bureau_balance_encoded)
    )

    previous_application_aggregator = PreviousApplicationAggregator(agg_funcs=['mean', 'sum', 'min', 'max', 'count'])
    previous_application_transformed = previous_application_aggregator.transform(
        (previous_application_encoded, credit_card_balance_encoded,
         test_splits['installments_payments_test'], POS_CASH_balance_encoded)
    )

    data_merged_test = get_data_merged_test(bureau_transformed,
                                            previous_application_transformed,
                                            test_splits)

    with open('results/preprocessor_encode.pkl', 'rb') as f:
        preprocessor_encode = pickle.load(f)

    X_test_processed = pd.DataFrame(preprocessor_encode.transform(data_merged_test),
                                    columns=preprocessor_encode.get_feature_names_out())
    fc = FeatureCreation2()
    X_test_processed = fc.transform(X_test_processed)

    model_path = "results/final_results.pkl"
    with open(model_path, 'rb') as file:
        final_results = pickle.load(file)
        model = final_results['Model']

    proba = model.predict_proba(X_test_processed)[:, 1]
    best_threshold = 0.6646
    predictions = (proba >= best_threshold).astype(int)

    # Add predictions to the data_merged_test DataFrame
    data_merged_test['predictions'] = predictions
    # Select only the first column and the predictions column
    reduced_data = data_merged_test.iloc[:, [0]].copy()  # Select the first column
    reduced_data['predictions'] = predictions  # Add predictions column

    # Convert to CSV
    csv_data = reduced_data.to_csv(index=False)

    # Print file size and dimensions
    file_size = len(csv_data.encode('utf-8')) / (1024 * 1024)  # Convert to MB
    num_rows, num_cols = reduced_data.shape
    print(f"CSV File Size: {file_size:.2f} MB")
    print(f"Number of Rows: {num_rows}")
    print(f"Number of Columns: {num_cols}")

    # Return the response
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=data_with_predictions.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)
