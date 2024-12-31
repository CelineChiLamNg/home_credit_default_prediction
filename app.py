from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input
    data = request.get_json(force=True)
    df = pd.DataFrame(data)

    # Encode for aggregation (complementary tables)

    # Aggregate tables to bureau and previous application

    # aggregate to main table Application

    # Encode bureau_balance, credit_card_balance, POS_CASH, installments_payments
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

    # Aggregate to bureau and previous_application
    bureau_aggregator = BureauBalanceAggregator(agg_funcs=['mean', 'sum', 'min', 'max', 'count'])
    bureau_transformed = bureau_aggregator.transform(
        (test_splits['bureau_test'], bureau_balance_encoded)
    )

    previous_application_aggregator = PreviousApplicationAggregator(agg_funcs=['mean', 'sum', 'min', 'max', 'count'])
    previous_application_transformed = previous_application_aggregator.transform(
        (previous_application_encoded, credit_card_balance_encoded,
         test_splits['installments_payments_test'], POS_CASH_balance_encoded)
    )

    # Aggregate bureau and previous_application to main table
    data_merged_test = get_data_merged_test(bureau_transformed,
                                            previous_application_transformed,
                                            test_splits)

    # Preprocess the data
    df_processed = pd.DataFrame(preprocessor_encode.transform(df),
                                columns=preprocessor_encode.get_feature_names_out())
    fc = FeatureCreation2()
    df_processed = fc.transform(df_processed)

    # Predict probabilities and apply threshold
    proba = model.predict_proba(df_processed)[:, 1]
    predictions = (proba >= best_threshold).astype(int)

    # Return predictions
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
