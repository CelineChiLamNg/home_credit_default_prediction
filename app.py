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
