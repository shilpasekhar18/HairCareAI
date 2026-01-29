import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load trained pipeline
model_path = os.path.join(MODEL_DIR, "hairfall_model.pkl")
model_pipeline = joblib.load(model_path)

# Load feature names
feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
feature_names = joblib.load(feature_names_path)

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input into DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all required features exist
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = None

        # Reorder columns exactly as during training
        input_df = input_df[feature_names]

        # Predict
        pred_proba = model_pipeline.predict_proba(input_df)[0][1]
        pred_class = int(model_pipeline.predict(input_df)[0])

        # Build explanation
        causes = []
        if data.get("genetics", "").lower() == "yes":
            causes.append("Genetics")
        if data.get("stress", "").lower() in ["high", "moderate"]:
            causes.append("Stress")
        if data.get("nutritional_deficiencies"):
            causes.append("Nutrition")
        if data.get("hormonal_changes", "").lower() == "yes":
            causes.append("Hormonal changes")

        response = {
            "hair_fall_risk": pred_class,
            "confidence": round(pred_proba, 3),
            "why": "/".join(causes) if causes else "Lifestyle factors",
            "nutrition_suggestions": [
                "Sunlight",
                "Eggs",
                "Fortified milk"
            ],
            "product_suggestions": [
                "Mild Shampoo",
                "Biotin Supplements",
                "Hair Oils"
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
