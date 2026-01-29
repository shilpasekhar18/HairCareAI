import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ======================
# App initialization (MUST be first)
# ======================
app = Flask(__name__)
CORS(app)

# ======================
# Paths
# ======================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ======================
# Load model artifacts
# ======================
model_pipeline = joblib.load(os.path.join(MODEL_DIR, "hairfall_model.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# ======================
# Health check route (VERY IMPORTANT FOR RENDER)
# ======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "HaircareAI API is running"}), 200

# ======================
# Prediction route
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])

        # Ensure all required features exist
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = None

        input_df = input_df[feature_names]

        pred_proba = model_pipeline.predict_proba(input_df)[0][1]
        pred_class = int(model_pipeline.predict(input_df)[0])

        causes = []
        if data.get("genetics", "").lower() == "yes":
            causes.append("Genetics")
        if data.get("stress", "").lower() in ["high", "medium", "moderate"]:
            causes.append("Stress")
        if data.get("nutritional_deficiencies"):
            causes.append("Nutrition")

        return jsonify({
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
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
