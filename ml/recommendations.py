import pandas as pd
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# --- Load product datasets ---
serum_df = pd.read_csv(os.path.join(DATA_DIR, "serum.csv"))
shampoo_df = pd.read_csv(os.path.join(DATA_DIR, "shampoo_data.csv"))
conditioner_df = pd.read_csv(os.path.join(DATA_DIR, "shampoo_conditoner.csv"))

def get_nutrition_suggestions(pred_input):
    """Suggest nutrition tips based on possible deficiencies"""
    if "nutrition" in pred_input.lower():
        return random.choice([
            "Protein-rich diet (eggs, fish, pulses)",
            "Vitamin D (sunlight, fortified milk)",
            "Iron and Zinc supplements",
        ])
    elif "stress" in pred_input.lower():
        return random.choice([
            "Increase magnesium intake",
            "Hydrate well and sleep at least 7 hours",
        ])
    else:
        return random.choice([
            "Balanced diet with leafy greens and nuts",
            "Biotin supplements and adequate hydration",
        ])

def get_product_suggestions():
    """Pick random items from product datasets"""
    shampoos = shampoo_df.sample(1)["Product Name"].values.tolist()
    serums = serum_df.sample(1)["Product Name"].values.tolist()
    return shampoos + serums

def get_reason_from_features(features):
    """Generate the 'Why' field intelligently"""
    reasons = []
    mapping = {
        "genetics": "Genetic factors",
        "stress": "Stress or lifestyle issues",
        "nutritional_deficiencies": "Nutrient imbalance",
        "hormonal_changes": "Hormonal fluctuations",
        "poor_hair_care_habits": "Improper hair care habits",
    }
    for key, desc in mapping.items():
        if key in features and features[key] in ["yes", "high", "moderate", "True", 1]:
            reasons.append(desc)
    return ", ".join(reasons) if reasons else "General hair health imbalance"
