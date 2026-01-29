import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading PredictHairFall.csv")
df = pd.read_csv(os.path.join(DATA_DIR, "PredictHairFall.csv"))

# -----------------------------
# 2. Clean column names
# -----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Target
target_col = "hair_loss"
y = df[target_col]

# Features we EXPLICITLY want
feature_cols = [
    "age",
    "genetics",
    "stress",
    "nutritional_deficiencies",
    "hormonal_changes",
    "poor_hair_care_habits",
    "medical_conditions",
    "smoking",
    "weight_loss"
]

X = df[feature_cols]

# -----------------------------
# 3. Define column types
# -----------------------------
numeric_features = ["age"]
categorical_features = [
    col for col in feature_cols if col != "age"
]

# -----------------------------
# 4. Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# 5. Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# 6. Train / Validate
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {acc:.3f}")

# -----------------------------
# 7. Save
# -----------------------------
joblib.dump(pipeline, os.path.join(MODEL_DIR, "hairfall_model.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_names.pkl"))

print("Model saved successfully")
