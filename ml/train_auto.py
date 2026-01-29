# ml/train_auto.py
import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH = "data/Predict Hair Fall.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
print("Loaded", CSV_PATH, "shape:", df.shape)

# auto-detect target column
possible_targets = ["Hair Loss","hair_loss","HairLoss","Hair loss","hair fall","HairFall","hair_fall","target","label"]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    for c in df.columns:
        if df[c].nunique() == 2:
            target_col = c
            print(f"Auto-detected target column as '{c}' (2 unique values).")
            break

if target_col is None:
    print("Could not auto-detect target. Run inspect.py and edit this script to set target_col.")
    print("Columns:", list(df.columns[:15]))
    raise SystemExit(1)

print("Using target column:", target_col)
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()

# drop very high-cardinality text (likely free text)
high_card = [c for c in X.columns if X[c].dtype == object and X[c].nunique() > 200]
if high_card:
    print("Dropping high-card columns:", high_card)
    X = X.drop(columns=high_card)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

num_pipeline = make_pipeline(num_imputer, StandardScaler())
cat_pipeline = make_pipeline(cat_imputer, encoder)

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ],
    remainder="drop"
)

X_proc = preprocessor.fit_transform(X)
print("Processed shape:", X_proc.shape)

# feature names (for reference)
feature_names = list(numeric_cols)
if cat_cols:
    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehotencoder'].get_feature_names_out(cat_cols)
    feature_names += list(cat_names)
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# train/test
X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(MODEL_DIR, "hairfall_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
print("Saved model and preprocessor to", MODEL_DIR)
