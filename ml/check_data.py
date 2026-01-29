# ml/inspect.py
import pandas as pd
import sys, os
CSV_PATH = "data/Predict Hair Fall.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: file not found at {CSV_PATH}")
    print("Place your CSV in ml/data/ and adjust CSV_PATH if necessary.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, nrows=50)
print("=== SHAPE ===")
print(df.shape)
print("\n=== COLUMNS ===")
for i,c in enumerate(df.columns):
    print(f"{i+1:02d}. {c!r}")
print("\n=== SAMPLE ROWS ===")
print(df.head(5).to_string(index=False))
print("\n=== DTYPE SUMMARY ===")
print(df.dtypes)
