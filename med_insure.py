import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import mlflow
import mlflow.sklearn
from urllib.parse import quote_plus
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASS = "142789"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Medical_Insurance_Cost_Prediction_db"

# Encode password safely for URL use
encoded_pass = quote_plus(DB_PASS)

# ✅ Correct connection string (f-string with actual substitution)
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_sql(r'SELECT * FROM "Medical_Insurance";', engine)
for col in ["age", "bmi", "children", "charges"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["charges"]).reset_index(drop=True)
# 1. Features / Target
# -----------------------------
target = "charges"
cat_cols = ["sex", "smoker", "region"]
exclude_cols = cat_cols + [target, "created_at"]  # exclude datetime column
num_cols = [c for c in df.columns if c not in exclude_cols]

X = df[num_cols + cat_cols].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 2. Preprocessor
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)



# 3. Candidate Models
# -----------------------------
pipelines = {
    "LinearRegression": Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ]),
}

if HAS_XGB:
    pipelines["XGBoost"] = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ))
    ])

# 4. MLflow experiment
# -----------------------------
mlflow.set_experiment("Medical Insurance Models")

best_name, best_r2, best_pipe = None, -1e9, None

for name, pipe in pipelines.items():
    with mlflow.start_run(run_name=name):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2",  r2)

        input_example = X_train.iloc[:1]
        signature = infer_signature(input_example, pipe.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # Track best
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

print(f"Best model: {best_name} (R²={best_r2:.4f})")

payload = {
    "pipeline": best_pipe,
    "feature_order": X.columns.tolist(),
}
joblib.dump(payload, "best_model.pkl")
print("✅ Saved best pipeline to best_model.pkl")