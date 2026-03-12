"""
Train a simple scikit-learn LinearRegression model and register it to
Databricks Unity Catalog via MLflow.

Unity Catalog target:
  Catalog : custom_ml
  Schema  : models
  Model   : sklearn_regression_sample
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# ── Unity Catalog config ─────────────────────────────────────────────────────
CATALOG    = "custom_ml"
SCHEMA     = "models"
MODEL_NAME = "sklearn_regression_sample"
REGISTERED_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Point MLflow at the Databricks UC model registry
mlflow.set_registry_uri("databricks-uc")

# ── Generate synthetic regression data ───────────────────────────────────────
rng = np.random.default_rng(42)
n   = 400

X = pd.DataFrame({
    "age":    rng.integers(18, 65, size=n).astype(float),
    "income": rng.normal(50_000, 15_000, size=n),
    "score":  rng.uniform(0, 100, size=n),
})

y = (
    0.5  * X["age"]
    + 0.0003 * X["income"]
    + 1.2  * X["score"]
    + rng.normal(0, 5, n)
)

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train + log with MLflow sklearn autolog ───────────────────────────────────
mlflow.sklearn.autolog(
    log_input_examples=True,
    registered_model_name=REGISTERED_NAME,
)

with mlflow.start_run(run_name="sklearn_regression_sample"):
    model = LinearRegression()
    model.fit(train_x, train_y)

    preds = model.predict(test_x)
    rmse  = np.sqrt(mean_squared_error(test_y, preds))
    r2    = r2_score(test_y, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",   r2)

    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

print(f"\nModel registered to Unity Catalog: {REGISTERED_NAME}")
