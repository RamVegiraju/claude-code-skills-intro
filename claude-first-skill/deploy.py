"""
Deploy the registered scikit-learn model from Unity Catalog to a
Databricks Model Serving real-time endpoint, then run a sample inference.

Prerequisites:
  - train.py has been run and the model is registered in UC.
  - DATABRICKS_HOST and DATABRICKS_TOKEN env vars are set (or ~/.databrickscfg exists).
"""

import time
import mlflow
from mlflow.deployments import get_deploy_client

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG       = "custom_ml"
SCHEMA        = "models"
MODEL_NAME    = "sklearn_regression_sample"
MODEL_VERSION = "1"           # bump if you re-ran train.py
ENDPOINT_NAME = "sklearn-regression-sample-ep"

# ── MLflow / deployment client ────────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

# ── Create the serving endpoint ───────────────────────────────────────────────
# sklearn is a tabular CPU workload → CPU / Small is the right fit
print(f"Creating endpoint '{ENDPOINT_NAME}' ...")
client.create_endpoint(
    name=ENDPOINT_NAME,
    config={
        "served_entities": [
            {
                "name": f"{MODEL_NAME}-v{MODEL_VERSION}",
                "entity_name": f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
                "entity_version": MODEL_VERSION,
                "workload_type": "CPU",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    },
)

# ── Poll until ready ──────────────────────────────────────────────────────────
print("Waiting for endpoint to become ready (this may take a few minutes) ...")
while True:
    ep    = client.get_endpoint(ENDPOINT_NAME)
    state = ep["state"]["ready"]
    print(f"  Endpoint state: {state}")

    if state == "READY":
        print("✅ Endpoint is ready")
        break
    if state == "FAILED":
        raise RuntimeError(f"❌ Endpoint creation failed: {ep}")

    time.sleep(30)

# ── Sample inference ──────────────────────────────────────────────────────────
payload = {
    "dataframe_split": {
        "columns": ["age", "income", "score"],
        "data": [
            [35, 62000.0, 78.5],
        ],
    }
}

print("\nRunning sample inference ...")
response = client.predict(endpoint=ENDPOINT_NAME, inputs=payload)
print("Prediction response:", response)
