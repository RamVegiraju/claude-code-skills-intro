"""
Deploy a local MLflow sklearn model artifact to a Databricks Model Serving real-time endpoint.

Prerequisites:
  - DATABRICKS_HOST env var: workspace URL (e.g. https://adb-xxx.azuredatabricks.net)
  - DATABRICKS_TOKEN env var: personal access token with Model Registry + Serving permissions
  - Run model.py first to generate the local artifact

Flow:
  1. Register the local MLflow artifact to Unity Catalog (custom_ml.models.iris_rf)
  2. Create or update a Model Serving real-time endpoint targeting that model version
"""

import os
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# ── Config ─────────────────────────────────────────────────────────────────────
DATABRICKS_HOST  = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

MODEL_PATH       = "iris_rf_model"                  # local artifact from model.py
REGISTERED_MODEL = "custom_ml.models.iris_rf"       # Unity Catalog: catalog.schema.model
ENDPOINT_NAME    = "iris-rf-endpoint"
# ──────────────────────────────────────────────────────────────────────────────


def register_model() -> str:
    """Register the local MLflow artifact to Unity Catalog and return the new version string."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()

    # Ensure the registered model exists in UC
    try:
        client.create_registered_model(REGISTERED_MODEL)
        print(f"Created registered model: {REGISTERED_MODEL}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e).lower():
            print(f"Registered model already exists: {REGISTERED_MODEL}")
        else:
            raise

    # Log the local artifact as a new MLflow run and register it
    with mlflow.start_run(run_name="iris-rf-register") as run:
        mlflow.sklearn.log_model(
            sk_model=mlflow.sklearn.load_model(MODEL_PATH),
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
        )
        run_id = run.info.run_id

    # Resolve the version that was just created
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    latest_version = max(versions, key=lambda v: int(v.version)).version
    print(f"Registered version {latest_version} (run: {run_id})")
    return latest_version


def deploy_endpoint(model_version: str):
    """Create or update the Model Serving endpoint and wait until it is ready."""
    w = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

    served_entity = ServedEntityInput(
        entity_name=REGISTERED_MODEL,
        entity_version=model_version,
        workload_size="Small",        # Small | Medium | Large
        scale_to_zero_enabled=True,
    )

    config = EndpointCoreConfigInput(served_entities=[served_entity])

    existing = [e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME]
    if existing:
        print(f"Endpoint '{ENDPOINT_NAME}' exists — updating config to version {model_version}")
        w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
    else:
        print(f"Creating endpoint '{ENDPOINT_NAME}' with version {model_version}")
        w.serving_endpoints.create(name=ENDPOINT_NAME, config=config)

    # Poll until the endpoint finishes updating
    print("Waiting for endpoint to become ready...")
    while True:
        state = w.serving_endpoints.get(ENDPOINT_NAME).state
        config_update = state.config_update if state else None
        print(f"  config_update={config_update}")
        if str(config_update) in ("EndpointStateConfigUpdate.NOT_UPDATING", "None"):
            break
        time.sleep(15)

    invocation_url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"\nEndpoint ready.\nInvocation URL: {invocation_url}")


if __name__ == "__main__":
    version = register_model()
    deploy_endpoint(version)
