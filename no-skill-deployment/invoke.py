"""
Send a sample inference request to the deployed Databricks Model Serving endpoint.

Prerequisites:
  - DATABRICKS_HOST env var
  - DATABRICKS_TOKEN env var
  - Endpoint must already be deployed (run deploy.py first)
"""

import os
import json
import requests

DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
ENDPOINT_NAME = "iris-rf-endpoint"

# Sample Iris feature rows: [sepal_length, sepal_width, petal_length, petal_width]
SAMPLE_INPUT = {
    "dataframe_records": [
        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3},
    ]
}

def invoke():
    url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, data=json.dumps(SAMPLE_INPUT))
    response.raise_for_status()

    print("Predictions:", response.json())


if __name__ == "__main__":
    invoke()
