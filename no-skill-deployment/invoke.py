import boto3
import json
import numpy as np

ENDPOINT_NAME = "dummy-torch-endpoint"
REGION = "us-east-1"

client = boto3.client("sagemaker-runtime", region_name=REGION)


def invoke(payload):
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode())
    print(f"Response: {result}")
    return result


if __name__ == "__main__":
    # Dummy input: batch of 1, 10 features
    payload = {"inputs": np.random.rand(1, 10).tolist()}
    invoke(payload)
