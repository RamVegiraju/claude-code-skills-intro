import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Config
ROLE = "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>"
REGION = "us-east-1"
BUCKET = "<your-s3-bucket>"
MODEL_S3_KEY = "models/model.tar.gz"
ENDPOINT_NAME = "dummy-torch-endpoint"

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)


def upload_model():
    s3 = boto_session.client("s3")
    s3.upload_file("model.tar.gz", BUCKET, MODEL_S3_KEY)
    model_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"
    print(f"Model uploaded to {model_uri}")
    return model_uri


def deploy(model_uri):
    model = PyTorchModel(
        model_data=model_uri,
        role=ROLE,
        framework_version="2.0",
        py_version="py310",
        sagemaker_session=sagemaker_session,
        entry_point="inference.py",  # custom inference script if needed
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"Endpoint deployed: {ENDPOINT_NAME}")
    return predictor


if __name__ == "__main__":
    model_uri = upload_model()
    deploy(model_uri)
