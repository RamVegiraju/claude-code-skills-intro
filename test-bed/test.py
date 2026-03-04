import sagemaker
from sagemaker.core import image_uris

# Get framework image
inference_image = image_uris.retrieve(
    framework="pytorch",
    region="us-west-2",
    version="1.12.0",
    py_version="py38",
    instance_type="ml.p3.2xlarge",
    image_scope="inference"
)
print(inference_image)