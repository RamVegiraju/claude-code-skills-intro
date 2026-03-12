---
name: model-serving
description: "Deploying models to Databricks Model Serving, specifically use for traditional ML models and non-LLM deployments. For LLM deployments refer to the Foundation Model API skill."
---

# 1. MLflow Integration
Databricks Model Serving directly integrates with MLflow based capabilities. Utilize the built-in flavors such as PyTorch, Sci-Kit-Learn, etc to log your model either during training or logging the model weights. An example of logging native flavors is like the following transformers example:

```
# Log and register with MLflow Transformers flavor: https://mlflow.org/docs/latest/ml/deep-learning/transformers/

# Track experiment & run
with mlflow.start_run(run_name="hf_transformers_bert"):
    mlflow.transformers.log_model(
        transformers_model=hf_pipe,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
    )
    run_id = mlflow.active_run().info.run_id

model_uri = f"runs:/{run_id}/model"
print(model_uri)
```

In the case that there's a non-supported flavor, utilize the PyFunc class like the following to wrap your custom model artifacts/scripts.

Two scripts are attached:

1. PyFunc Model:

```
import json
import mlflow
import mlflow.pyfunc

class LangIdPyFunc(mlflow.pyfunc.PythonModel):
    """
    Minimal MLflow PyFunc that detects language using 'langid'.
    Input: DataFrame or dict with 'text'
    Output: JSON string, e.g., {"lang": "en", "score": 0.98}
    """
    def load_context(self, context):
        import langid
        self.langid = langid

    def predict(self, context, model_input):
        # Accept pandas DataFrame or dict with 'text'
        if hasattr(model_input, "iloc"):
            text = model_input["text"].iloc[0]
        elif isinstance(model_input, dict):
            text = model_input.get("text")
        else:
            raise ValueError("Expected DataFrame column 'text' or dict key 'text'.")
        lang, score = self.langid.classify(text)
        return json.dumps({"lang": lang, "score": float(score)})
```

2. Track and log with MLflow:

```
from mlflow.models.signature import ModelSignature, Schema, ColSpec
import mlflow

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "text")]),
    outputs=Schema([ColSpec("string", "lang_json")])
)

with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        name="langid_pyfunc",                 # MLflow 3-style name
        python_model=LangIdPyFunc(),
        signature=signature,
        pip_requirements=[
            "mlflow>=2.4.0",
            "langid"                          # small pure-Python lib
        ],
        input_example={"text": "OpenAI is based in San Francisco."}
    )
print("model_uri:", info.model_uri)
```

Once you have tracked your model as an MLflow experiment, we can focus on registering your model to Unity Catalog.

# 2. Unity Catalog Registry
Unity Catalog is the central governance platform for all data and AI assets within Databricks including your model artifacts. We leverage Unity Catalog here to register our model in a central location while pointing towards our model artifacts as well. We grab the model URI from the mlflow run that we started in the previous step:

```
# Ensure we use Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Catalog details
catalog = "custom_ml"
schema = "models" 
model_name = "hf_bert_transformers"
# Adjust model name as needed
registered_name = f"{catalog}.{schema}.{model_name}"
     

# Register explicitly in UC, issues with autolog at times depending on package version
res = mlflow.register_model(model_uri=model_uri, name=registered_name)
print("Model URI:", model_uri)
print("Registered UC model:", registered_name, "version:", res.version)
```

# 3. Deploy to a Databricks Model Serving Endpoint
After we have registered our model we can utilize the mlflow-deployments SDK to create our online endpoint for inference. First understand your model size to understand the compute needed to host, your expected traffic to configure the concurrency expected as well.


## Workload Type Options

| Workload Type | Hardware Class | Typical ML / AI Use Cases |
|----------------|---------------|---------------------------|
| CPU | CPU serverless compute | Traditional ML models (sklearn, XGBoost, LightGBM), feature engineering pipelines, low-latency REST inference, tabular prediction APIs |
| GPU_SMALL | 1× NVIDIA T4 (~16 GB VRAM) | Embedding models, small LLMs (7B quantized), image classification, lightweight NLP models, sentence transformers |
| GPU_MEDIUM | Mid-tier GPU compute | Medium transformer models (7B–13B), moderate vision models, fine-tuned LLM inference, multimodal models |
| GPU_LARGE | High-end GPU compute (often A100-class) | Large LLM inference (13B–70B), high-throughput LLM serving, generative AI pipelines, diffusion models, heavy multimodal models |


## Workload Size Options

| Workload Size | Concurrency Capacity | Typical Use Cases |
|----------------|---------------------|------------------|
| Small | ~4 concurrent requests | Development endpoints, low traffic APIs, internal tools, batch-triggered inference |
| Medium | ~8–16 concurrent requests | Production APIs with moderate traffic, embedding services, RAG pipelines |
| Large | ~16–64 concurrent requests | High-traffic production services, large scale LLM serving, enterprise chatbots, real-time AI applications |

## General Use-Case Mapping

| ML Application | workload_type | workload_size |
|----------------|---------------|---------------|
| Tabular ML prediction API | CPU | Small |
| Production recommender system | CPU | Medium |
| Embedding service for RAG | GPU_SMALL | Medium |
| Small LLM chatbot (7B) | GPU_SMALL | Medium |
| Fine-tuned LLM inference | GPU_MEDIUM | Medium |
| Large LLM serving (13B–70B) | GPU_LARGE | Large |

For advanced use-cases you can enable Provisioned Concurrency and AutoScaling as well. Only do this when you have an understanding of your load tests and what configurations are needed:

```
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

# Create or Update an endpoint with scaling enabled
w.serving_endpoints.create(
    name="my-model-endpoint",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="my_model_name",
                entity_version="1",
                workload_size="Small",
                # Scaling configuration
                min_provisioned_concurrency=1,  # Prevent cold starts
                max_provisioned_concurrency=5,  # Scale up to 5 parallel requests
                scale_to_zero_enabled=False     # Keep at least min alive
            )
        ]
    )
)
```

Other methods also include Route Optimization.

## SDK Configuration

```
import mlflow
from mlflow.deployments import get_deploy_client

#mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks") #Client to interfact with deployment & inference

# specify version
version = "1" 

# Main Deployment API
endpoint = client.create_endpoint(
    name="bert-transformers-ep",
    config={
        "served_entities": [
            {
                "name": f"{model_name}-{version}",                 # served model name on the endpoint
                "entity_name": f"{catalog}.{schema}.{model_name}", # UC model
                "entity_version": version,
                "workload_size": "Small",
                "workload_type": "CPU",
                "scale_to_zero_enabled": True
            }
        ]
    }
)

# poll the endpoint till creation, this varies for different endpoints
import time
endpoint_name = "bert-transformers-ep"

while True:
    ep = client.get_endpoint(endpoint_name)
    state = ep["state"]["ready"]

    print(f"Endpoint state: {state}")

    if state == "READY":
        print("✅ Endpoint is ready")
        break

    if state == "FAILED":
        raise RuntimeError(f"❌ Endpoint creation failed: {ep}")

    time.sleep(45)  # poll every 45s
```

# Invoke Endpoint
Once your REST Endpoint has been created you should see it in the Serving tab in your UI. You can then invoke it using the same Deployment SDK, ensure the payload is structured and sized in the same way you tracked in MLflow.

```
# sample payload, shape: 
payload = {
  "dataframe_split": {
    "columns": ["text"],
    "data": [["I am so angry and upset."]]
  }
}
print(payload)
# Invoke the endpoint with your raw payload
resp = client.predict(
    endpoint=endpoint_name,
    inputs=payload,
)
print(resp)
```

# Deployment Examples

See the `examples/` directory for complete notebooks:

| File | Use-Case |
|------|--------------|-------------|
| `` | Notebook | Model Deployment |
| `dbx-serving-sklearn.ipynb` | Deploying a sci-kit learn model |
| `transformers-dbx-serving.ipynb` | Deploying a transformers model |