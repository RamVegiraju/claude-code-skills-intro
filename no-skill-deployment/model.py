import torch
import torch.nn as nn
import tarfile
import os

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def save_model():
    model = DummyModel()
    model.eval()

    os.makedirs("model_artifacts", exist_ok=True)
    model_path = "model_artifacts/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # SageMaker expects model artifacts as model.tar.gz
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add(model_path, arcname="model.pth")
    print("Model serialized to model.tar.gz")


if __name__ == "__main__":
    save_model()
