"""
Train and save a simple scikit-learn model artifact for Databricks deployment.
Saves the model in MLflow format, which Databricks Model Serving expects.
"""

import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "iris_rf_model"


def train_and_save():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.4f}")

    mlflow.sklearn.save_model(model, MODEL_PATH)
    print(f"Model artifact saved to: {MODEL_PATH}/")


if __name__ == "__main__":
    train_and_save()
