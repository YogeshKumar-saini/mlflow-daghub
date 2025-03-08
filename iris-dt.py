import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

import dagshub
from mlflow.models.signature import infer_signature

# Initialize Dagshub tracking
dagshub.init(repo_owner='YogeshKumar-saini', repo_name='mlflow-dagshub-demo', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/YogeshKumar-saini/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameter
max_depth = 10

# Start MLflow experiment
mlflow.set_experiment('iris-dt')

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)

    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)

    # ✅ Add Input Example & Signature
    input_example = pd.DataFrame(X_test[:1], columns=iris.feature_names)
    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(dt, "decision_tree_model", input_example=input_example, signature=signature)

    # Add metadata
    mlflow.set_tag('author', 'Yogesh')
    mlflow.set_tag('model', 'Decision Tree')

    print('Model Accuracy:', accuracy)

