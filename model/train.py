"""
Train a heart disease classifier and save it as a joblib file.

Usage:
    python -m model.train

This script loads the Heart Disease CSV, trains a Random Forest classifier,
prints evaluation metrics, and saves the model to model/heart_model.joblib.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")

    # Features and target
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    X = df[feature_cols]
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Build pipeline with scaler + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1,
        )),
    ])

    # Train
    print("\nTraining Random Forest classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "heart_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
