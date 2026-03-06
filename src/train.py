# src/train.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

DATA_PATH = os.getenv("DATA_PATH", "data/housing.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "model.joblib")
METRICS_OUT = os.getenv("METRICS_OUT", "metrics.json")

def load_data(path):
    df = pd.read_csv(path)
    # simple feature engineering — drop non-numeric if present
    X = df.drop(columns=["median_house_value"], errors="ignore")
    # If median_house_value column exists:
    if "median_house_value" in df.columns:
        y = df["median_house_value"]
    else:
        raise SystemExit("median_house_value column not found")
    return X.select_dtypes(include=[np.number]).fillna(0), y

def train():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, MODEL_OUT)
    metrics = {"rmse": float(rmse), "r2": float(r2), "dataset_size": int(len(X))}
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model ->", MODEL_OUT)
    print("Saved metrics ->", METRICS_OUT)
    print(metrics)

if __name__ == "__main__":
    train()
