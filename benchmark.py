import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"


def timed(label, fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return value, elapsed


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df, load_seconds = timed("Load data", lambda: pd.read_csv(DATA_PATH))

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=None,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    _, training_seconds = timed("Training", lambda: model.fit(X_train, y_train))

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    one_row = X_test.iloc[[0]]
    start = time.perf_counter()
    _ = model.predict_proba(one_row)
    one_row_latency_ms = (time.perf_counter() - start) * 1000

    batch = X_test.iloc[:1000]
    start = time.perf_counter()
    _ = model.predict_proba(batch)
    batch_seconds = time.perf_counter() - start
    throughput_rows_per_second = len(batch) / batch_seconds

    metrics = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "load_data_seconds": load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(getattr(model, "best_iteration_", None) or model.n_estimators),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "inference_latency_ms_1_row": one_row_latency_ms,
        "inference_throughput_rows_per_second_1000_rows": throughput_rows_per_second,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nBenchmark metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
