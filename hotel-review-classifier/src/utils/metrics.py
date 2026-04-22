from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np


def compute_classification_metrics(y_true, y_pred, prefix: str = "") -> dict:
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
    }


def compute_regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    prefix = f"{prefix}_" if prefix else ""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        f"{prefix}mae": float(np.mean(np.abs(y_true - y_pred))),
        f"{prefix}rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def print_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
