import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import tempfile
import os

def evaluate_with_optimal_threshold_mlflow(y_true, y_proba, run_name="optimal_threshold"):
    """Версия для Airflow с логированием в MLflow"""
    
    # 1. ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # 2. Оптимальный порог по Youden's J
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # 3. Предсказания с оптимальным порогом
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # 4. Метрики
    roc_auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Логируем в MLflow
    mlflow.log_metric("optimal_threshold", optimal_threshold)
    mlflow.log_metric("roc_auc_optimal", roc_auc)
    mlflow.log_metric("accuracy_optimal", accuracy)
    mlflow.log_metric("precision_optimal", precision)
    mlflow.log_metric("recall_optimal", recall)
    mlflow.log_metric("f1_score_optimal", f1)

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {
        "threshold": optimal_threshold,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }