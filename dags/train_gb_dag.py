from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.utils import class_weight
import tempfile
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import sys

# === Добавляем src в PYTHONPATH ===
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from metrics_utils import evaluate_with_optimal_threshold_mlflow

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0
}

def load_processed_data():
    """Загрузка предобработанных данных из файлов"""
    
    X_train = pd.read_csv(f"{BASE_DIR}/data/processed/X_train.csv")
    X_test = pd.read_csv(f"{BASE_DIR}/data/processed/X_test.csv")
    y_train = pd.read_csv(f"{BASE_DIR}/data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{BASE_DIR}/data/processed/y_test.csv").squeeze()
    
    print(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def convert_numpy_types(obj):
    """Рекурсивно преобразует numpy типы в стандартные Python типы"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def train_gradient_boosting(**kwargs):
    """Обучение Gradient Boosting (XGBoost) с оптимальным порогом"""
    
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_processed_data()

    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    mlflow.set_experiment("Gradient_Boosting_Optimal")
    
    with mlflow.start_run(run_name="xgboost_optimal_threshold"):
        # Вычисление весов классов
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        
        # Параметры модели (из вашего кода)
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'scale_pos_weight': 1,
            'learning_rate': 0.1,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'gamma': 0.5,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': 2
        }
        
        # Логирование параметров
        mlflow.log_params(params)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("sample_weights", "balanced")
        
        # Обучение модели
        model = XGBClassifier(**params)
        
        # Обучение с весами и eval_set
        model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_train, y_train)], 
            verbose=False, 
            sample_weight=classes_weights
        )
        
        # Предсказания вероятностей
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Оценка с оптимальным порогом
        optimal_metrics = evaluate_with_optimal_threshold_mlflow(y_test, y_pred_proba)
        
        # Логирование метрик в MLflow
        mlflow.log_metric("optimal_threshold", optimal_metrics["threshold"])
        mlflow.log_metric("roc_auc_optimal", optimal_metrics["roc_auc"])
        mlflow.log_metric("accuracy_optimal", optimal_metrics["accuracy"])
        mlflow.log_metric("precision_optimal", optimal_metrics["precision"])
        mlflow.log_metric("recall_optimal", optimal_metrics["recall"])
        mlflow.log_metric("f1_score_optimal", optimal_metrics["f1"])
        
        # Логирование модели
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("threshold_method", "optimal_youden")
        
        # Логирование информации о данных
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        
        # Сохранение метрик для XCom
        metrics = {
            'model_name': 'XGBoost_Optimal',
            **optimal_metrics,
            'run_id': mlflow.active_run().info.run_id
        }

        metrics = convert_numpy_types(metrics)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            temp_path = f.name
        
        kwargs['ti'].xcom_push(key='metrics_path', value=temp_path)
        
        print("=== Gradient Boosting (XGBoost) with Optimal Threshold ===")
        for metric, value in optimal_metrics.items():
            print(f"{metric}: {value:.4f}")

def save_metrics_locally(**kwargs):
    """Сохранение метрик локально"""
    ti = kwargs['ti']
    metrics_path = ti.xcom_pull(key='metrics_path', task_ids='train_gb')
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Сохраняем метрики в локальную директорию
    output_dir = f"{BASE_DIR}/data/models/metrics"
    output_path = f"{output_dir}/{metrics['model_name']}_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"=== Metrics saved locally ===")
    print(f"Model: {metrics['model_name']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Metrics saved to: {output_path}")

    os.unlink(metrics_path)

with DAG(
    'train_gradient_boosting_optimal',
    default_args=default_args,
    description='Train Gradient Boosting (XGBoost) with optimal threshold selection',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=['mlops', 'training', 'xgboost', 'gradient_boosting', 'optimal_threshold'],
) as dag:

    train_task = PythonOperator(
        task_id='train_gb',
        python_callable=train_gradient_boosting,
        provide_context=True,
    )

    save_metrics_task = PythonOperator(
        task_id='save_metrics',
        python_callable=save_metrics_locally,
        provide_context=True,
    )

    train_task >> save_metrics_task