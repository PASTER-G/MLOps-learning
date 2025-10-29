from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import tempfile
import os
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

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

def train_logistic_regression(**kwargs):
    """Обучение Logistic Regression с логированием в MLflow"""
    
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow-service:5000")  # в docker-compose
    mlflow.set_experiment("Logistic_Regression")
    
    with mlflow.start_run(run_name="logistic_regression_best_params"):
        # Параметры модели
        params = {
            'max_iter': 2000,
            'random_state': 42,
            'C': 0.0001,
            'penalty': 'l2',
            'solver': 'liblinear',
            'class_weight': {0: 1, 1: 7}
        }
        
        # Логирование параметров
        mlflow.log_params(params)
        
        # Обучение модели
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Расчет метрик
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Детальный отчет классификации
        class_report = classification_report(y_test, y_pred, output_dict=True)
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"class_{class_name}_{metric_name}", value)
        
        # Логирование модели
        mlflow.sklearn.log_model(model, "model")
        
        # Логирование дополнительной информации
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("data_source", "processed_csv_files")
        
        # Сохранение метрик для последующего сравнения
        metrics = {
            'model_name': 'LogisticRegression',
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'run_id': mlflow.active_run().info.run_id
        }
        
        # Сохранение метрик во временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            temp_path = f.name
        
        # Передача пути к файлу с метриками через XCom
        kwargs['ti'].xcom_push(key='metrics_path', value=temp_path)
        
        print("=== Logistic Regression Training Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

def save_metrics_locally(**kwargs):
    """Сохранение метрик локально (вместо S3)"""
    ti = kwargs['ti']
    metrics_path = ti.xcom_pull(key='metrics_path', task_ids='train_lr')
    
    # Загрузка метрик
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
    
    # Очистка временного файла
    os.unlink(metrics_path)

# Определение DAG
with DAG(
    'train_logistic_regression',
    default_args=default_args,
    description='Train Logistic Regression model with MLflow tracking',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=['mlops', 'training', 'logistic_regression'],
) as dag:

    train_task = PythonOperator(
        task_id='train_lr',
        python_callable=train_logistic_regression,
        provide_context=True,
    )

    save_metrics_task = PythonOperator(
        task_id='save_metrics',
        python_callable=save_metrics_locally,
        provide_context=True,
    )

    train_task >> save_metrics_task