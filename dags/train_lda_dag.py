from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

def train_linear_discriminant_analysis(**kwargs):
    """Обучение Linear Discriminant Analysis"""
    
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    mlflow.set_experiment("Linear_Discriminant_Analysis")
    
    with mlflow.start_run(run_name="lda_standard"):
        # Параметры модели (из вашего кода)
        priors = [0.5241379310344827, 0.47586206896551725]
        params = {
            'priors': priors
        }
        
        # Логирование параметров
        mlflow.log_params({'priors': priors})
        
        # Обучение модели
        model = LinearDiscriminantAnalysis(priors=priors)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Расчет метрик
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Логирование метрик в MLflow
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Логирование модели
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_type", "LinearDiscriminantAnalysis")
        mlflow.log_param("threshold", "default_0.5")
        
        # Логирование информации о данных
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        
        # Сохранение метрик для XCom
        metrics = {
            'model_name': 'LinearDiscriminantAnalysis',
            'roc_auc': float(roc_auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'run_id': mlflow.active_run().info.run_id
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            temp_path = f.name
        
        kwargs['ti'].xcom_push(key='metrics_path', value=temp_path)
        
        print("=== Linear Discriminant Analysis Results ===")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

def save_metrics_locally(**kwargs):
    """Сохранение метрик локально"""
    ti = kwargs['ti']
    metrics_path = ti.xcom_pull(key='metrics_path', task_ids='train_lda')
    
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
    'train_linear_discriminant_analysis',
    default_args=default_args,
    description='Train Linear Discriminant Analysis model',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=['mlops', 'training', 'lda', 'linear_discriminant_analysis'],
) as dag:

    train_task = PythonOperator(
        task_id='train_lda',
        python_callable=train_linear_discriminant_analysis,
        provide_context=True,
    )

    save_metrics_task = PythonOperator(
        task_id='save_metrics',
        python_callable=save_metrics_locally,
        provide_context=True,
    )

    train_task >> save_metrics_task