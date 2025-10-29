# airflow/dags/preprocess_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import sys

# === Добавляем src в PYTHONPATH ===
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from preprocess import preprocess_data

# === Параметры DAG ===
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

with DAG(
    dag_id="preprocess_dag",
    default_args=default_args,
    description="Предобработка исходных данных для обучения моделей",
    schedule_interval=None,  # запуск вручную
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=True,
    tags=["data", "preprocessing"],
) as dag:

    def run_preprocessing():
        raw_path = str(BASE_DIR / "data" / "raw" / "dataset.csv")
        output_dir = str(BASE_DIR / "data" / "processed")

        preprocess_data(raw_path, output_dir)
        
    preprocess_task = PythonOperator(
        task_id="preprocess_dataset",
        python_callable=run_preprocessing,
    )

    preprocess_task
