from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
import json
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
report_dir = 'data/models/comparison/reports'

def get_mlflow_client():
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    return MlflowClient()

def collect_models_metrics():
    client = get_mlflow_client()
    
    # Определяем эксперименты для каждой модели
    experiments_config = {
        'logistic_regression': 'Logistic_Regression_Optimal',
        'random_forest': 'Random_Forest_Optimal',
        'gradient_boosting': 'Gradient_Boosting_Optimal',
        'lda': 'Linear_Discriminant_Analysis',
        'catboost': 'CatBoost_Optimal'
    }
    
    all_metrics = []
    for model_name, exp_name in experiments_config.items():
        try:
            # Поиск эксперимента по имени
            experiment = client.get_experiment_by_name(exp_name)
            if not experiment:
                print(f"Эксперимент {exp_name} не найден")
                continue
            
            # Получение последнего успешного run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                print(f"Не найдены успешные runs для эксперимента {exp_name}")
                continue
            
            run = runs[0]
            
            # Сбор метрик
            metrics_data = {
                'model_name': model_name,
                'experiment_name': exp_name,
                'run_id': run.info.run_id,
                'start_time': run.info.start_time,
            }
            
            # Основные метрики (используем оптимальные метрики если есть)
            metrics_to_collect = [
                'roc_auc_optimal', 'roc_auc',
                'accuracy_optimal', 'accuracy',
                'precision_optimal', 'precision',
                'recall_optimal', 'recall',
                'f1_score_optimal', 'f1_score',
                'optimal_threshold'
            ]
            
            for metric_name in metrics_to_collect:
                if metric_name in run.data.metrics:
                    metrics_data[metric_name] = run.data.metrics[metric_name]
            
            all_metrics.append(metrics_data)
            print(f"Собраны метрики для {model_name}: ROC-AUC = {metrics_data.get('roc_auc_optimal', metrics_data.get('roc_auc', 'N/A'))}")
            
        except Exception as e:
            print(f"Ошибка при сборе метрик для {model_name}: {e}")
            continue
    
    # Сохранение собранных метрик во временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_metrics, f)
        temp_path = f.name
    return temp_path

def select_best_model(**kwargs):
    ti = kwargs['ti']
    metrics_path = ti.xcom_pull(task_ids='collect_metrics')
    
    # Загрузка собранных метрик
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)
    if not all_metrics:
        print("Нет метрик для сравнения!")
        return
    
    # Преобразование в DataFrame для анализа
    df_metrics = pd.DataFrame(all_metrics)
    # Предпочитаем оптимальные метрики, если они есть
    comparison_metric = 'roc_auc_optimal'
    if comparison_metric not in df_metrics.columns:
        comparison_metric = 'roc_auc'
    if comparison_metric not in df_metrics.columns:
        print(f"Метрика {comparison_metric} не найдена в данных!")
        return

    # Находим лучшую модель по основной метрике
    df_metrics[comparison_metric] = pd.to_numeric(df_metrics[comparison_metric], errors='coerce')
    best_row = df_metrics.loc[df_metrics[comparison_metric].idxmax()]
    best_model = {
        'model_name': best_row['model_name'],
        'experiment_name': best_row['experiment_name'],
        'run_id': best_row['run_id'],
        'best_metric': comparison_metric,
        'metric_value': float(best_row[comparison_metric]),
        'selection_timestamp': datetime.now().isoformat()
    }
    
    # Добавляем дополнительные метрики лучшей модели
    additional_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in additional_metrics:
        opt_metric = f"{metric}_optimal"
        if opt_metric in best_row and not pd.isna(best_row[opt_metric]):
            best_model[metric] = float(best_row[opt_metric])
        elif metric in best_row and not pd.isna(best_row[metric]):
            best_model[metric] = float(best_row[metric])
    
    print(f"Лучшая модель: {best_model['model_name']}")
    print(f"Метрика сравнения: {best_model['best_metric']} = {best_model['metric_value']:.4f}")

    # Создание директории
    os.makedirs(report_dir, exist_ok=True)
    
    # Сохранение информации о лучшей модели
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(best_model, f)
        best_model_path = f.name
    
    # Сохранение всех метрик для отчета
    df_metrics.to_csv('data/models/comparison/all_models_metrics.csv', index=False)
    
    ti.xcom_push(key='best_model_path', value=best_model_path)
    ti.xcom_push(key='all_metrics_path', value=metrics_path)
    
    return best_model_path

def generate_comparison_report(**kwargs):
    ti = kwargs['ti']
    best_model_path = ti.xcom_pull(key='best_model_path', task_ids='select_best_model')
    all_metrics_path = ti.xcom_pull(key='all_metrics_path', task_ids='select_best_model')
    
    # Загрузка данных
    with open(best_model_path, 'r') as f:
        best_model = json.load(f)
    
    with open(all_metrics_path, 'r') as f:
        all_metrics = json.load(f)
    
    # 1. Создание таблицы сравнения моделей
    df_comparison = pd.DataFrame(all_metrics)
    
    # Выбираем метрики для отображения
    display_metrics = []
    for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']:
        opt_metric = f"{metric}_optimal"
        if opt_metric in df_comparison.columns:
            display_metrics.append(opt_metric)
        elif metric in df_comparison.columns:
            display_metrics.append(metric)
    
    df_display = df_comparison[['model_name'] + display_metrics].copy()
    
    # Форматирование числовых значений
    for col in display_metrics:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # Сохранение таблицы
    table_path = f"{report_dir}/models_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Сравнение моделей машинного обучения\n\n")
        f.write(f"**Дата сравнения**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Лучшая модель**: {best_model['model_name']} ({best_model['best_metric']} = {best_model['metric_value']:.4f})\n\n")
        f.write("## Метрики моделей\n\n")
        f.write(df_display.to_markdown(index=False))
    
    print(f"Таблица сравнения сохранена в {table_path}")
    
    # 2. Создание визуализаций
    if len(all_metrics) > 1:
        # Подготовка данных для графиков
        df_viz = pd.DataFrame(all_metrics)
        
        # График 1: ROC-AUC сравнение
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1.1 ROC-AUC
        roc_metric = 'roc_auc_optimal' if 'roc_auc_optimal' in df_viz.columns else 'roc_auc'
        if roc_metric in df_viz.columns:
            ax = axes[0, 0]
            df_viz_sorted = df_viz.sort_values(roc_metric, ascending=False)
            bars = ax.bar(df_viz_sorted['model_name'], df_viz_sorted[roc_metric])
            
            # Подсветка лучшей модели
            best_idx = df_viz_sorted[df_viz_sorted['model_name'] == best_model['model_name']].index
            if len(best_idx) > 0:
                bars[best_idx[0]].set_color('green')
            
            ax.set_title('ROC-AUC Comparison')
            ax.set_ylabel('ROC-AUC')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 1.2 Accuracy сравнение
        acc_metric = 'accuracy_optimal' if 'accuracy_optimal' in df_viz.columns else 'accuracy'
        if acc_metric in df_viz.columns:
            ax = axes[0, 1]
            df_viz_sorted = df_viz.sort_values(acc_metric, ascending=False)
            bars = ax.bar(df_viz_sorted['model_name'], df_viz_sorted[acc_metric])
            best_idx = df_viz_sorted[df_viz_sorted['model_name'] == best_model['model_name']].index
            if len(best_idx) > 0:
                bars[best_idx[0]].set_color('green')
            ax.set_title('Accuracy Comparison')
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 1.3 F1-Score сравнение
        f1_metric = 'f1_score_optimal' if 'f1_score_optimal' in df_viz.columns else 'f1_score'
        if f1_metric in df_viz.columns:
            ax = axes[1, 0]
            df_viz_sorted = df_viz.sort_values(f1_metric, ascending=False)
            bars = ax.bar(df_viz_sorted['model_name'], df_viz_sorted[f1_metric])
            
            best_idx = df_viz_sorted[df_viz_sorted['model_name'] == best_model['model_name']].index
            if len(best_idx) > 0:
                bars[best_idx[0]].set_color('green')
            
            ax.set_title('F1-Score Comparison')
            ax.set_ylabel('F1-Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 1.4 Radar chart для основных метрик
        ax = axes[1, 1]
        metrics_for_radar = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = []
        
        for metric in metrics_for_radar:
            opt_metric = f"{metric}_optimal"
            if opt_metric in df_viz.columns:
                available_metrics.append(opt_metric)
            elif metric in df_viz.columns:
                available_metrics.append(metric)
        
        if available_metrics and len(df_viz) > 0:
            # Нормализация метрик для radar chart
            df_normalized = df_viz[available_metrics].copy()
            for col in available_metrics:
                max_val = df_normalized[col].max()
                if max_val > 0:
                    df_normalized[col] = df_normalized[col] / max_val
            
            # Создание radar chart для первой модели
            import numpy as np
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            model_idx = 0
            values = df_normalized.iloc[model_idx].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_optimal', '').replace('_', ' ').title() for m in available_metrics])
            ax.set_title(f'Metrics Radar Chart - {df_viz.iloc[model_idx]["model_name"]}')
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = f"{report_dir}/models_comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Визуализации сохранены в {plot_path}")
    
    # 3. Создание сводного отчета в JSON
    summary_report = {
        'selection_timestamp': datetime.now().isoformat(),
        'best_model': best_model,
        'total_models_compared': len(all_metrics),
        'comparison_criteria': 'roc_auc_optimal' if 'roc_auc_optimal' in str(all_metrics) else 'roc_auc',
        'models_compared': [m['model_name'] for m in all_metrics]
    }
    
    summary_path = f"{report_dir}/comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"Сводный отчет сохранен в {summary_path}")
    # Очистка временных файлов
    os.unlink(best_model_path)
    os.unlink(all_metrics_path)
    
    return summary_path

def register_best_model(**kwargs):
    ti = kwargs['ti']
    summary_path = ti.xcom_pull(task_ids='generate_comparison_report')
    
    if not summary_path:
        print("Не найден отчет с лучшей моделью")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    best_model = summary['best_model']
    client = get_mlflow_client()

    try:
        # Поиск существующей зарегистрированной модели
        model_name = "Best_Production_Model"
        model_version = None
        # Проверяем, есть ли уже зарегистрированная модель с таким именем
        try:
            registered_model = client.get_registered_model(model_name)
            print(f"Найдена существующая модель: {model_name}")
            
            # Создаем новую версию
            model_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{best_model['run_id']}/model",
                run_id=best_model['run_id']
            )
            
            # Добавляем описание
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Best model selected on {summary['selection_timestamp']}. "
                          f"Model type: {best_model['model_name']}, "
                          f"ROC-AUC: {best_model['metric_value']:.4f}"
            )
            
            # Переводим версию в Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            print(f"Модель зарегистрирована как {model_name} версия {model_version.version}")
            
        except MlflowException:
            # Если модели нет, создаем новую
            print(f"Создание новой зарегистрированной модели: {model_name}")
            model_version = client.create_registered_model(
                name=model_name,
                tags={
                    "best_model_type": best_model['model_name'],
                    "selection_date": summary['selection_timestamp'][:10]
                }
            )
            
            # Создаем первую версию
            model_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{best_model['run_id']}/model",
                run_id=best_model['run_id']
            )
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
        
        # Логирование информации о регистрации
        registration_info = {
            'registered_model_name': model_name,
            'model_version': model_version.version if model_version else 'unknown',
            'source_run_id': best_model['run_id'],
            'registration_timestamp': datetime.now().isoformat(),
            'model_metrics': best_model
        }
        
        # Сохранение информации о регистрации
        registry_dir = 'data/models/registry'
        os.makedirs(registry_dir, exist_ok=True)
        
        registry_path = f"{registry_dir}/model_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(registry_path, 'w') as f:
            json.dump(registration_info, f, indent=2)
        
        print(f"Информация о регистрации сохранена в {registry_path}")
        print(f"Лучшая модель ({best_model['model_name']}) успешно зарегистрирована в MLflow Model Registry")
        
    except Exception as e:
        print(f"Ошибка при регистрации модели: {e}")
        raise

with DAG(
    'compare_and_select_best_model',
    default_args=default_args,
    description='Сравнение метрик моделей и выбор лучшей',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=['mlops', 'model_selection', 'comparison'],
) as dag:

    collect_metrics_task = PythonOperator(
        task_id='collect_metrics',
        python_callable=collect_models_metrics,
    )

    select_best_task = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        provide_context=True,
    )

    generate_report_task = PythonOperator(
        task_id='generate_comparison_report',
        python_callable=generate_comparison_report,
        provide_context=True,
    )

    register_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        provide_context=True,
    )

    collect_metrics_task >> select_best_task >> generate_report_task >> register_model_task