# MLOps Project - Пайплайн обучения моделей

Этот проект демонстрирует полный MLOps пайплайн для обучения и сравнения нескольких моделей машинного обучения с использованием Airflow, MLflow и Docker. Для обучения модели используется заранее обработанный датасет.

## Архитектура

- **Оркестрация**: Apache Airflow
- **Трекинг экспериментов**: MLflow
- **Контейнеризация**: Docker + Docker Compose
- **Хранилище**: Yandex Cloud S3 (для артефактов моделей)
- **Модели**: 5 различных алгоритмов с оптимизацией порогов

## Структура проекта
```
.
├── dags/
│ ├── train_lr_dag.py # Logistic Regression
│ ├── train_rf_dag.py # Random Forest
│ ├── train_gb_dag.py # Gradient Boosting
│ ├── train_lda_dag.py # Linear Discriminant Analysis
│ └── train_catboost_dag.py # CatBoost
├── src/
│ ├── preprocess.py # Предобработка данных
│ └── metrics_utils.py # Утилиты для оценки моделей
├── data/
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные
├── docker-compose.yml # Docker компоновка
├── Dockerfile # Образ для сервисов
├── requirements.txt # Зависимости Python
└── README.md
```

## Модели

- **Logistic Regression** - с балансировкой весов классов
- **Random Forest** - с оптимальным порогом классификации
- **Gradient Boosting (XGBoost)** - с SMOTE и весами классов
- **Linear Discriminant Analysis** - с заданными priors
- **CatBoost** - с автоматической балансировкой классов

## Предварительные требования

- Docker >= 20.10
- Docker Compose >= 2.20
- Python 3.10

## Использование

1. Клонируйте репозиторий:
```bash
git clone https://github.com/PASTER-G/MLOps-learning.git
cd mlops-project
```
2. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл с вашими S3 credentials
```
3. Запустите сервисы:
```bash
docker-compose up -d
```
4. Получите доступ к сервисам:
- **Airflow UI**: http://localhost:8080
    - **Логин**: `airflow`
    - **Пароль**: `airflow`
- **MLflow UI**: http://localhost:5050
- **Jupyter Lab**: http://localhost:8888
5. Запустите обучение моделей:
- В Airflow UI найдите DAGs с префиксом train_
- Запустите вручную нужные DAGs для обучения моделей
- Отслеживайте эксперименты в MLflow

## Мониторинг и логирование

- Все метрики моделей автоматически логируются в MLflow
- Матрицы ошибок и ROC-кривые сохраняются как артефакты
- Логи выполнения доступны в Airflow UI и в папке logs/

## Что можно улучшить

- Добавление DAG для анализа метрик всех обученных моделей и автоматического выбора лучшей
- Реализация CI/CD пайплайна для автоматического развертывания
- Интеграция с Weights & Biases для расширенного трекинга экспериментов
- Создание API для обслуживания моделей с помощью FastAPI
- Добавление тестов для данных и моделей

## Автор

[PASTER-G](https://github.com/PASTER-G)