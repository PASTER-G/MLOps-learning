# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str, output_dir: str):
    """Загрузка, очистка и подготовка датасета."""
    df = pd.read_csv(input_path)

    # Очистка и нормализация названий столбцов
    df.columns = (
        df.columns.str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('/', '_')
        .str.replace('([a-z0-9])([A-Z])', r'\1_\2', regex=True)
        .str.lower()
    )

    # Приведение текстовых значений к нижнему регистру
    def to_lower(val):
        if isinstance(val, str):
            return val.strip().lower()
        return val

    for col in df.select_dtypes(include=['object', 'category']):
        df[col] = df[col].apply(to_lower)

    # Бинарные флаги
    binary_columns = ['has_mortgage', 'has_dependents', 'has_co_signer']
    for col in binary_columns:
        df[col] = df[col].map({'yes': True, 'no': False})

    # Удаление дубликатов и ненужных столбцов
    df.drop_duplicates(inplace=True)
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    # One-hot кодирование категорий
    df = pd.get_dummies(
        df,
        columns=['education', 'employment_type', 'marital_status', 'loan_purpose'],
        drop_first=True
    )

    # Разделение на признаки и целевую переменную
    X = df.drop(columns=['default'])
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем обработанные данные
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("✅ Preprocessing completed and saved to:", output_dir)
