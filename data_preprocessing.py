import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Load dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_info(df):
    """Quick overview of the dataset."""
    print("\n--- Dataset Overview ---")
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)


def clean_data(df):
    """
    Handle missing values, remove duplicates,
    and fix any obvious data issues.
    """
    df = df.drop_duplicates()

    # Drop customerID if it exists — not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # TotalCharges sometimes comes as string — fix that
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill numeric nulls with median (safer than mean for skewed data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical nulls with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"\nAfter cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_features(df, target_col='Churn'):
    """
    Encode categorical columns using LabelEncoder.
    Returns encoded dataframe, encoder dict, and feature/target split.
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, encoders


def scale_features(X_train, X_test):
    """Standardize features so no single column dominates the model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(filepath, target_col='Churn', test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline — load, clean, encode, scale, split.
    Returns everything needed for model training.
    """
    df = load_data(filepath)
    basic_info(df)

    df = clean_data(df)
    X, y, encoders = encode_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print(f"\nTraining samples: {X_train_scaled.shape[0]}")
    print(f"Testing samples:  {X_test_scaled.shape[0]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X.columns.tolist()


if __name__ == "__main__":
    prepare_data("data/telco_churn.csv")
