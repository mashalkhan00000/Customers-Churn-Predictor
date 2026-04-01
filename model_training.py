import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from data_preprocessing import prepare_data
from visualizations import generate_all_plots, plot_churn_distribution


def train_all_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  Trained: {name}")
    return trained


def evaluate_models(trained_models, X_test, y_test):
    print("\n--- Model Comparison ---")
    print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 47)

    results = {}
    best_model_name = None
    best_auc = 0

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model":    model,
            "accuracy": acc,
            "roc_auc":  auc,
            "y_pred":   y_pred,
            "y_prob":   y_prob
        }
        print(f"{name:<25} {acc:>9.2%} {auc:>10.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    print(f"\nBest model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    return results, best_model_name


def detailed_report(results, best_model_name, y_test):
    best = results[best_model_name]
    print(f"\n--- Detailed Report: {best_model_name} ---")
    print(classification_report(y_test, best['y_pred'], target_names=['Stay', 'Churn']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best['y_pred']))


def save_best_model(results, best_model_name, scaler, encoders, feature_names):
    best_model = results[best_model_name]['model']
    joblib.dump(best_model,    "models/best_model.pkl")
    joblib.dump(scaler,        "models/scaler.pkl")
    joblib.dump(encoders,      "models/encoders.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print(f"\nModel saved to models/best_model.pkl")


def run_training(filepath="data/telco_churn.csv"):
    print("=" * 50)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = prepare_data(filepath)

    print("\nTraining models...")
    trained_models = train_all_models(X_train, y_train)

    results, best_model_name = evaluate_models(trained_models, X_test, y_test)
    detailed_report(results, best_model_name, y_test)
    save_best_model(results, best_model_name, scaler, encoders, feature_names)

    # Generate all charts automatically
    print("\nGenerating charts...")
    plot_churn_distribution(y_test)
    generate_all_plots(results, best_model_name, y_test, feature_names)

    return results, best_model_name


if __name__ == "__main__":
    import os
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    run_training()
