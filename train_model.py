import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier


# ============================================================
# 1. CREATE TRAINING DATA
# ============================================================

np.random.seed(42)

N_NORMAL = 3000
N_INSIDER = 300

normal = pd.DataFrame({
    "login_frequency": np.random.poisson(5, N_NORMAL),
    "file_access_frequency": np.random.poisson(18, N_NORMAL),
    "file_modification_frequency": np.random.poisson(7, N_NORMAL),
    "usb_usage": np.random.binomial(1, 0.08, N_NORMAL),
    "external_email_ratio": np.random.beta(2, 15, N_NORMAL),
    "after_hours_ratio": np.random.beta(2, 15, N_NORMAL),
    "unique_files_accessed": np.random.poisson(12, N_NORMAL),
    "web_activity": np.random.poisson(25, N_NORMAL),
})

normal["label"] = 0


insider = pd.DataFrame({
    "login_frequency": np.random.poisson(9, N_INSIDER),
    "file_access_frequency": np.random.poisson(65, N_INSIDER),
    "file_modification_frequency": np.random.poisson(30, N_INSIDER),
    "usb_usage": np.random.binomial(1, 0.80, N_INSIDER),
    "external_email_ratio": np.random.beta(8, 4, N_INSIDER),
    "after_hours_ratio": np.random.beta(8, 4, N_INSIDER),
    "unique_files_accessed": np.random.poisson(45, N_INSIDER),
    "web_activity": np.random.poisson(60, N_INSIDER),
})

insider["label"] = 1


data = pd.concat([normal, insider], ignore_index=True)

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "login_frequency",
    "file_access_frequency",
    "file_modification_frequency",
    "usb_usage",
    "external_email_ratio",
    "after_hours_ratio",
    "unique_files_accessed",
    "web_activity"
]

X = data[FEATURES]
y = data["label"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. SUPERVISED MODEL — XGBOOST
# ============================================================

print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)


# ============================================================
# 5. ANOMALY MODEL — ISOLATION FOREST
# ============================================================

print("Training Isolation Forest...")

normal_training_data = X_train[y_train == 0]

isolation_model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

isolation_model.fit(normal_training_data)


# ============================================================
# 6. ANOMALY MODEL — ONE CLASS SVM
# ============================================================

print("Training One-Class SVM...")

svm_model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.08
)

svm_model.fit(normal_training_data)


# ============================================================
# 7. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

precision = precision_score(
    y_test,
    xgb_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    xgb_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    xgb_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    xgb_predictions,
    target_names=["Normal", "Insider"],
    zero_division=0
))


# ============================================================
# 8. SAVE MODELS
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    xgb_model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    isolation_model,
    "models/isolation_forest.pkl"
)

joblib.dump(
    svm_model,
    "models/one_class_svm.pkl"
)

joblib.dump(
    FEATURES,
    "models/features.pkl"
)

joblib.dump(
    {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    },
    "models/metrics.pkl"
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nSaved models:")
print("✓ models/xgboost_model.pkl")
print("✓ models/isolation_forest.pkl")
print("✓ models/one_class_svm.pkl")