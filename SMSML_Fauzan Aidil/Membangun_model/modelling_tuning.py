"""
modelling_tuning.py
===================
Kriteria 2 — Skilled: Hyperparameter Tuning + Manual Logging
Dataset  : Titanic — ML from Disaster (Kaggle)
Nama     : Fauzan
NIM      : 23

Perbedaan dari Basic (modelling.py):
  - Menggunakan GridSearchCV untuk hyperparameter tuning
  - Menggunakan MANUAL LOGGING (bukan autolog)
  - Menyimpan artefak tambahan: confusion matrix, feature importance
  - Struktur artefak model lengkap (MLmodel, conda.yaml, dll)

Cara menjalankan:
  1. Pastikan MLflow UI sudah berjalan:
       mlflow ui --host 127.0.0.1 --port 5000
  2. Jalankan script ini:
       python modelling_tuning.py
  3. Buka browser: http://127.0.0.1:5000
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, ConfusionMatrixDisplay,
    classification_report
)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

TRACKING_URI = "http://127.0.0.1:5000/"
EXPERIMENT   = "Titanic Fauzan"
DATA_PATH    = "titanic_preprocessing.csv"
TARGET       = "Survived"
RANDOM_STATE = 42
CV_FOLDS     = 5

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("=" * 58)
print("  KRITERIA 2 — SKILLED: TUNING + MANUAL LOGGING")
print("  Nama : Fauzan | NIM : 23")
print("=" * 58)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"File '{DATA_PATH}' tidak ditemukan!\n"
        f"Pastikan file hasil preprocessing ada di folder ini."
    )

df = pd.read_csv(DATA_PATH)
train_df = df[df['split'] == 'train'].drop(columns=['split'])
test_df  = df[df['split'] == 'test'].drop(columns=['split'])

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"\n[INFO] Train: {X_train.shape} | Test: {X_test.shape}")


# ─────────────────────────────────────────────
# 2. FUNGSI HELPER
# ─────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, filename="training_confusion_matrix.png"):
    """Buat dan simpan plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Tidak Selamat", "Selamat"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix — Titanic\nFauzan (NIM: 23)", fontsize=12)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix disimpan: {filename}")
    return filename


def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    """Buat dan simpan plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(11, len(feature_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(top_n),
           importances[indices[:top_n]],
           color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)),
           edgecolor='white')
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices[:top_n]],
                       rotation=30, ha='right', fontsize=10)
    ax.set_title("Feature Importance — RandomForest\nFauzan (NIM: 23)", fontsize=12)
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance disimpan: {filename}")
    return filename


def plot_roc_curve_manual(model, X_test, y_test, filename="roc_curve.png"):
    """Buat dan simpan plot ROC curve."""
    from sklearn.metrics import roc_curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Titanic\nFauzan (NIM: 23)', fontsize=12)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curve disimpan: {filename}")
    return filename


# ─────────────────────────────────────────────
# 3. HYPERPARAMETER TUNING — RANDOM FOREST
# ─────────────────────────────────────────────

print("\n[INFO] Melakukan GridSearchCV — RandomForestClassifier...")

param_grid_rf = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [4, 6, 8, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf' : [1, 2, 4],
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid_rf,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
gs_rf.fit(X_train, y_train)

best_rf    = gs_rf.best_estimator_
best_params = gs_rf.best_params_
best_cv_f1  = gs_rf.best_score_

print(f"\n[INFO] Best params  : {best_params}")
print(f"[INFO] Best CV F1   : {best_cv_f1:.4f}")


# ─────────────────────────────────────────────
# 4. EVALUASI MODEL TERBAIK
# ─────────────────────────────────────────────

y_pred      = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy"  : accuracy_score(y_test, y_pred),
    "f1_score"  : f1_score(y_test, y_pred),
    "precision" : precision_score(y_test, y_pred),
    "recall"    : recall_score(y_test, y_pred),
    "roc_auc"   : roc_auc_score(y_test, y_pred_prob),
    "cv_best_f1": best_cv_f1,
}

print("\n" + "=" * 58)
print("  HASIL EVALUASI MODEL TERBAIK")
print("=" * 58)
for k, v in metrics.items():
    print(f"  {k:<15}: {v:.4f}")
print("=" * 58)

print("\n" + classification_report(
    y_test, y_pred,
    target_names=["Tidak Selamat", "Selamat"]
))


# ─────────────────────────────────────────────
# 5. BUAT ARTEFAK (PLOT)
# ─────────────────────────────────────────────

cm_file   = plot_confusion_matrix(best_rf, X_test, y_test)
fi_file   = plot_feature_importance(best_rf, X_train.columns.tolist())
roc_file  = plot_roc_curve_manual(best_rf, X_test, y_test)


# ─────────────────────────────────────────────
# 6. LOG KE MLFLOW — MANUAL LOGGING
# ─────────────────────────────────────────────

print("\n[INFO] Logging ke MLflow (manual)...")

with mlflow.start_run(run_name="RandomForest_Skilled_Fauzan"):

    # ── Log parameter ──
    mlflow.log_param("nama",            "Fauzan")
    mlflow.log_param("nim",             "23")
    mlflow.log_param("dataset",         "Titanic Kaggle")
    mlflow.log_param("level",           "Skilled")
    mlflow.log_param("model_type",      "RandomForestClassifier")
    mlflow.log_param("cv_folds",        CV_FOLDS)
    mlflow.log_param("random_state",    RANDOM_STATE)

    # Log best hyperparameters
    for param_name, param_val in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_val)

    # ── Log metrik (sama seperti autolog) ──
    mlflow.log_metric("accuracy",   metrics["accuracy"])
    mlflow.log_metric("f1_score",   metrics["f1_score"])
    mlflow.log_metric("precision",  metrics["precision"])
    mlflow.log_metric("recall",     metrics["recall"])
    mlflow.log_metric("roc_auc",    metrics["roc_auc"])
    mlflow.log_metric("cv_best_f1", metrics["cv_best_f1"])

    # ── Log artefak ──
    mlflow.log_artifact(cm_file,  "plots")
    mlflow.log_artifact(fi_file,  "plots")
    mlflow.log_artifact(roc_file, "plots")

    # ── Log model (menghasilkan MLmodel, conda.yaml, dll) ──
    mlflow.sklearn.log_model(
        sk_model       = best_rf,
        artifact_path  = "model",
        registered_model_name = "TitanicModel_Fauzan"
    )

    run_id = mlflow.active_run().info.run_id

print("\n" + "=" * 58)
print("  MLFLOW LOGGING SELESAI!")
print("=" * 58)
print(f"  Run ID      : {run_id}")
print(f"  Tracking URI: {TRACKING_URI}")
print(f"  Experiment  : {EXPERIMENT}")
print("=" * 58)
print("\n[INFO] SCREENSHOT YANG HARUS DIAMBIL:")
print("  1. screenshoot_dashboard.jpg")
print("     → Buka http://127.0.0.1:5000")
print("     → Klik experiment 'Titanic Fauzan'")
print("     → Screenshot daftar semua run")
print("  2. screenshoot_artifak.jpg")
print("     → Klik run 'RandomForest_Skilled_Fauzan'")
print("     → Klik tab 'Artifacts'")
print("     → Screenshot artefak: model/, plots/")
print("=" * 58)
