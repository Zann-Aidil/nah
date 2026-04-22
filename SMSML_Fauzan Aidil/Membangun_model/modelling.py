"""
modelling.py
============
Kriteria 2 — Basic: Membangun Model Machine Learning
Dataset  : Titanic — ML from Disaster (Kaggle)
Nama     : Fauzan
NIM      : 23

Menjalankan pelatihan model Random Forest menggunakan MLflow autolog.
Semua artefak disimpan ke MLflow Tracking UI (localhost:5000).

Cara menjalankan:
  1. Pastikan MLflow UI sudah berjalan:
       mlflow ui --host 127.0.0.1 --port 5000
  2. Jalankan script ini:
       python modelling.py
  3. Buka browser: http://127.0.0.1:5000
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score
)

# ─────────────────────────────────────────────
# KONFIGURASI MLFLOW
# ─────────────────────────────────────────────

TRACKING_URI  = "http://127.0.0.1:5000/"
EXPERIMENT    = "Titanic Fauzan"
DATA_PATH     = "titanic_preprocessing.csv"
TARGET        = "Survived"
RANDOM_STATE  = 42

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────
# 1. LOAD DATA PREPROCESSING
# ─────────────────────────────────────────────

print("=" * 55)
print("  KRITERIA 2 — BASIC: MODELLING TITANIC")
print("  Nama : Fauzan | NIM : 23")
print("=" * 55)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"File '{DATA_PATH}' tidak ditemukan!\n"
        f"Pastikan file hasil preprocessing dari Kriteria 1 "
        f"sudah ada di folder yang sama dengan modelling.py"
    )

df = pd.read_csv(DATA_PATH)
print(f"\n[INFO] Data dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")

# Pisahkan train dan test berdasarkan kolom 'split'
train_df = df[df['split'] == 'train'].drop(columns=['split'])
test_df  = df[df['split'] == 'test'].drop(columns=['split'])

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"[INFO] Train : {X_train.shape[0]} baris")
print(f"[INFO] Test  : {X_test.shape[0]} baris")
print(f"[INFO] Fitur : {X_train.shape[1]} kolom")
print(f"[INFO] Target distribusi train — "
      f"Selamat: {y_train.sum()} ({y_train.mean():.1%}), "
      f"Tidak: {(y_train==0).sum()}")

# ─────────────────────────────────────────────
# 2. TRAINING DENGAN MLFLOW AUTOLOG
# ─────────────────────────────────────────────

print(f"\n[INFO] Memulai MLflow run...")
print(f"[INFO] Tracking URI : {TRACKING_URI}")
print(f"[INFO] Experiment   : {EXPERIMENT}")

# Aktifkan autolog — mencatat semua parameter & metrik otomatis
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic_Fauzan"):

    # Inisialisasi model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Training
    print("\n[INFO] Melatih model RandomForestClassifier...")
    model.fit(X_train, y_train)

    # Prediksi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Hitung metrik
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    # Log metrik tambahan (autolog sudah log sebagian besar)
    mlflow.log_metric("test_accuracy",  acc)
    mlflow.log_metric("test_f1_score",  f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall",    recall)
    mlflow.log_metric("test_roc_auc",   roc_auc)

    # Log info tambahan
    mlflow.log_param("nama",    "Fauzan")
    mlflow.log_param("nim",     "23")
    mlflow.log_param("dataset", "Titanic Kaggle")
    mlflow.log_param("level",   "Basic")

    # Dapatkan run ID
    run_id = mlflow.active_run().info.run_id

    print("\n" + "=" * 55)
    print("  HASIL EVALUASI MODEL")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  ROC AUC   : {roc_auc:.4f}")
    print("=" * 55)
    print(f"\n[INFO] Run ID : {run_id}")
    print(f"[INFO] Cek hasil di: {TRACKING_URI}")
    print("\n[INFO] SCREENSHOT YANG HARUS DIAMBIL:")
    print("  1. screenshoot_dashboard.jpg")
    print("     → Buka http://127.0.0.1:5000")
    print("     → Klik experiment 'Titanic Fauzan'")
    print("     → Screenshot tampilan daftar run")
    print("  2. screenshoot_artifak.jpg")
    print("     → Klik run 'RandomForest_Basic_Fauzan'")
    print("     → Screenshot tab Artifacts")
    print("=" * 55)
