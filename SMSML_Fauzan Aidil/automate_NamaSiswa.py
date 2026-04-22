"""
automate_NamaSiswa.py
=====================
Script otomatisasi preprocessing dataset Titanic — ML from Disaster (Kaggle).
Menjalankan seluruh tahapan preprocessing secara otomatis dan menghasilkan
dataset yang siap dilatih.

Sumber dataset:
  Kaggle Competition — Titanic: ML from Disaster
  https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data

  Cara download:
    1. Login ke Kaggle
    2. Kunjungi link di atas → klik Download All
    3. Ekstrak → letakkan train.csv di folder yang sama

Penggunaan:
  python automate_NamaSiswa.py
  python automate_NamaSiswa.py --input train.csv --output titanic_preprocessing.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Kolom yang dihapus sebelum modeling
COLS_TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

# Kolom target
TARGET = 'Survived'

# Kolom kategorikal yang perlu di-encode
CAT_COLS = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']

# Mapping title: title langka → 'Rare'
COMMON_TITLES = {'Mr', 'Miss', 'Mrs', 'Master'}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset Titanic dari file CSV.

    Dataset harus diunduh dari Kaggle terlebih dahulu:
    https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data

    Args:
        path (str): Path ke file train.csv dari Kaggle.

    Returns:
        pd.DataFrame: Dataframe mentah Titanic.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] File '{path}' tidak ditemukan!\n"
            f"Silakan download dataset dari Kaggle:\n"
            f"  https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data\n"
            f"Kemudian letakkan file train.csv di lokasi: {os.path.abspath(path)}"
        )

    df = pd.read_csv(path)
    print(f"[INFO] Data dimuat dari: {path}")
    print(f"       Shape: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


# ─────────────────────────────────────────────
# 2. VALIDASI DATA
# ─────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memvalidasi integritas dataset Titanic:
    - Cek kolom wajib ada
    - Laporan missing values
    - Hapus duplikat jika ada

    Args:
        df (pd.DataFrame): Dataframe input.

    Returns:
        pd.DataFrame: Dataframe valid.
    """
    required_cols = ['PassengerId', 'Survived', 'Pclass', 'Name',
                     'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Kolom wajib tidak ditemukan: {missing_cols}")

    # Laporan missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("[INFO] Missing values yang ditemukan:")
        for col, cnt in missing[missing > 0].items():
            print(f"       {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    else:
        print("[INFO] Tidak ada missing values.")

    # Hapus duplikat
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"[INFO] Duplikat dihapus: {before - len(df)} baris.")

    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur baru dari kolom yang ada:
    - Title: diekstrak dari Name, title langka → 'Rare'
    - FamilySize: SibSp + Parch + 1
    - IsAlone: 1 jika travelling sendiri
    - AgeGroup: kategorisasi usia (Child/Teen/Adult/Middle/Senior)
    - FareGroup: kategorisasi harga tiket (Low/Medium/High/Very High)
    - HasCabin: 1 jika info kabin tersedia, 0 jika tidak

    Args:
        df (pd.DataFrame): Dataframe input.

    Returns:
        pd.DataFrame: Dataframe dengan fitur tambahan.
    """
    df = df.copy()

    # 1. Title dari Name
    df['Title'] = df['Name'].str.extract(r',\s*([^.]+)\.')
    df['Title'] = df['Title'].apply(
        lambda x: x.strip() if isinstance(x, str) and x.strip() in COMMON_TITLES else 'Rare'
    )

    # 2. FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 4. AgeGroup (akan di-impute dulu sebelum cut)
    # Simpan dulu, isi setelah imputasi Age
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'],
        right=True
    )

    # 5. FareGroup
    try:
        df['FareGroup'] = pd.qcut(
            df['Fare'], q=4,
            labels=['Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
    except ValueError:
        df['FareGroup'] = 'Medium'

    # 6. HasCabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    print(f"[INFO] Feature engineering selesai.")
    print(f"       Fitur baru: Title, FamilySize, IsAlone, AgeGroup, FareGroup, HasCabin")

    return df


# ─────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengisi missing values pada dataset Titanic:
    - Age: median berdasarkan Title (lebih akurat daripada median global)
    - Embarked: modus (Southampton 'S')
    - Fare: median global
    - AgeGroup: diisi ulang setelah Age diimputasi
    - FareGroup: diisi dengan 'Unknown'

    Args:
        df (pd.DataFrame): Dataframe input.

    Returns:
        pd.DataFrame: Dataframe tanpa missing values.
    """
    df = df.copy()

    # Age → median per Title
    age_median_by_title = df.groupby('Title')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median_by_title)
    df['Age'] = df['Age'].fillna(df['Age'].median())  # fallback

    # Recalculate AgeGroup setelah Age diisi
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'],
        right=True
    )
    # Tambah kategori 'Unknown' untuk jaga-jaga
    if hasattr(df['AgeGroup'], 'cat'):
        df['AgeGroup'] = df['AgeGroup'].cat.add_categories('Unknown')
    df['AgeGroup'] = df['AgeGroup'].fillna('Unknown')

    # Embarked → modus
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare → median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # FareGroup → Unknown
    if hasattr(df['FareGroup'], 'cat'):
        if 'Unknown' not in df['FareGroup'].cat.categories:
            df['FareGroup'] = df['FareGroup'].cat.add_categories('Unknown')
    df['FareGroup'] = df['FareGroup'].fillna('Unknown')

    remaining = df.isnull().sum().sum()
    print(f"[INFO] Missing values setelah imputasi: {remaining}")
    return df


# ─────────────────────────────────────────────
# 5. ENCODE KOLOM KATEGORIKAL
# ─────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    Melakukan Label Encoding pada kolom kategorikal.

    Kolom yang di-encode: Sex, Embarked, Title, AgeGroup, FareGroup

    Args:
        df (pd.DataFrame): Dataframe input.

    Returns:
        tuple: (Dataframe ter-encode, dict label_encoders)
    """
    df = df.copy()
    label_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"[INFO] Encode '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df, label_encoders


# ─────────────────────────────────────────────
# 6. HAPUS KOLOM TIDAK PERLU
# ─────────────────────────────────────────────

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak digunakan untuk modeling.

    Kolom yang dihapus: PassengerId, Name, Ticket, Cabin, SibSp, Parch
    (SibSp & Parch sudah direpresentasikan dalam FamilySize & IsAlone)

    Args:
        df (pd.DataFrame): Dataframe input.

    Returns:
        pd.DataFrame: Dataframe tanpa kolom yang tidak perlu.
    """
    cols_exist = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_exist)
    print(f"[INFO] Kolom dihapus: {cols_exist}")
    return df


# ─────────────────────────────────────────────
# 7. SPLIT DATA
# ─────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Membagi dataset menjadi train dan test dengan stratifikasi.

    Args:
        X (pd.DataFrame): Fitur.
        y (pd.Series): Target.
        test_size (float): Proporsi test. Default 0.2.
        random_state (int): Seed. Default 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} baris | Test: {len(X_test)} baris")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 8. SCALING FITUR
# ─────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Menstandarisasi fitur menggunakan StandardScaler.
    Fit hanya pada train, transform pada keduanya.

    Args:
        X_train (pd.DataFrame): Data train.
        X_test (pd.DataFrame): Data test.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    print("[INFO] StandardScaler diterapkan pada train dan test.")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 9. SIMPAN HASIL
# ─────────────────────────────────────────────

def save_result(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                output_path: str) -> str:
    """
    Menyimpan hasil preprocessing ke CSV.

    Args:
        X_train, X_test: Fitur setelah scaling.
        y_train, y_test: Target.
        output_path (str): Path file output.

    Returns:
        str: Path file yang disimpan.
    """
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    train_df['split'] = 'train'

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df['split'] = 'test'

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"[INFO] Hasil disimpan ke: {output_path} | Shape: {combined.shape}")
    return output_path


# ─────────────────────────────────────────────
# 10. PIPELINE UTAMA
# ─────────────────────────────────────────────

def preprocess(input_path: str,
               output_path: str = 'titanic_preprocessing.csv',
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Pipeline preprocessing lengkap untuk dataset Titanic Kaggle:

    load → validate → feature_engineering → handle_missing →
    encode → drop_cols → split → scale → save

    Args:
        input_path (str): Path ke train.csv dari Kaggle.
        output_path (str): Path output CSV preprocessing.
        test_size (float): Proporsi test split.
        random_state (int): Random seed.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*58)
    print("  PIPELINE PREPROCESSING — TITANIC: ML FROM DISASTER")
    print("  Kaggle: tfugp-titanic-ml-from-disaster")
    print("="*58)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Validasi
    df = validate_data(df)

    # Step 3: Feature engineering
    df = feature_engineering(df)

    # Step 4: Handle missing values
    df = handle_missing(df)

    # Step 5: Encode kategorikal
    df, _ = encode_categorical(df)

    # Step 6: Hapus kolom tidak perlu
    df = drop_columns(df)

    # Step 7: Pisahkan fitur dan target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"[INFO] Fitur: {X.shape[1]} kolom | Target: '{TARGET}'")
    print(f"[INFO] Distribusi target — Selamat: {y.sum()} ({y.mean():.1%}), "
          f"Tidak: {len(y)-y.sum()} ({1-y.mean():.1%})")

    # Step 8: Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # Step 9: Scaling
    X_train, X_test, _ = scale_features(X_train, X_test)

    # Step 10: Simpan
    save_result(X_train, X_test, y_train, y_test, output_path)

    print("\n" + "="*58)
    print("  PREPROCESSING SELESAI!")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Train  : {len(X_train)} baris")
    print(f"  Test   : {len(X_test)} baris")
    print(f"  Fitur  : {X_train.shape[1]} kolom")
    print("="*58 + "\n")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing Titanic Kaggle dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cara penggunaan:
  python automate_NamaSiswa.py --input train.csv
  python automate_NamaSiswa.py --input train.csv --output titanic_preprocessing.csv
  python automate_NamaSiswa.py --input train.csv --test_size 0.2 --random_state 42

Download dataset dari:
  https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data
        """
    )
    parser.add_argument(
        '--input', type=str,
        default='train.csv',
        help='Path ke file train.csv dari Kaggle (default: train.csv)'
    )
    parser.add_argument(
        '--output', type=str,
        default='titanic_preprocessing.csv',
        help='Path output file CSV hasil preprocessing (default: titanic_preprocessing.csv)'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help='Proporsi test split, 0.0-1.0 (default: 0.2)'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed untuk reproduktivitas (default: 42)'
    )

    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
