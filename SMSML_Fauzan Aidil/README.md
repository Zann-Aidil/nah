# Eksperimen_SML_NamaSiswa

Repository Kriteria 1 — Eksperimen pada Dataset Pelatihan
untuk submission **SMSML (Sistem Machine Learning)**.

---

## Dataset

| Atribut | Detail |
|---------|--------|
| Nama | Titanic: ML from Disaster |
| Sumber | Kaggle Competition |
| Link | https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data |
| Train set | 891 baris, 12 kolom |
| Target | `Survived` (0 = Tidak Selamat, 1 = Selamat) |
| Tipe Task | Binary Classification |

### Cara Download Dataset
1. Login ke Kaggle
2. Kunjungi: https://www.kaggle.com/competitions/tfugp-titanic-ml-from-disaster/data
3. Klik **Download All**
4. Ekstrak dan letakkan `train.csv` di root folder repository ini

---

## Struktur Repository

```
Eksperimen_SML_NamaSiswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml        ← GitHub Actions (Advance)
├── preprocessing/
│   ├── Eksperimen_NamaSiswa.ipynb   ← Notebook EDA + Preprocessing (Basic)
│   ├── automate_NamaSiswa.py        ← Script otomatisasi (Skilled)
│   └── titanic_preprocessing.csv   ← Output hasil preprocessing
├── train.csv                        ← Dataset dari Kaggle (download manual)
├── requirements.txt
└── README.md
```

---

## Deskripsi Fitur

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| PassengerId | int | ID unik penumpang (dihapus saat modeling) |
| **Survived** | int | **TARGET** — 0=Tidak Selamat, 1=Selamat |
| Pclass | int | Kelas tiket (1=1st, 2=2nd, 3=3rd) |
| Name | str | Nama penumpang (mengandung title) |
| Sex | str | Jenis kelamin |
| Age | float | Usia penumpang (~19% missing) |
| SibSp | int | Jumlah saudara/pasangan di kapal |
| Parch | int | Jumlah orang tua/anak di kapal |
| Ticket | str | Nomor tiket (dihapus saat modeling) |
| Fare | float | Harga tiket |
| Cabin | str | Nomor kabin (~77% missing) |
| Embarked | str | Port keberangkatan (C/Q/S) |

---

## Tahapan Preprocessing

1. **Load Data** — Baca `train.csv` dari Kaggle
2. **Validasi** — Cek missing values dan duplikat
3. **Feature Engineering** — Buat 6 fitur baru:
   - `Title` — diekstrak dari Name (Mr/Mrs/Miss/Master/Rare)
   - `FamilySize` — SibSp + Parch + 1
   - `IsAlone` — 1 jika tidak ada keluarga di kapal
   - `AgeGroup` — Child/Teen/Adult/Middle/Senior
   - `FareGroup` — Low/Medium/High/Very High
   - `HasCabin` — 1 jika info kabin tersedia
4. **Handle Missing Values** — Age: median per Title, Embarked: modus
5. **Encoding** — LabelEncoder untuk Sex, Embarked, Title, AgeGroup, FareGroup
6. **Drop Columns** — Hapus PassengerId, Name, Ticket, Cabin, SibSp, Parch
7. **Train-Test Split** — 80% train / 20% test (stratified)
8. **Scaling** — StandardScaler
9. **Simpan** — Output ke `titanic_preprocessing.csv`

---

## Cara Menjalankan

### Setup environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# atau
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Jalankan notebook (Basic)
```bash
jupyter notebook preprocessing/Eksperimen_NamaSiswa.ipynb
```

### Jalankan script otomatisasi (Skilled)
```bash
# Default (mencari train.csv di folder saat ini)
python preprocessing/automate_NamaSiswa.py

# Dengan argumen lengkap
python preprocessing/automate_NamaSiswa.py \
  --input train.csv \
  --output preprocessing/titanic_preprocessing.csv \
  --test_size 0.2 \
  --random_state 42
```

---

## GitHub Actions (Advance)

Workflow otomatis berjalan saat:
- Ada push ke branch `main` yang mengubah folder `preprocessing/` atau file `train.csv`
- Trigger manual melalui tab **Actions → Run workflow**

### Setup Secrets yang diperlukan:
Masuk ke `Settings → Secrets and variables → Actions`, tambahkan:
- `KAGGLE_USERNAME` — username akun Kaggle kamu
- `KAGGLE_KEY` — API key dari https://www.kaggle.com/account

### Cara mendapat Kaggle API Key:
1. Login Kaggle → klik foto profil → Account
2. Scroll ke bagian **API** → klik **Create New Token**
3. File `kaggle.json` akan terdownload — salin isinya ke secrets
