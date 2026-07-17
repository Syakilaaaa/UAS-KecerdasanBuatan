# 🛒 Prediksi Intensi Pembelian Pelanggan E-commerce

**UAS Kecerdasan Buatan - Prediksi Intensi Pembelian Pelanggan E-commerce Menggunakan Algoritma Decision Tree dan KNN**

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi apakah seorang pengunjung situs e-commerce akan melakukan pembelian atau tidak berdasarkan data perilaku mereka. Model yang dikembangkan menggunakan dua algoritma klasifikasi:

1. **Decision Tree** - Algoritma berbasis pohon keputusan yang interpretable
2. **K-Nearest Neighbors (KNN)** - Algoritma berbasis jarak yang sederhana dan efektif

Dataset yang digunakan adalah **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository (Sakar et al., 2018) yang terdiri dari 12.330 sesi dengan 18 fitur.

---

## 👥 Anggota Kelompok

| Nama | NIM |
|:---|:---|
| [Syakila Nurul Ijati] | [2406060] |

---

## 📊 Dataset

### Sumber Data
- **Nama Dataset**: Online Shoppers Purchasing Intention Dataset
- **Sumber**: UCI Machine Learning Repository
- **Link**: [UCI Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Penulis**: Sakar, C.O., Polat, S.O., Katircioglu, M., & Kastro, Y. (2018)
- **Jumlah Sampel**: 12.330 sesi
- **Jumlah Fitur**: 18 kolom (10 numerik, 8 kategorikal)
- **Target**: `Revenue` (True = Pembelian, False = Tidak Pembelian)

### Distribusi Target

| Kelas | Jumlah | Persentase |
|:---|:---|:---|
| Tidak Membeli (False) | 10.422 | 84.5% |
| Membeli (True) | 1.908 | 15.5% |

**Status**: Data Tidak Seimbang (Imbalanced) dengan rasio 5.5:1

---

## 🔄 Langkah Pengerjaan Proyek

### 1. Data Understanding
- Membaca dataset menggunakan `pd.read_csv()`
- Melihat struktur data dengan `df.head()`, `df.info()`, `df.describe()`
- Memeriksa missing values dengan `df.isnull().sum()`
- Menganalisis distribusi target dengan `df['Revenue'].value_counts()`

**Hasil**: Dataset terdiri dari 12.330 baris dan 18 kolom, tidak ada missing values, target tidak seimbang (84.5% tidak membeli, 15.5% membeli).

### 2. Exploratory Data Analysis (EDA)
- **Visualisasi Distribusi Data**: Pie chart, bar chart, histogram, bar chart kategorikal
- **Analisis Korelasi**: Heatmap korelasi antar fitur
- **Pairplot**: Hubungan antar fitur dengan 500 sampel
- **Deteksi Data Tidak Seimbang**: Rasio 5.5:1

**Insight**: 
- Fitur paling prediktif: PageValues, ProductRelated_Duration, BounceRates
- Pengunjung yang membeli cenderung memiliki durasi produk tinggi, PageValues tinggi, dan BounceRate rendah
- Korelasi tinggi: BounceRates & ExitRates (0.91), ProductRelated & ProductRelated_Duration (0.86)

### 3. Data Preparation
- **Pembersihan Data**: Tidak ada missing values, tidak ada duplikasi
- **Encoding Data Kategorik**: 
  - Label Encoding untuk VisitorType (New=0, Returning=1, Other=2)
  - One-Hot Encoding untuk Month (menjadi 12 kolom biner)
- **Normalisasi/Standardisasi**: StandardScaler (mean=0, std=1) untuk KNN
- **Split Data**: 80% Train (9.864), 20% Test (2.466) dengan stratify

### 4. Modeling
- **Decision Tree**: max_depth=5, min_samples_split=10, min_samples_leaf=5
- **KNN**: n_neighbors=5, metric='euclidean', weights='distance'

### 5. Evaluation
- **Confusion Matrix** untuk kedua model
- **Metrik Evaluasi**: Accuracy, Precision, Recall, F1-Score
- **ROC Curve** dan AUC

---

## 📊 Hasil

### Perbandingan Model

| Metrik | Decision Tree | KNN |
|:---|:---|:---|
| **Accuracy** | **89.9%** | 87.1% |
| **Precision** | **70.7%** | 64.7% |
| **Recall** | **70.7%** | 64.7% |
| **F1-Score** | **70.7%** | 64.7% |
| **ROC-AUC** | **0.917** | 0.776 |

### Confusion Matrix Decision Tree

| | Pred: Tidak Beli | Pred: Beli |
|:---|:---|:---|
| **Actual: Tidak Beli** | 1.991 | 93 |
| **Actual: Beli** | 156 | 226 |

- **True Negative**: 1.991 (benar tidak membeli)
- **False Positive**: 93 (salah prediksi membeli) → Biaya marketing terbuang
- **False Negative**: 156 (salah prediksi tidak membeli) → Pelanggan terlewat
- **True Positive**: 226 (benar membeli)

### 🏆 Model Terbaik: Decision Tree

**Alasan:**
1. F1-Score tertinggi (70.7% vs 64.7%)
2. Recall lebih baik (70.7% vs 64.7%) → Menemukan +87 pelanggan potensial
3. ROC-AUC lebih baik (0.917 vs 0.776) → Kemampuan diskriminasi sangat baik
4. Interpretabilitas tinggi → Tim bisnis dapat memahami model
5. Keseimbangan Precision dan Recall (70.7% : 70.7%)

---

## 📁 Struktur Repository

UAS-KecerdasanBuatan/

├── README.md # File ini (penjelasan proyek)

├── Laporan uas.md # Laporan lengkap UAS

├── uas_model.ipynb # Notebook kode lengkap

├── data/

└── online_shoppers_intention.csv # Dataset

└── Jurnal/ # Folder referensi jurnal



---
## Cara menjalankan

### 1.Buka Google Colab 
### 2. Upload Dataset 
```python
from google.colab import files
uploaded = files.upload()
# pilih file online_shoppers_intention.csv
```
----
### 3. Jalankan NoteBook
1. Buka uas_model.ipynb
2. Jalankan semua cell secara berurutan (Runtime → Run all)
### 4 Hasil yang Diharapkan 
- Visualisasi EDA (grafik, heatmap, pairplot)
- Model Decision Tree dan KNN yang sudah dilatih
- Confusion Matrix dan metrik evaluasi
- ROC Curve dan visualisasi Decision Tree

---
### Kesimpulan
✅ Tujuan tercapai - Model Decision Tree berhasil memprediksi intensi pembelian dengan F1-Score 70.7%

---
### Penulis
- **Nama:** Syakila Nurul Ijati
- **Mata Kuliah:** Kecerdasan Buatan
- **Program Stusi:** Teknik Informatika

---
 2026 - UAS Kecerdasan Buatan
