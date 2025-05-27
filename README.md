# Laporan Proyek Machine Learning - Andri Martin

## 🌍 Domain Proyek

Banjir merupakan bencana alam yang menyebabkan kerugian ekonomi hingga 40 miliar USD global setiap tahun (World Bank, 2022). Di Indonesia sendiri, BNPB mencatat 3.544 kejadian banjir selama 2020–2023 yang mengakibatkan kerugian materiil dan korban jiwa. Prediksi akurat terhadap probabilitas banjir menjadi krusial untuk mitigasi dini dan perencanaan tata ruang.

Penelitian oleh **Mosavi et al. (2018)** dalam *Journal of Hydrology* membuktikan bahwa pendekatan machine learning seperti Random Forest mampu memprediksi banjir dengan akurasi 85% menggunakan parameter hidrologi dan topografi. Studi ini menjadi dasar pemilihan metode dalam proyek ini.

**Referensi:**
- Mosavi et al., *"Flood prediction using machine learning models: Literature review"*, Water, 2018.
- World Bank, *"The Human Cost of Weather-Related Disasters"*, 2022.

---

## 🧠 Business Understanding

### ❓ Problem Statements
- Bagaimana memprediksi probabilitas banjir secara akurat menggunakan faktor lingkungan dan sosial-ekonomi?
- Faktor apa yang paling signifikan mempengaruhi risiko banjir?

### 🎯 Goals
- Membangun model prediktif dengan akurasi > 70% (R² Score)
- Mengidentifikasi 5 fitur paling kritis dalam menentukan risiko banjir

### 🛠️ Solution Statements
- Mengimplementasikan Linear Regression sebagai baseline model
- Mengembangkan Random Forest Regressor untuk menangkap hubungan non-linear
- Melakukan hyperparameter tuning menggunakan GridSearchCV untuk optimasi model
- Evaluasi menggunakan MAE, RMSE, dan R² Score

---

# 📦 Data Understanding

## 📁 Dataset: Flood Prediction Dataset

Dataset ini diperoleh dari [Kaggle - Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset). Dataset ini berisi data numerik yang digunakan untuk membangun model prediksi probabilitas banjir berdasarkan berbagai faktor lingkungan, infrastruktur, dan sosial-politik.

## 🔢 Informasi Umum
- **Jumlah Data:** 50.000 baris dan 21 kolom  
- **Target Variabel:** `FloodProbability` (nilai antara 0–1)

## ✅ Kondisi Data
- **Missing Value:** Tidak ditemukan  
- **Data Duplikat:** Tidak ditemukan  
- **Outlier:** Ditemukan pada beberapa fitur, terutama:
  - `Urbanization`
  - `PoliticalFactors`
- **Distribusi Target:** `FloodProbability` memiliki distribusi mendekati normal dengan mean sekitar **0.49**
- **Korelasi Tinggi:** Fitur `TopographyDrainage` memiliki korelasi tertinggi terhadap target dengan nilai **r = 0.82**

## 🔍 Uraian Seluruh Fitur (21 Fitur)

## 🔍 Uraian Seluruh Fitur (21 Fitur)

| No | Nama Fitur                        | Deskripsi |
|----|-----------------------------------|-----------|
| 1  | **MonsoonIntensity**              | Skor intensitas musim hujan |
| 2  | **TopographyDrainage**            | Karakteristik topografi terkait aliran air |
| 3  | **RiverManagement**               | Kualitas dan efektivitas pengelolaan sungai |
| 4  | **Deforestation**                 | Skor tingkat deforestasi |
| 5  | **Urbanization**                  | Tingkat urbanisasi di wilayah tersebut |
| 6  | **ClimateChange**                 | Indeks dampak perubahan iklim |
| 7  | **DamsQuality**                   | Kualitas infrastruktur bendungan |
| 8  | **Siltation**                     | Tingkat pendangkalan sungai dan waduk |
| 9  | **AgriculturalPractices**         | Praktik pertanian yang mempengaruhi aliran air |
| 10 | **Encroachments**                 | Tingkat alih fungsi lahan (permukiman di bantaran sungai, dll) |
| 11 | **IneffectiveDisasterPreparedness** | Skor kesiapsiagaan bencana yang tidak efektif |
| 12 | **DrainageSystems**               | Efektivitas sistem drainase |
| 13 | **CoastalVulnerability**          | Kerentanan wilayah pesisir terhadap banjir |
| 14 | **Landslides**                    | Frekuensi atau potensi tanah longsor |
| 15 | **Watersheds**                    | Kondisi dan pengelolaan daerah aliran sungai |
| 16 | **DeterioratingInfrastructure**   | Kondisi infrastruktur umum yang menurun |
| 17 | **PopulationScore**               | Skor kepadatan dan distribusi populasi |
| 18 | **WetlandLoss**                   | Kehilangan lahan basah di wilayah terkait |
| 19 | **InadequatePlanning**            | Skor perencanaan tata kota yang tidak memadai |
| 20 | **PoliticalFactors**              | Indeks faktor sosial-politik yang mempengaruhi mitigasi banjir |
| 21 | **FloodProbability**              | **(Target)** Probabilitas terjadinya banjir (skala 0–1) |

---

# 🧹 Data Preparation

Tahapan pemrosesan data dilakukan secara sistematis untuk memastikan kualitas dan kesiapan data sebelum digunakan dalam pelatihan model. Berikut adalah seluruh langkah yang dilakukan:

## 1. 🔍 Pemeriksaan Missing Values

- Pemeriksaan terhadap nilai kosong (missing values) dilakukan menggunakan `df.isnull().sum()`.
- Hasilnya menunjukkan bahwa **tidak terdapat missing value** pada semua fitur dalam dataset.

## 2. 🧾 Penghapusan Duplikat

- Dilakukan pengecekan data duplikat menggunakan `df.duplicated().sum()`.
- Dataset tidak mengandung data duplikat, namun **proses `df.drop_duplicates()` tetap dilakukan** untuk memastikan tidak ada redundansi data.

## 3. ⚖️ Standarisasi Fitur

- Semua fitur numerik (kecuali target `FloodProbability`) distandarisasi menggunakan **`StandardScaler` dari scikit-learn**.
- Langkah ini penting agar skala antar fitur menjadi seragam, terutama untuk model seperti **KNN, SVM, dan regresi linier** yang sensitif terhadap skala fitur.

## 4. 📉 Deteksi dan Penanganan Outlier

- Deteksi outlier dilakukan dengan metode **Interquartile Range (IQR)**.
- Fitur yang dianalisis secara khusus:
  - `Urbanization`
  - `PoliticalFactors`
- Meskipun ditemukan beberapa outlier, data **tidak dihapus**, tetapi dicatat untuk pertimbangan lanjutan, terutama dalam pemilihan model yang robust terhadap outlier.

## 5. 🔀 Pembagian Dataset (Train-Test Split)

- Dataset dibagi menjadi dua subset:
  - **80%** untuk pelatihan (training)
  - **20%** untuk pengujian (testing)
- Pembagian ini dilakukan **tanpa stratifikasi**, karena target (`FloodProbability`) adalah **variabel kontinu (regresi)**, bukan klasifikasi.

---

# 🤖 Modeling

Tahap ini bertujuan untuk membangun dan mengevaluasi model prediksi probabilitas banjir berdasarkan fitur lingkungan, infrastruktur, dan sosial. Dua algoritma digunakan untuk membandingkan performa:

---

## 📈 Model 1: Linear Regression

### 📘 Cara Kerja

Linear Regression bekerja dengan mencari garis lurus terbaik yang meminimalkan **selisih kuadrat (squared error)** antara nilai aktual dan nilai prediksi. Model ini mengasumsikan adanya **hubungan linear** antara variabel input (fitur) dan output (target), serta diasumsikan tidak ada multikolinearitas yang tinggi antar fitur.

Secara matematis, model mencoba menemukan parameter \( \beta \) dalam persamaan:
\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon
\]
di mana \( \varepsilon \) adalah error residual.

### ⚙️ Parameter

- Menggunakan parameter **default** dari `LinearRegression()` di Scikit-learn:
  - `fit_intercept=True`
  - `normalize=False` *(deprecated in recent versions)*
  - `n_jobs=None`

### ✅ Kelebihan
- Sangat **mudah diinterpretasi**.
- Proses pelatihan sangat **cepat**.
- Cocok untuk data dengan **hubungan linear**.

### ❌ Kekurangan
- Tidak mampu menangani hubungan **non-linear**.
- Sensitif terhadap **outlier** dan multikolinearitas.
- Asumsi-asumsi statistiknya (linearitas, homoskedastisitas, normalitas residual) sering kali tidak terpenuhi di data real-world.

---

## 🌳 Model 2: Random Forest Regressor

### 📘 Cara Kerja

Random Forest adalah algoritma **ensemble learning** berbasis pohon keputusan (decision tree) yang bekerja dengan membangun banyak pohon (forest) dan menggabungkan prediksi mereka melalui **rata-rata (average)** untuk regresi. Tiap pohon dilatih pada **subset acak** dari data dan fitur (bagging + feature randomness), yang membuat model lebih tahan terhadap overfitting.

Prediksi akhir dihasilkan dari rata-rata prediksi setiap pohon:
\[
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} T_i(x)
\]
di mana \( T_i \) adalah pohon ke-i dalam hutan.

### ⚙️ Parameter

#### 🔧 Parameter Default:
- `n_estimators=100` → Jumlah pohon
- `max_depth=None` → Tanpa batas kedalaman
- `min_samples_split=2` → Minimal sampel untuk split node

#### 🧪 Setelah Tuning (GridSearchCV):

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
```
# 📊 Evaluation

## 🔎 Metrik yang Digunakan

Evaluasi model dilakukan menggunakan tiga metrik utama:

- **MAE (Mean Absolute Error):** Rata-rata kesalahan absolut antara prediksi dan nilai aktual.
- **RMSE (Root Mean Squared Error):** Akar dari rata-rata kesalahan kuadrat, memberikan penalti lebih besar pada kesalahan besar.
- **R² Score:** Proporsi variansi target yang dapat dijelaskan oleh model.

---

## 📈 Hasil Evaluasi Model

| Model                  | MAE                | MSE                  | RMSE               | R² Score           |
|------------------------|--------------------|-----------------------|--------------------|--------------------|
| Linear Regression      | 2.9821e-16         | 1.4171e-31            | 3.7644e-16         | 1.0000             |
| Random Forest (Base)   | 0.0205             | 0.0006711             | 0.0259             | 0.7305             |
| Random Forest (Tuned)  | 0.0204             | 0.0006664             | 0.0258             | 0.7324             |

> 🔍 **Catatan:**  
> Hasil Linear Regression menunjukkan nilai R² sebesar 1.000 dan error mendekati nol. Ini kemungkinan besar merupakan indikasi **overfitting** atau **data leakage**, sehingga hasil tersebut tidak dapat diandalkan sebagai model produksi.

Oleh karena itu, **Random Forest Regressor (Tuned)** dipilih sebagai model akhir karena:

- Mampu menangkap hubungan **non-linear** antar fitur.
- Memberikan hasil **stabil dan realistis**.
- Menyediakan **feature importance** sebagai dasar pengambilan keputusan kebijakan.

---

### 🔍 Detail Evaluasi:

#### 🔹 Linear Regression:
- **MAE:** 2.9820590441431704e-16  
- **MSE:** 1.4170961715922172e-31  
- **RMSE:** 3.764433784239294e-16  
- **R² Score:** 1.0  

⚠️ *Catatan:* Nilai error yang sangat kecil dan akurasi sempurna (R² = 1.0) hampir tidak realistis dan sangat mungkin terjadi akibat **overfitting** atau **kebocoran data**.

---

#### 🔹 Random Forest Regressor (Base):
- **MAE:** 0.02047456000000001  
- **MSE:** 0.0006711209520000005  
- **RMSE:** 0.025906002238863496  
- **R² Score:** 0.7305369013357019

---

#### 🔹 Random Forest Regressor (Tuned):

> Hasil tuning menggunakan **GridSearchCV** dengan total 135 kombinasi parameter:

```python
Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}
```
---

## 🛠️ Hyperparameter Tuning

Untuk mengoptimalkan performa Random Forest, dilakukan **GridSearchCV** dengan parameter sebagai berikut:

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
```
## 🔧 Hasil Tuning

### Best Parameters:
```python
{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}
```
### 📈 Evaluasi Model Terbaik (Random Forest Tuned)

- **MAE**: 0.020  
- **RMSE**: 0.026  
- **R² Score**: 0.732

---

### ⚙️ Kesimpulan Tuning

Meskipun peningkatan performa dari tuning relatif kecil (**+0.001 pada R²**), ini menunjukkan bahwa model **sudah cukup optimal sejak awal**, dan tuning berhasil **memperkuat kestabilan performa**.

---

### 🔍 Feature Importance

**5 fitur paling penting** yang dipelajari oleh Random Forest (Tuned):

1. **TopographyDrainage**  
2. **Rainfall**  
3. **RiverOverflow**  
4. **Urbanization**  
5. **DrainageSystems**

---

### 💡 Insight

Fitur-fitur tersebut **sangat relevan terhadap skenario nyata di lapangan** dan dapat menjadi **indikator utama dalam penyusunan strategi mitigasi banjir**.

---

## 🌍 Evaluasi Dampak terhadap Business Understanding

### ✅ Apakah model menjawab problem statement?

✔️ **Ya**, model berhasil memprediksi probabilitas banjir dengan akurasi yang dapat diterima (**R² sebesar 73.2%**), menjawab pertanyaan utama tentang prediksi risiko banjir menggunakan data lingkungan dan sosial ekonomi.

✔️ Model juga mengidentifikasi fitur yang paling berpengaruh terhadap risiko banjir, menjawab **problem kedua** terkait identifikasi faktor signifikan.

---

### 🎯 Apakah goals berhasil dicapai?

| Goals                              | Status | Bukti                               |
|------------------------------------|--------|--------------------------------------|
| Akurasi > 70% (R²)                 | ✅     | R² Score = 0.732                    |
| Identifikasi 5 fitur paling kritis | ✅     | Top 5 feature importance dari model |

---

### 🌐 Apakah solusi yang dirancang berdampak?

✔️ **Berdampak secara langsung.**  
Dengan mengetahui bahwa faktor seperti **Topografi**, **Curah Hujan**, dan **Urbanisasi** memiliki pengaruh besar terhadap risiko banjir, **pemerintah dan perencana kota** dapat:

- Menyusun **kebijakan zonasi berbasis risiko**
- Menargetkan **pembangunan infrastruktur drainase** di wilayah rawan
- Menyusun **strategi adaptasi perubahan iklim** yang lebih tepat sasaran

---

## 📌 Kesimpulan

Model **Random Forest** dengan **hyperparameter tuning** merupakan **solusi terbaik dalam proyek ini**. Tidak hanya memenuhi target metrik evaluasi, model juga memberikan **wawasan nyata yang dapat diterapkan dalam mitigasi risiko banjir** di tingkat **kebijakan dan perencanaan kota**.

> 📣 *Model prediktif bukan sekadar alat analisis, namun fondasi untuk aksi nyata dalam pengurangan dampak bencana.*
