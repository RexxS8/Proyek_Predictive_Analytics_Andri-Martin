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

| No | Nama Fitur                 | Deskripsi |
|----|----------------------------|-----------|
| 1  | **Rainfall**               | Intensitas curah hujan |
| 2  | **MonsoonIntensity**       | Skor intensitas musim hujan |
| 3  | **RiverOverflow**          | Volume luapan sungai |
| 4  | **DamsQuality**            | Kualitas infrastruktur bendungan |
| 5  | **DrainageSystems**        | Efektivitas sistem drainase |
| 6  | **SoilPermeability**       | Permeabilitas tanah terhadap air |
| 7  | **TopographyDrainage**     | Karakteristik topografi terkait aliran air |
| 8  | **Urbanization**           | Tingkat urbanisasi di wilayah tersebut |
| 9  | **Deforestation**          | Skor tingkat deforestasi |
| 10 | **ClimateChange**          | Indeks dampak perubahan iklim |
| 11 | **SeaLevelRise**           | Kenaikan permukaan laut |
| 12 | **GreenCover**             | Persentase tutupan vegetasi hijau |
| 13 | **WetlandPresence**        | Keberadaan lahan basah di wilayah terkait |
| 14 | **RiverProximity**         | Jarak ke sungai terdekat |
| 15 | **InfrastructureDevelopment** | Skor pembangunan infrastruktur umum |
| 16 | **PoliticalFactors**       | Indeks faktor sosial-politik yang mempengaruhi mitigasi banjir |
| 17 | **DisasterPreparedness**   | Tingkat kesiapan menghadapi bencana |
| 18 | **PopulationScore**        | Skor kepadatan dan distribusi populasi |
| 19 | **PublicAwareness**        | Skor kesadaran publik terhadap risiko banjir |
| 20 | **EmergencyServices**      | Kualitas dan kesiapan layanan darurat |
| 21 | **FloodProbability**       | **(Target)** Probabilitas terjadinya banjir (skala 0–1) |

---

# 🧹 Data Preparation

Tahapan pemrosesan data dilakukan secara sistematis untuk memastikan kualitas dan kesiapan data sebelum digunakan dalam pelatihan model. Berikut adalah seluruh langkah yang dilakukan:

## 1. 🔍 Pemeriksaan Awal Data

- **Drop Duplicate:**  
  Telah dilakukan pengecekan terhadap data duplikat, dan **tidak ditemukan duplikat** pada dataset.

- **Cek Missing Values:**  
  Pemeriksaan terhadap nilai kosong (missing values) dilakukan, dan **tidak terdapat missing value** pada semua fitur.

## 2. 📉 Deteksi dan Penanganan Outlier

- Deteksi outlier dilakukan menggunakan metode **Interquartile Range (IQR)** pada beberapa fitur yang memiliki nilai ekstrem:
  - `Urbanization`
  - `PoliticalFactors`

- Penanganan dapat berupa **penghapusan** atau **transformasi** nilai-nilai outlier, tergantung strategi modeling yang digunakan. Namun, dalam proses ini, outlier **tidak dihapus secara langsung**, melainkan dicatat untuk pertimbangan lanjutan dalam pemilihan algoritma yang lebih tahan terhadap outlier.

## 3. ⚖️ Standarisasi Fitur

- Dilakukan standarisasi terhadap seluruh fitur numerik (selain target) menggunakan **`StandardScaler` dari scikit-learn**.
- Tujuan standarisasi adalah untuk menyamakan skala antar fitur, terutama karena algoritma seperti **Linear Regression, SVM, dan KNN** sangat sensitif terhadap perbedaan skala.

## 4. 🔀 Pembagian Dataset (Train-Test Split)

- Dataset dibagi menjadi dua bagian:
  - **80%** data untuk pelatihan (training)
  - **20%** data untuk pengujian (testing)

- **Stratifikasi tidak digunakan**, karena target (`FloodProbability`) bersifat **kontinu (regresi)**, bukan kategori. Stratifikasi umumnya hanya digunakan pada kasus klasifikasi untuk menjaga distribusi label target tetap proporsional di setiap subset.

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

| Model                  | MAE     | RMSE    | R²      |
|------------------------|---------|---------|---------|
| Linear Regression      | ~0.000  | ~0.000  | 1.000   |
| Random Forest (Base)   | 0.020   | 0.026   | 0.731   |
| Random Forest (Tuned)  | 0.020   | 0.026   | 0.732   |

> 🔍 **Catatan:**  
> Hasil Linear Regression menunjukkan nilai R² sebesar 1.000 dan error mendekati nol. Ini kemungkinan besar merupakan indikasi **overfitting** atau **data leakage**, sehingga hasil tersebut tidak dapat diandalkan sebagai model produksi.

Oleh karena itu, **Random Forest Regressor** dipilih sebagai model akhir karena:

- Mampu menangkap hubungan **non-linear** antar fitur.
- Memberikan hasil **stabil dan realistis**.
- Menyediakan **feature importance** sebagai dasar pengambilan keputusan kebijakan.

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
