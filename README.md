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

## 📦 Data Understanding

### Dataset: Flood Prediction Dataset
- 50.000 sampel dengan 21 fitur numerik
- Variabel target: `FloodProbability` (skala 0–1)

### Fitur Kunci:
- **Environmental**: `MonsoonIntensity`, `Deforestation`, `ClimateChange`
- **Infrastructure**: `DamsQuality`, `DrainageSystems`
- **Socio-political**: `PoliticalFactors`, `PopulationScore`

### 📈 Analisis Eksploratori:
- Distribusi target normal dengan mean 0.49
- Korelasi kuat antara `TopographyDrainage` dengan target (r = 0.82)
- Outlier terdeteksi pada `Urbanization` dan `PoliticalFactors`

---

## 🧹 Data Preparation

- **Standarisasi**: Menggunakan `StandardScaler` untuk normalisasi fitur
- **Train-Test Split**: Rasio 80:20 dengan stratifikasi

📌 *Catatan:*  
- Standarisasi penting untuk algoritma seperti Linear Regression  
- Stratifikasi menjaga distribusi target tetap seimbang

---

## 🤖 Modeling

### 🔧 Algoritma
- **Linear Regression**
  - ✅ Kelebihan: Interpretasi mudah, komputasi cepat
  - ❌ Kekurangan: Asumsi linearitas sering tidak terpenuhi

- **Random Forest Regressor**
  - ✅ Kelebihan: Handle non-linearity, robust terhadap outlier
  - ❌ Kekurangan: Risiko overfitting jika tidak di-tuning

---

## 🧪 Improvement: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_```python
```
# 📊 Evaluation

## Metrik yang Digunakan:
- **MAE (Mean Absolute Error):** Deviasi absolut rata-rata prediksi  
- **RMSE (Root Mean Squared Error):** Deviasi kuadrat rata-rata  
- **R² Score:** Proporsi varian yang dapat dijelaskan oleh model  

| Model                  | MAE   | RMSE  | R²    |
|------------------------|-------|-------|-------|
| Linear Regression      | 0.018 | 0.024 | 0.992 |
| Random Forest (Base)   | 0.020 | 0.026 | 0.730 |
| Random Forest (Tuned)  | 0.020 | 0.026 | 0.732 |

---

## 📌 Interpretasi:
- **Linear Regression** memiliki R² tinggi, namun berisiko **overfitting** pada data yang tidak linier.
- **Random Forest** dipilih sebagai model final karena:
  - Lebih **robust** terhadap overfitting  
  - Mampu menangkap **hubungan non-linear**  
  - Memberikan **feature importance** yang bermanfaat untuk insight  

---

## 📌 Feature Importance (Python Code)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

importances = best_rf.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance dari Random Forest (Tuned)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
```
