# House Price Prediction — Streamlit App

Aplikasi web interaktif untuk memprediksi harga rumah, dibangun dengan **Streamlit**
sebagai lapisan deployment dari project Machine Learning *House Price Prediction*
(dikembangkan di Google Colab menggunakan dataset **House Prices - Advanced
Regression Techniques** dari Kaggle).

> Aplikasi ini **tidak mengubah** algoritma, preprocessing, atau model ML yang
> sudah dibuat di notebook Google Colab. Aplikasi hanya memuat ulang model yang
> sudah dilatih dan disimpan (`best_model.pkl`, `scaler.pkl`).

---

## Struktur Project

```
house_price_app/
├── app.py                     # Aplikasi utama Streamlit
├── requirements.txt           # Daftar dependency
├── README.md                  # Dokumentasi ini
├── model/
│   ├── best_model.pkl         # Model terbaik hasil training (wajib)
│   ├── scaler.pkl             # StandardScaler hasil training (wajib)
│   ├── feature_columns.pkl    # Daftar kolom hasil one-hot encoding (wajib untuk prediksi)
│   ├── feature_medians.pkl    # Nilai median tiap kolom training (wajib untuk prediksi)
│   └── eval_results.pkl       # Hasil evaluasi model (opsional, untuk menu Model Evaluation & Feature Importance)
└── data/
    └── train.csv              # Dataset training (opsional, untuk Dashboard EDA)
```

---

## File Model yang Dibutuhkan

Karena preprocessing di notebook melakukan **One-Hot Encoding** terhadap ± 80 fitur
mentah (menghasilkan ratusan kolom), form prediksi yang hanya berisi 10 input
sederhana **tidak bisa langsung** dimasukkan ke model. Agar struktur data saat
prediksi identik dengan saat training, dibutuhkan 2 file tambahan selain
`best_model.pkl` dan `scaler.pkl`.

Tambahkan cell berikut **di akhir notebook Google Colab Anda** (kode ML yang sudah
ada TIDAK diubah sama sekali — ini hanya menambahkan langkah export):

```python
import joblib

# 1. Daftar kolom hasil encoding (urutan harus identik saat inference)
feature_columns = X_train_encoded.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

# 2. Nilai median tiap kolom sebagai baseline untuk fitur yang tidak
#    muncul di form prediksi (misalnya fitur hasil one-hot encoding)
feature_medians = X_train_encoded.median()
joblib.dump(feature_medians, 'feature_medians.pkl')

# 3. Hasil evaluasi model untuk ditampilkan di menu Model Evaluation
eval_results = {
    'model_name': model_name,
    'rmse': best_rmse,
    'mae': best_mae,
    'r2': best_r2,
    'y_val_actual': y_val_final,
    'y_val_pred': y_pred_final,
    'comparison_table': results_comparison,
    'feature_importance': feature_importance if model_name in
        ['XGBoost', 'Random Forest'] else None,
}
joblib.dump(eval_results, 'eval_results.pkl')

print(" File tambahan untuk deployment berhasil disimpan:")
print("   - feature_columns.pkl")
print("   - feature_medians.pkl")
print("   - eval_results.pkl")
```

Setelah itu, download semua file `.pkl` dari Colab dan letakkan di folder `model/`.
File `best_model.pkl` dan `scaler.pkl` yang sudah Anda simpan sebelumnya juga
diletakkan di folder yang sama.

(Opsional) Letakkan `train.csv` di folder `data/` agar menu **Dashboard EDA**
menampilkan visualisasi dari dataset asli. Jika tidak tersedia, aplikasi akan
menawarkan fitur upload file CSV langsung dari browser.

> **Catatan tentang `price_per_sqft`**: fitur ini dihitung dari `SalePrice`
> (variabel target) pada notebook asli, sehingga tidak dapat dihitung ulang saat
> prediksi (harga rumah belum diketahui pada saat itu). Aplikasi menggunakan
> nilai **median** dari data training sebagai baseline untuk fitur ini.

---

## Cara Install

1. **Clone / download** folder project ini.

2. **(Opsional tapi disarankan)** buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. Pastikan file model (`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`,
   `feature_medians.pkl`, dan opsional `eval_results.pkl`) sudah ada di folder
   `model/`.

---

## Cara Menjalankan Streamlit

Dari terminal, jalankan:

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat:

```
http://localhost:8501
```

---

## Deploy ke Streamlit Community Cloud

1. Push seluruh folder project ini (termasuk folder `model/` berisi file `.pkl`)
   ke repository GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **New app**, pilih repository, branch, dan file utama `app.py`.
4. Klik **Deploy** — aplikasi akan otomatis menginstall dependency dari
   `requirements.txt` dan langsung berjalan tanpa perlu mengubah kode apapun.

> Pastikan ukuran file `.pkl` (terutama jika menggunakan Random Forest / XGBoost
> dengan banyak trees) tidak melebihi batas repository/deploy Anda. Jika terlalu
> besar, pertimbangkan menyimpan model di Git LFS atau storage eksternal.

---

## Penjelasan Project

Project ini memprediksi harga jual rumah (`SalePrice`) menggunakan dataset
**House Prices - Advanced Regression Techniques** (Kaggle), melalui tahapan:

1. **Data Acquisition** — mengunduh dataset dari Kaggle.
2. **Exploratory Data Analysis (EDA)** — statistik deskriptif, distribusi target,
   analisis missing value, outlier, dan korelasi antar fitur.
3. **Data Preparation** — drop fitur missing tinggi, imputasi, *feature
   engineering* (`total_sqft`, `age`, `remod_age`, `price_per_sqft`), One-Hot
   Encoding, dan `StandardScaler`.
4. **Modeling** — melatih dan membandingkan **Linear Regression**,
   **Random Forest Regressor**, dan **XGBoost Regressor** (hyperparameter tuning
   dengan `GridSearchCV`).
5. **Evaluation** — RMSE, MAE, R², analisis residual, dan *Absolute Percentage
   Error* (APE).
6. **Deployment** — model dengan performa validasi terbaik disimpan sebagai
   `best_model.pkl` dan digunakan pada aplikasi Streamlit ini.

Aplikasi Streamlit menyediakan 5 menu:

| Menu | Isi |
|---|---|
| **Home** | Judul, deskripsi, tujuan project, model terbaik, dataset |
| **Dashboard EDA** | Distribusi SalePrice, missing values, correlation heatmap, feature importance, statistik dataset |
| **House Price Prediction** | Form input interaktif + prediksi harga menggunakan `best_model.pkl` |
| **Model Evaluation** | RMSE, MAE, R², grafik Actual vs Predicted, Residual Plot, Model Comparison |
| **Documentation** | Penjelasan dataset, metodologi, pipeline ML, perbandingan model, cara pakai |

---

## Kredit

Universitas Dian Nuswantoro — Mata Kuliah Pembelajaran Mesin
Dataset: [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
