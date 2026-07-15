"""
==================================================================
 HOUSE PRICE PREDICTION - STREAMLIT WEB APP
==================================================================
 Aplikasi ini adalah lapisan deployment (UI) untuk project
 Machine Learning "House Price Prediction" yang sudah dibuat di
 Google Colab.

 PENTING:
 - File ini TIDAK mengubah algoritma ML, preprocessing, ataupun
   proses training yang ada di notebook Google Colab.
 - Aplikasi ini hanya MEMUAT ULANG (load) model yang sudah dilatih
   dan disimpan sebagai file .pkl (best_model.pkl, scaler.pkl, dst).
 - Semua prediksi dihasilkan oleh model yang sama persis dengan
   yang ada di notebook.
==================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CUSTOM CSS (modern look)
# ==================================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .price-box {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .price-box h1 {
        font-size: 2.8rem;
        margin: 0;
        color: white;
    }
    .price-box p {
        opacity: 0.9;
        margin: 0;
        font-size: 1rem;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        padding-top: 2rem;
        padding-bottom: 1rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
# PATH CONFIG
# ==================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_CANDIDATES = [os.path.join(BASE_DIR, "model"), BASE_DIR]
DATA_DIR_CANDIDATES = [os.path.join(BASE_DIR, "data"), BASE_DIR]


def find_file(filename, candidates):
    """Cari file di beberapa kemungkinan folder (model/ atau root)."""
    for folder in candidates:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None


# ==================================================================
# CACHED LOADERS
# ==================================================================
@st.cache_resource
def load_pickle_asset(filename, candidates=MODEL_DIR_CANDIDATES):
    path = find_file(filename, candidates)
    if path is None:
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Gagal memuat {filename}: {e}")
        return None


@st.cache_data
def load_dataset():
    path = find_file("train.csv", DATA_DIR_CANDIDATES)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None


best_model = load_pickle_asset("best_model.pkl")
scaler = load_pickle_asset("scaler.pkl")
feature_columns = load_pickle_asset("feature_columns.pkl")
feature_medians = load_pickle_asset("feature_medians.pkl")
eval_results = load_pickle_asset("eval_results.pkl")
df_train = load_dataset()

MODEL_READY = (best_model is not None) and (scaler is not None)
FULL_PIPELINE_READY = MODEL_READY and (feature_columns is not None) and (feature_medians is not None)

MODEL_NAME = eval_results["model_name"] if eval_results else "Best Model (tersimpan di best_model.pkl)"

CURRENT_YEAR_IN_TRAINING = 2024  # sesuai perhitungan 'age' & 'remod_age' di notebook

# ==================================================================
# HELPER FUNCTIONS
# ==================================================================
def format_usd(value):
    return f"${value:,.0f}"


def build_feature_row(user_inputs: dict):
    """
    Membangun 1 baris fitur lengkap (sesuai struktur X_train_encoded
    hasil one-hot encoding di notebook), lalu men-scale-nya dengan
    scaler.pkl yang sama persis dipakai saat training.

    Fitur yang TIDAK diisi lewat form (fitur kategorikal hasil one-hot
    encoding & fitur numerik lain) diisi dengan nilai MEDIAN dari data
    training (feature_medians.pkl), supaya struktur kolom tetap identik
    dengan saat model dilatih.
    """
    # baseline = median tiap kolom hasil encoding pada data training
    row = feature_medians.copy()

    # override kolom yang memang diinput user di form
    for col, val in user_inputs.items():
        if col in row.index:
            row[col] = val

    # feature engineering ULANG untuk kolom turunan, PERSIS seperti di notebook
    # (total_sqft, age, remod_age dihitung ulang dari input user)
    if "total_sqft" in row.index and "GrLivArea" in user_inputs and "TotalBsmtSF" in user_inputs:
        row["total_sqft"] = user_inputs["GrLivArea"] + user_inputs["TotalBsmtSF"]

    if "age" in row.index and "YearBuilt" in user_inputs:
        row["age"] = CURRENT_YEAR_IN_TRAINING - user_inputs["YearBuilt"]

    if "remod_age" in row.index and "YearRemodAdd" in user_inputs:
        row["remod_age"] = CURRENT_YEAR_IN_TRAINING - user_inputs["YearRemodAdd"]

    # NOTE: 'price_per_sqft' pada notebook dihitung dari SalePrice (target),
    # sehingga nilainya tidak bisa dihitung ulang saat inference (harga belum
    # diketahui). Kolom ini tetap memakai nilai MEDIAN dari data training.

    # pastikan urutan kolom identik dengan saat scaler & model dilatih
    row = row.reindex(feature_columns).fillna(0)

    X = row.values.reshape(1, -1)
    return X


def predict_price(user_inputs: dict):
    X_raw = build_feature_row(user_inputs)
    X_scaled = scaler.transform(X_raw)
    prediction = best_model.predict(X_scaled)[0]
    return prediction


# ==================================================================
# SIDEBAR NAVIGATION
# ==================================================================
with st.sidebar:
    st.markdown("## 🏠 House Price ML")
    st.caption("Prediksi Harga Rumah dengan Machine Learning")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        [
            "🏡 Home",
            "📊 Dashboard EDA",
            "🔮 House Price Prediction",
            "📈 Model Evaluation",
            "📚 Documentation",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if MODEL_READY:
        st.success("Model siap digunakan ✅")
    else:
        st.error("Model belum ditemukan ❌")
        st.caption("Pastikan best_model.pkl & scaler.pkl ada di folder `model/`.")
    st.caption(f"Model terbaik: **{MODEL_NAME}**")

# ==================================================================
# MENU 1: HOME
# ==================================================================
if menu == "🏡 Home":
    st.markdown('<p class="main-title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Prediksi Harga Rumah Menggunakan Machine Learning — '
        "Ames Housing / House Prices: Advanced Regression Techniques (Kaggle)</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Deskripsi Project")
        st.write(
            """
            Project ini bertujuan membangun model *machine learning* untuk
            memprediksi harga jual rumah (**SalePrice**) berdasarkan
            karakteristik properti seperti luas bangunan, kualitas material,
            jumlah kamar, tahun dibangun, dan berbagai fitur lainnya.

            Model dikembangkan menggunakan tahapan standar *data science
            pipeline*: mulai dari eksplorasi data (EDA), penanganan
            *missing values*, *feature engineering*, *encoding*,
            *scaling*, hingga perbandingan beberapa algoritma regresi.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Tujuan Project")
        st.write(
            """
            - Membangun model prediksi harga rumah dengan akurasi tinggi.
            - Membandingkan performa **Linear Regression**, **Random Forest**,
              dan **XGBoost** untuk menentukan model terbaik.
            - Target performa: RMSE < \\$25.000, MAE < \\$20.000, R² > 0.85.
            - Menyediakan aplikasi interaktif agar model dapat digunakan
              secara langsung oleh pengguna non-teknis.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Model Terbaik")
        st.metric("Model", MODEL_NAME)
        if eval_results:
            st.metric("Validation R²", f"{eval_results['r2']:.4f}")
            st.metric("Validation RMSE", format_usd(eval_results["rmse"]))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📂 Dataset")
        st.write(
            """
            **House Prices - Advanced Regression Techniques** (Kaggle)

            - ± 1.460 baris data training
            - ± 80 fitur properti (ukuran, kualitas, lokasi, tahun, dll.)
            - Target: `SalePrice` (harga jual rumah dalam USD)
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "Gunakan menu di sidebar untuk menjelajahi analisis data (EDA), "
        "melakukan prediksi harga rumah, melihat evaluasi model, atau "
        "membaca dokumentasi lengkap project."
    )

# ==================================================================
# MENU 2: DASHBOARD EDA
# ==================================================================
elif menu == "📊 Dashboard EDA":
    st.markdown('<p class="main-title">📊 Dashboard EDA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Eksplorasi data dari dataset House Prices</p>',
        unsafe_allow_html=True,
    )

    if df_train is None:
        st.warning(
            "Dataset `train.csv` tidak ditemukan di folder `data/`. "
            "Silakan unggah file dataset untuk menampilkan visualisasi EDA."
        )
        uploaded = st.file_uploader("Upload train.csv", type=["csv"])
        if uploaded is not None:
            df_train = pd.read_csv(uploaded)

    if df_train is not None:
        # ---- Statistik dataset ----
        st.subheader("📋 Statistik Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Baris", f"{df_train.shape[0]:,}")
        c2.metric("Total Kolom", f"{df_train.shape[1]:,}")
        if "SalePrice" in df_train.columns:
            c3.metric("Rata-rata Harga", format_usd(df_train["SalePrice"].mean()))
            c4.metric("Median Harga", format_usd(df_train["SalePrice"].median()))

        st.dataframe(df_train.describe(), use_container_width=True)

        st.markdown("---")

        # ---- Distribusi SalePrice ----
        if "SalePrice" in df_train.columns:
            st.subheader("💰 Distribusi SalePrice")
            colA, colB = st.columns(2)
            with colA:
                fig = px.histogram(
                    df_train, x="SalePrice", nbins=50,
                    title="Distribusi SalePrice (Original)",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig.update_layout(bargap=0.05)
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                fig2 = px.histogram(
                    df_train, x=np.log1p(df_train["SalePrice"]), nbins=50,
                    title="Distribusi Log(SalePrice)",
                    color_discrete_sequence=["#22c55e"],
                )
                fig2.update_layout(bargap=0.05, xaxis_title="Log(SalePrice)")
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # ---- Missing values ----
        st.subheader("🧩 Missing Values")
        missing = df_train.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            missing_pct = (missing / len(df_train) * 100).round(2)
            missing_df = pd.DataFrame(
                {"Feature": missing.index, "Missing Count": missing.values, "Missing %": missing_pct.values}
            )
            fig3 = px.bar(
                missing_df.head(20), x="Missing %", y="Feature", orientation="h",
                title="Top 20 Fitur dengan Missing Values",
                color="Missing %", color_continuous_scale="Reds",
            )
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.success("Tidak ada missing values pada dataset ini.")

        st.markdown("---")

        # ---- Correlation heatmap ----
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = df_train.select_dtypes(include=[np.number])
        if "SalePrice" in numeric_df.columns:
            top_corr = numeric_df.corr()["SalePrice"].sort_values(ascending=False)
            top_features = top_corr[1:11].index.tolist()
            corr_matrix = numeric_df[["SalePrice"] + top_features].corr()
            fig4 = px.imshow(
                corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                aspect="auto", title="Correlation Heatmap - Top 10 Fitur vs SalePrice",
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Kolom SalePrice tidak ditemukan pada dataset yang diunggah.")

        st.markdown("---")

        # ---- Feature importance (jika tersedia) ----
        st.subheader("⭐ Feature Importance")
        if eval_results and eval_results.get("feature_importance") is not None:
            fi = eval_results["feature_importance"].head(20)
            fig5 = px.bar(
                fi.sort_values("Importance"), x="Importance", y="Feature",
                orientation="h", title=f"Top 20 Feature Importance ({MODEL_NAME})",
                color="Importance", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info(
                "Feature importance belum tersedia. Jalankan sel export "
                "tambahan di notebook Colab (menyimpan `eval_results.pkl`) "
                "untuk menampilkan grafik ini."
            )
    else:
        st.stop()

# ==================================================================
# MENU 3: HOUSE PRICE PREDICTION
# ==================================================================
elif menu == "🔮 House Price Prediction":
    st.markdown('<p class="main-title">🔮 House Price Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Masukkan karakteristik rumah untuk memprediksi estimasi harga jual</p>',
        unsafe_allow_html=True,
    )

    if not FULL_PIPELINE_READY:
        st.error(
            "Model belum bisa digunakan untuk prediksi. Pastikan file berikut "
            "tersedia di folder `model/`: `best_model.pkl`, `scaler.pkl`, "
            "`feature_columns.pkl`, `feature_medians.pkl`. "
            "Lihat menu **Documentation** untuk cara membuat file-file ini."
        )
    else:
        with st.form("prediction_form"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
                gr_liv_area = st.number_input("Ground Living Area (sq ft)", 300, 6000, 1500, step=10)
                garage_cars = st.slider("Garage Cars (kapasitas mobil)", 0, 5, 2)
                total_bsmt_sf = st.number_input("Total Basement (sq ft)", 0, 6000, 800, step=10)

            with col2:
                full_bath = st.slider("Full Bathrooms", 0, 4, 2)
                year_built = st.number_input("Year Built", 1870, 2026, 2000, step=1)
                year_remod = st.number_input("Year Remodel/Add", 1870, 2026, 2005, step=1)
                bedroom_abvgr = st.slider("Bedrooms (Above Ground)", 0, 8, 3)

            with col3:
                tot_rms_abvgrd = st.slider("Total Rooms (Above Ground)", 2, 14, 7)
                lot_area = st.number_input("Lot Area (sq ft)", 1000, 220000, 9500, step=100)

            st.markdown("</div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

        if submitted:
            user_inputs = {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageCars": garage_cars,
                "TotalBsmtSF": total_bsmt_sf,
                "FullBath": full_bath,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "BedroomAbvGr": bedroom_abvgr,
                "TotRmsAbvGrd": tot_rms_abvgrd,
                "LotArea": lot_area,
            }
            try:
                prediction = predict_price(user_inputs)
                prediction = max(prediction, 0)
                st.markdown(
                    f"""
                    <div class="price-box">
                        <p>Estimated House Price</p>
                        <h1>{format_usd(prediction)}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "⚠️ Fitur lain di luar 10 input di atas menggunakan nilai "
                    "median dari data training (lihat menu Documentation)."
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# ==================================================================
# MENU 4: MODEL EVALUATION
# ==================================================================
elif menu == "📈 Model Evaluation":
    st.markdown('<p class="main-title">📈 Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Hasil evaluasi model pada data validasi (dari notebook)</p>',
        unsafe_allow_html=True,
    )

    if eval_results is None:
        st.warning(
            "File `eval_results.pkl` belum ditemukan. Jalankan sel export "
            "tambahan di notebook Google Colab untuk menyimpan hasil evaluasi "
            "(RMSE, MAE, R², prediksi validasi, tabel perbandingan model). "
            "Lihat menu **Documentation** untuk kode lengkapnya."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", format_usd(eval_results["rmse"]))
        c2.metric("MAE", format_usd(eval_results["mae"]))
        c3.metric("R² Score", f"{eval_results['r2']:.4f}")

        st.markdown("---")

        y_actual = eval_results.get("y_val_actual")
        y_pred = eval_results.get("y_val_pred")

        if y_actual is not None and y_pred is not None:
            colA, colB = st.columns(2)
            with colA:
                st.subheader("🎯 Actual vs Predicted")
                fig = px.scatter(
                    x=y_actual, y=y_pred, opacity=0.5,
                    labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
                )
                min_v, max_v = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
                fig.add_trace(
                    go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                name="Ideal", line=dict(color="red", dash="dash"))
                )
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                st.subheader("📉 Residual Plot")
                residuals = np.array(y_actual) - np.array(y_pred)
                fig2 = px.scatter(
                    x=y_pred, y=residuals, opacity=0.5,
                    labels={"x": "Predicted Price ($)", "y": "Residuals ($)"},
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        comparison = eval_results.get("comparison_table")
        if comparison is not None:
            st.subheader("🏁 Model Comparison")
            fig3 = make_subplots(rows=1, cols=3, subplot_titles=("Val RMSE", "Val MAE", "Val R²"))
            fig3.add_trace(go.Bar(x=comparison["Model"], y=comparison["Val RMSE"], marker_color="#3b82f6"), row=1, col=1)
            fig3.add_trace(go.Bar(x=comparison["Model"], y=comparison["Val MAE"], marker_color="#f59e0b"), row=1, col=2)
            fig3.add_trace(go.Bar(x=comparison["Model"], y=comparison["Val R²"], marker_color="#22c55e"), row=1, col=3)
            fig3.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(comparison, use_container_width=True)

# ==================================================================
# MENU 5: DOCUMENTATION
# ==================================================================
elif menu == "📚 Documentation":
    st.markdown('<p class="main-title">📚 Documentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Dokumentasi lengkap project & aplikasi</p>', unsafe_allow_html=True)

    with st.expander("📂 Penjelasan Dataset", expanded=True):
        st.write(
            """
            Dataset yang digunakan adalah **House Prices - Advanced Regression
            Techniques** dari Kaggle, berisi data penjualan rumah di Ames, Iowa,
            dengan ± 80 fitur yang menjelaskan karakteristik properti (luas
            bangunan, kualitas material, jumlah kamar, garasi, tahun dibangun,
            lokasi, dll.), dengan target prediksi **SalePrice** (harga jual
            rumah dalam USD).
            """
        )

    with st.expander("🔬 Metodologi"):
        st.write(
            """
            1. **Data Acquisition** — dataset diunduh dari Kaggle (train.csv & test.csv).
            2. **Exploratory Data Analysis (EDA)** — analisis statistik deskriptif,
               distribusi target, missing values, outlier, dan korelasi antar fitur.
            3. **Data Preparation**:
               - Drop fitur dengan missing value > 50%
               (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`)
               - Imputasi median (numerik) / modus (kategorikal) untuk missing < 50%
               - Feature engineering: `total_sqft`, `age`, `remod_age`, `price_per_sqft`
               - One-Hot Encoding untuk fitur kategorikal
               - Standardisasi fitur dengan `StandardScaler`
               - Split data training/validasi 80:20
            4. **Modeling** — melatih & membandingkan 3 algoritma regresi.
            5. **Evaluation** — RMSE, MAE, R², analisis residual, dan APE.
            6. **Deployment** — model terbaik disimpan sebagai `.pkl` dan
               digunakan pada aplikasi Streamlit ini.
            """
        )

    with st.expander("⚙️ Pipeline Machine Learning"):
        st.code(
            """
Raw Data (train.csv + test.csv)
        │
        ▼
Drop fitur missing > 50%
        │
        ▼
Imputasi missing values (median / modus)
        │
        ▼
Feature Engineering (total_sqft, age, remod_age, price_per_sqft)
        │
        ▼
One-Hot Encoding (fitur kategorikal)
        │
        ▼
StandardScaler (standardisasi seluruh fitur numerik)
        │
        ▼
Train-Test Split (80:20)
        │
        ▼
Training Model (Linear Regression / Random Forest / XGBoost)
        │
        ▼
Evaluasi (RMSE, MAE, R², Residual Analysis)
        │
        ▼
Pemilihan Model Terbaik → best_model.pkl + scaler.pkl
            """,
            language="text",
        )

    with st.expander("🤖 Model yang Dibandingkan"):
        st.markdown(
            """
            | Model | Deskripsi |
            |---|---|
            | **Linear Regression** | Model dasar (baseline) sebagai pembanding. |
            | **Random Forest Regressor** | Ensemble *bagging*, dituning dengan `GridSearchCV`. |
            | **XGBoost Regressor** | Ensemble *gradient boosting*, dituning dengan `GridSearchCV`. |
            """
        )

    with st.expander("🏆 Model Terbaik"):
        st.write(f"Model dengan performa validasi terbaik adalah **{MODEL_NAME}**, "
                 "dipilih berdasarkan nilai R² tertinggi pada data validasi "
                 "(sesuai kriteria pemilihan pada notebook Colab).")
        if eval_results:
            st.write(
                f"- Validation RMSE: **{format_usd(eval_results['rmse'])}**\n"
                f"- Validation MAE: **{format_usd(eval_results['mae'])}**\n"
                f"- Validation R²: **{eval_results['r2']:.4f}**"
            )

    with st.expander("🖱️ Cara Menggunakan Aplikasi"):
        st.write(
            """
            1. Buka menu **Dashboard EDA** untuk melihat eksplorasi data.
            2. Buka menu **House Price Prediction**, isi form karakteristik
               rumah (kualitas, luas bangunan, jumlah kamar, tahun dibangun, dll.).
            3. Klik tombol **Predict House Price** untuk melihat estimasi harga.
            4. Buka menu **Model Evaluation** untuk melihat metrik performa
               model (RMSE, MAE, R²) beserta grafik pendukung.
            """
        )

    with st.expander("🛠️ Catatan Teknis untuk Developer (Setup File Model)"):
        st.write(
            """
            Aplikasi ini membutuhkan beberapa file tambahan (di luar
            `best_model.pkl` dan `scaler.pkl`) supaya form prediksi dengan
            10 input sederhana tetap menghasilkan struktur fitur yang PERSIS
            sama dengan saat model dilatih (karena hasil One-Hot Encoding di
            notebook menghasilkan ratusan kolom, bukan hanya 10).

            Tambahkan cell berikut **di akhir notebook Colab** (tanpa mengubah
            kode ML yang sudah ada) untuk menghasilkan file pendukung:

            ```python
            import joblib

            # Kolom hasil encoding, urutannya harus identik saat inference
            feature_columns = X_train_encoded.columns.tolist()
            joblib.dump(feature_columns, 'feature_columns.pkl')

            # Nilai median tiap kolom sebagai baseline fitur yang tidak
            # muncul di form prediksi
            feature_medians = X_train_encoded.median()
            joblib.dump(feature_medians, 'feature_medians.pkl')

            # Hasil evaluasi untuk ditampilkan di menu Model Evaluation
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
            ```

            Lalu letakkan semua file `.pkl` di folder `model/`, dan
            (opsional) `train.csv` di folder `data/` agar Dashboard EDA
            menampilkan data asli.

            ⚠️ **Catatan mengenai `price_per_sqft`**: fitur ini dihitung dari
            `SalePrice` (target) pada notebook asli, sehingga tidak dapat
            dihitung ulang saat prediksi (harga belum diketahui). Pada
            aplikasi ini, nilai fitur tersebut memakai nilai **median** dari
            data training sebagai baseline.
            """
        )

# ==================================================================
# FOOTER
# ==================================================================
st.markdown(
    """
    <div class="footer">
        🏠 House Price Prediction App &nbsp;|&nbsp; Built with Streamlit &amp; Machine Learning
        <br>Universitas Dian Nuswantoro — Pembelajaran Mesin
    </div>
    """,
    unsafe_allow_html=True,
)
