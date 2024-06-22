import yfinance as yf
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st

# Kontributor atau Anggota Tim
kontributor = [
    "Annalia Alfia Rahma (13012131981)",
    "Tsania Millatina Aghnia Fariha (1301210051)",
    "Putri Ayu Sedyo Mukti (1301213453)",
    "Kartina Halawa (1301210245)"
]

# Streamlit interface untuk interaksi pengguna
st.title("Prediksi Harga Saham menggunakan Pembelajaran Mesin dan GBM")

# Menampilkan kontributor di sidebar
st.sidebar.header("Kontributor")
for nama in kontributor:
    st.sidebar.write(nama)

# Input sebelum simulasi
default_ticker = "GOTO.JK"
ticker = st.sidebar.text_input("Masukkan kode saham:", default_ticker)
start_date = st.sidebar.date_input("Tanggal mulai:", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Tanggal akhir:", pd.to_datetime("2023-12-31"))
features = st.sidebar.multiselect("Pilih fitur", options=['volume_adi', 'momentum_rsi', 'trend_sma_fast', 'trend_ema_fast', 'volatility_bbh'], default=['momentum_rsi'])

# Memuat dan menyiapkan data
data = load_data(ticker, start_date, end_date)
data, scaler = prepare_features(data, features)

# Mendefinisikan tombol jalankan simulasi
simulate = st.sidebar.button("Jalankan Simulasi")

# Simulasi GBM dan prediksi model
if simulate:
    model, train_data, test_data = train_xgb_with_split(data, features, 'logreturn')
    spot_price = test_data["Adj Close"].iloc[-1]
    volatility = test_data['logreturn'].std() * np.sqrt(252)
    simulated_paths, drifts = gbm_sim_xgb(spot_price, volatility, len(test_data), model, features, test_data)

    # Menampilkan Hasil
    st.line_chart(test_data["Adj Close"].rename('Harga Sebenarnya'))
    st.line_chart(pd.Series(simulated_paths, index=test_data.index).rename('Harga Simulasi'))
    abs_error = [abs(i-j) for (i, j) in zip(simulated_paths, test_data['Adj Close'].values)]
    rel_error = [abs(i-j)/j*100 for (i, j) in zip(simulated_paths, test_data['Adj Close'].values)]
    st.line_chart(pd.DataFrame({'Error Absolut': abs_error, 'Error Relatif (%)': rel_error}, index=test_data.index))

    y_true = test_data['logreturn'].values
    y_pred = drifts
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    st.write(f"Skor R²: {r2}")
    st.write(f"Mean Squared Error (MSE): {mse}")
