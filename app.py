import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure'logistic_regression_model.pkl' is in the same directory.")
    st.stop()
# Judul aplikasi Streamlit
st.title('Aplikasi Prediksi Persetujuan Pinjaman')
st.write("""Aplikasi ini memprediksi apakah pinjaman akan disetujui berdasarkan input dari pengguna.""")
# Buat form input untuk data dummy
st.header('Masukkan Data Calon Peminjam')
gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
married = st.selectbox('Status Pernikahan', ['Ya', 'Tidak'])
dependents = st.selectbox('Jumlah Tanggungan', ['0', '1', '2', '3+'])
education = st.selectbox('Pendidikan', ['Sarjana', 'Tidak Sarjana'])
self_employed = st.selectbox('Wiraswasta', ['Ya', 'Tidak'])
applicant_income = st.number_input('Pendapatan Pemohon', min_value=0)
coapplicant_income = st.number_input('Pendapatan Pasangan', min_value=0)
loan_amount = st.number_input('Jumlah Pinjaman (dalam ribuan)',
min_value=0.0)
loan_amount_term = st.selectbox('Jangka Waktu Pinjaman (dalam bulan)',
[12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0])
credit_history = st.selectbox('Riwayat Kredit (1 jika ada, 0 jika tidak)',
[0.0, 1.0])
property_area = st.selectbox('Area Properti', ['Pedesaan', 'Semiurban',
'Perkotaan'])
# Tombol untuk memprediksi
if st.button('Prediksi Persetujuan Pinjaman'):
# Buat DataFrame dari input pengguna
    input_data = {
    'Gender': [1 if gender == 'Laki-laki' else 2],
    'Married': [1 if married == 'Ya' else 2],
    'Dependents': [3 if dependents == '3+' else int(dependents)],
    'Education': [1 if education == 'Sarjana' else 2],
    'Self_Employed': [1 if self_employed == 'Ya' else 2],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [1 if property_area == 'Pedesaan' else (2 if
    property_area == 'Semiurban' else 3)]
    }
input_df = pd.DataFrame(input_data)
try:
    input_df = input_df[['Gender', 'Married', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']]
except KeyError as e:
    st.error(f"Column mismatch: {e}. Please check the column names and order.")
    st.stop()
# Lakukan prediksi
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)[:, 1]
# Tampilkan hasil prediksi
st.header('Hasil Prediksi')
if prediction[0] == 1:
    st.success('Pinjaman DIPREDIKSI DISETUJUI')
else:
    st.error('Pinjaman DIPREDIKSI DITOLAK')
    st.write(f"Probabilitas Persetujuan: {prediction_proba[0]:.2f}")