import streamlit as st
import pandas as pd
import numpy as np

# --- 1. LOGIKA FUZZY FROM SCRATCH ---
def fungsi_segitiga(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    elif b < x < c: return (c - x) / (c - b)
    return 0.0

def fungsi_trapesium(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    elif b < x <= c: return 1.0
    elif c < x < d: return (d - x) / (d - c)
    return 0.0

# --- 2. INTERFACE STREAMLIT ---
st.title("Prediksi Risiko Kardiovaskular (Fuzzy Mamdani & Sugeno)")
st.write("Masukkan data medis Anda di bawah ini:")

# Input User
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=50)
    sys = st.number_input("Tekanan Darah Sistolik (ap_hi)", value=120)
    dia = st.number_input("Tekanan Darah Diastolik (ap_lo)", value=80)
with col2:
    chol = st.selectbox("Kolesterol", [1, 2, 3], help="1: Normal, 2: Tinggi, 3: Sangat Tinggi")
    weight = st.number_input("Berat Badan (kg)", value=70)
    height = st.number_input("Tinggi Badan (cm)", value=170)

# Hitung BMI otomatis seperti di Colab
bmi = weight / ((height/100)**2)
st.info(f"BMI Anda: {bmi:.2f}")

if st.button("Hitung Risiko"):
    # Di sini masukkan fungsi inferensi Mamdani/Sugeno dari file Colab kamu
    # Contoh pemanggilan sederhana:
    # hasil_mamdani = hitung_mamdani(age, sys, dia, chol, bmi)
    st.success(f"Hasil Prediksi: Risiko Tinggi")