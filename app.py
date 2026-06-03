import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Risiko Kardiovaskular", layout="wide")

# =====================================================
# FUNGSI KEANGGOTAAN
# =====================================================

def fungsi_segitiga(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0


def fungsi_trapesium(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)
    return 0.0


# =====================================================
# FUZZIFIKASI
# =====================================================

def hitung_fuzzy_usia(x):
    return {
        'muda': fungsi_trapesium(x, 0, 0, 35, 45),
        'paruh_baya': fungsi_segitiga(x, 35, 50, 65),
        'tua': fungsi_trapesium(x, 55, 65, 100, 100)
    }


def hitung_fuzzy_sistolik(x):
    return {
        'normal': fungsi_trapesium(x, 80, 80, 110, 125),
        'prehiper': fungsi_segitiga(x, 115, 130, 140),
        'hipertensi': fungsi_trapesium(x, 135, 150, 200, 200)
    }


def hitung_fuzzy_diastolik(x):
    return {
        'normal': fungsi_trapesium(x, 40, 40, 70, 80),
        'prehiper': fungsi_segitiga(x, 75, 85, 90),
        'hipertensi': fungsi_trapesium(x, 85, 95, 120, 120)
    }


def hitung_fuzzy_kolesterol(x):
    return {
        'normal': fungsi_segitiga(x, 0, 1, 2),
        'di_atas_normal': fungsi_segitiga(x, 1, 2, 3),
        'tinggi': fungsi_segitiga(x, 2, 3, 4)
    }


def hitung_fuzzy_bmi(x):
    return {
        'normal': fungsi_trapesium(x, 10, 10, 18.5, 25),
        'overweight': fungsi_segitiga(x, 23, 27, 30),
        'obesitas': fungsi_trapesium(x, 29, 35, 50, 50)
    }


# =====================================================
# RULE BASE (15 RULES)
# =====================================================

def jalankan_inferensi_15_rules(u, sis, dias, kol, bmi):

    rules = []

    rules.append(('tinggi', min(u['tua'], sis['hipertensi'])))
    rules.append(('tinggi', min(kol['tinggi'], bmi['obesitas'])))
    rules.append(('sedang', min(sis['normal'], dias['normal'])))
    rules.append(('rendah', min(u['muda'], bmi['normal'])))
    rules.append(('rendah', min(kol['normal'], sis['normal'])))
    rules.append(('sedang', min(u['paruh_baya'], sis['prehiper'])))
    rules.append(('tinggi', min(dias['hipertensi'], bmi['obesitas'])))
    rules.append(('sedang', min(kol['di_atas_normal'], bmi['overweight'])))
    rules.append(('tinggi', min(u['tua'], kol['tinggi'])))
    rules.append(('sedang', min(u['muda'], dias['prehiper'])))
    rules.append(('tinggi', min(sis['hipertensi'], kol['tinggi'])))
    rules.append(('sedang', min(u['paruh_baya'], bmi['normal'])))
    rules.append(('rendah', min(dias['normal'], kol['normal'])))
    rules.append(('sedang', min(sis['prehiper'], bmi['overweight'])))
    rules.append(('tinggi', min(u['tua'], bmi['obesitas'])))

    return rules


# =====================================================
# DEFUZZIFIKASI
# =====================================================

def defuzzifikasi_sugeno(hasil_rules):

    bobot = {
        'rendah': 20,
        'sedang': 50,
        'tinggi': 80
    }

    num = 0
    den = 0

    for kategori, alpha in hasil_rules:
        num += alpha * bobot[kategori]
        den += alpha

    return num / den if den != 0 else 20


def defuzzifikasi_mamdani(hasil_rules):

    x_sampel = np.linspace(0, 100, 1000)

    num = 0
    den = 0

    agregasi = []

    for x in x_sampel:

        mu_rendah = fungsi_trapesium(x, 0, 0, 20, 40)
        mu_sedang = fungsi_segitiga(x, 30, 50, 70)
        mu_tinggi = fungsi_trapesium(x, 60, 80, 100, 100)

        area_aktif = 0

        for kategori, alpha in hasil_rules:

            if kategori == 'rendah':
                mu = min(alpha, mu_rendah)

            elif kategori == 'sedang':
                mu = min(alpha, mu_sedang)

            else:
                mu = min(alpha, mu_tinggi)

            area_aktif = max(area_aktif, mu)

        agregasi.append(area_aktif)

        num += x * area_aktif
        den += area_aktif

    centroid = num / den if den != 0 else 20

    return centroid, x_sampel, agregasi


def kategori_risiko(skor):

    if skor < 40:
        return "🟢 Risiko Rendah"

    elif skor < 60:
        return "🟡 Risiko Sedang"

    return "🔴 Risiko Tinggi"


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("Prediksi Risiko Kardiovaskular")
st.subheader("Metode Fuzzy Mamdani & Sugeno")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia", 1, 100, 45)
    sistolik = st.number_input("Tekanan Sistolik", 80, 250, 120)
    diastolik = st.number_input("Tekanan Diastolik", 40, 150, 80)

with col2:
    kolesterol = st.selectbox(
        "Kolesterol",
        [1, 2, 3],
        format_func=lambda x:
        {
            1: "Normal",
            2: "Di Atas Normal",
            3: "Tinggi"
        }[x]
    )

    berat = st.number_input("Berat Badan (kg)", 20.0, 200.0, 70.0)
    tinggi = st.number_input("Tinggi Badan (cm)", 100.0, 250.0, 170.0)

bmi = berat / ((tinggi/100)**2)

st.info(f"BMI : {bmi:.2f}")

if st.button("Hitung Risiko"):

    fuzzy_usia = hitung_fuzzy_usia(usia)
    fuzzy_sis = hitung_fuzzy_sistolik(sistolik)
    fuzzy_dias = hitung_fuzzy_diastolik(diastolik)
    fuzzy_kol = hitung_fuzzy_kolesterol(kolesterol)
    fuzzy_bmi = hitung_fuzzy_bmi(bmi)

    rules = jalankan_inferensi_15_rules(
        fuzzy_usia,
        fuzzy_sis,
        fuzzy_dias,
        fuzzy_kol,
        fuzzy_bmi
    )

    skor_sugeno = defuzzifikasi_sugeno(rules)

    skor_mamdani, x, y = defuzzifikasi_mamdani(rules)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Skor Sugeno",
            f"{skor_sugeno:.2f}"
        )

        st.success(
            kategori_risiko(skor_sugeno)
        )

    with col2:
        st.metric(
            "Skor Mamdani",
            f"{skor_mamdani:.2f}"
        )

        st.success(
            kategori_risiko(skor_mamdani)
        )

    st.subheader("Grafik Defuzzifikasi Mamdani")

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(x, y, linewidth=2)

    ax.axvline(
        skor_mamdani,
        linestyle='--',
        linewidth=2,
        label=f'Centroid = {skor_mamdani:.2f}'
    )

    ax.set_xlabel("Skor Risiko")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.set_title("Hasil Agregasi Fuzzy Mamdani")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)