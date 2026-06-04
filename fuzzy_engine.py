"""
fuzzy_engine.py
===============
Modul inti dari sistem Fuzzy Logic Mamdani dan Sugeno
untuk prediksi risiko penyakit kardiovaskular.

Dibuat berdasarkan Tugas Besar DKA - Kelompok Leker
Savila Nur Fadilla | Dwi Okta Suryaningrum | Sheila Stephanie Anindya
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNGSI KEANGGOTAAN (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

def fungsi_segitiga(x, a, b, c):
    """Kurva segitiga: naik dari a ke b, turun dari b ke c."""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0


def fungsi_trapesium(x, a, b, c, d):
    """Kurva trapesium: naik dari a ke b, flat dari b ke c, turun dari c ke d."""
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. FUZZIFIKASI VARIABEL INPUT
# ─────────────────────────────────────────────────────────────────────────────

def hitung_fuzzy_usia(x):
    return {
        'muda': fungsi_trapesium(x, 0, 0, 35, 45),
        'paruh_baya': fungsi_segitiga(x, 35, 50, 65),
        'tua': fungsi_trapesium(x, 55, 65, 100, 100),
    }


def hitung_fuzzy_sistolik(x):
    return {
        'normal': fungsi_trapesium(x, 80, 80, 110, 125),
        'pre_hipertensi': fungsi_segitiga(x, 115, 130, 140),
        'hipertensi': fungsi_trapesium(x, 135, 150, 200, 200),
    }


def hitung_fuzzy_diastolik(x):
    return {
        'normal': fungsi_trapesium(x, 40, 40, 70, 80),
        'pre_hipertensi': fungsi_segitiga(x, 75, 85, 90),
        'hipertensi': fungsi_trapesium(x, 85, 95, 120, 120),
    }


def hitung_fuzzy_kolesterol(x):
    return {
        'normal': fungsi_segitiga(x, 0, 1, 2),
        'di_atas_normal': fungsi_segitiga(x, 1, 2, 3),
        'tinggi': fungsi_segitiga(x, 2, 3, 4),
    }


def hitung_fuzzy_bmi(x):
    return {
        'normal': fungsi_trapesium(x, 10, 10, 18.5, 25),
        'overweight': fungsi_segitiga(x, 23, 27, 30),
        'obesitas': fungsi_trapesium(x, 29, 35, 50, 50),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFERENSI — 15 RULE BASE
# ─────────────────────────────────────────────────────────────────────────────

def jalankan_inferensi_15_rules(u, sis, dias, kol, bmi):
    """
    Mengembalikan list tuple (kategori_output, kekuatan_aturan).
    Operator AND = min().
    """
    rules = []
    # R1  – Tua + Sistolik Hipertensi → Tinggi
    rules.append(('tinggi',  min(u['tua'],        sis['hipertensi'])))
    # R2  – Kolesterol Tinggi + BMI Obesitas → Tinggi
    rules.append(('tinggi',  min(kol['tinggi'],   bmi['obesitas'])))
    # R3  – Sistolik Normal + Diastolik Normal → Sedang
    rules.append(('sedang',  min(sis['normal'],   dias['normal'])))
    # R4  – Muda + BMI Normal → Rendah
    rules.append(('rendah',  min(u['muda'],        bmi['normal'])))
    # R5  – Kolesterol Normal + Sistolik Normal → Rendah
    rules.append(('rendah',  min(kol['normal'],   sis['normal'])))
    # R6  – Paruh Baya + Sistolik Pre-Hipertensi → Sedang
    rules.append(('sedang',  min(u['paruh_baya'], sis['pre_hipertensi'])))
    # R7  – Diastolik Hipertensi + BMI Obesitas → Tinggi
    rules.append(('tinggi',  min(dias['hipertensi'], bmi['obesitas'])))
    # R8  – Kolesterol Di Atas Normal + BMI Overweight → Sedang
    rules.append(('sedang',  min(kol['di_atas_normal'], bmi['overweight'])))
    # R9  – Tua + Kolesterol Tinggi → Tinggi
    rules.append(('tinggi',  min(u['tua'],        kol['tinggi'])))
    # R10 – Muda + Diastolik Pre-Hipertensi → Sedang
    rules.append(('sedang',  min(u['muda'],        dias['pre_hipertensi'])))
    # R11 – Sistolik Hipertensi + Kolesterol Tinggi → Tinggi
    rules.append(('tinggi',  min(sis['hipertensi'], kol['tinggi'])))
    # R12 – Paruh Baya + BMI Normal → Sedang
    rules.append(('sedang',  min(u['paruh_baya'], bmi['normal'])))
    # R13 – Diastolik Normal + Kolesterol Normal → Rendah
    rules.append(('rendah',  min(dias['normal'],  kol['normal'])))
    # R14 – Sistolik Pre-Hipertensi + BMI Overweight → Sedang
    rules.append(('sedang',  min(sis['pre_hipertensi'], bmi['overweight'])))
    # R15 – Tua + BMI Obesitas → Tinggi
    rules.append(('tinggi',  min(u['tua'],        bmi['obesitas'])))
    return rules


def get_rule_details(u, sis, dias, kol, bmi):
    """Mengembalikan detail setiap rule untuk ditampilkan di UI."""
    rule_defs = [
        ("R1",  "Tua ∧ Sistolik Hipertensi", "Tinggi",
         min(u['tua'], sis['hipertensi'])),
        ("R2",  "Kol Tinggi ∧ BMI Obesitas", "Tinggi",
         min(kol['tinggi'], bmi['obesitas'])),
        ("R3",  "Sistolik Normal ∧ Diastolik Normal", "Sedang",
         min(sis['normal'], dias['normal'])),
        ("R4",  "Muda ∧ BMI Normal", "Rendah",
         min(u['muda'], bmi['normal'])),
        ("R5",  "Kol Normal ∧ Sistolik Normal", "Rendah",
         min(kol['normal'], sis['normal'])),
        ("R6",  "Paruh Baya ∧ Sistolik Pre-Hipertensi", "Sedang",
         min(u['paruh_baya'], sis['pre_hipertensi'])),
        ("R7",  "Diastolik Hipertensi ∧ BMI Obesitas", "Tinggi",
         min(dias['hipertensi'], bmi['obesitas'])),
        ("R8",  "Kol Di Atas Normal ∧ BMI Overweight", "Sedang",
         min(kol['di_atas_normal'], bmi['overweight'])),
        ("R9",  "Tua ∧ Kol Tinggi", "Tinggi",
         min(u['tua'], kol['tinggi'])),
        ("R10", "Muda ∧ Diastolik Pre-Hipertensi", "Sedang",
         min(u['muda'], dias['pre_hipertensi'])),
        ("R11", "Sistolik Hipertensi ∧ Kol Tinggi", "Tinggi",
         min(sis['hipertensi'], kol['tinggi'])),
        ("R12", "Paruh Baya ∧ BMI Normal", "Sedang",
         min(u['paruh_baya'], bmi['normal'])),
        ("R13", "Diastolik Normal ∧ Kol Normal", "Rendah",
         min(dias['normal'], kol['normal'])),
        ("R14", "Sistolik Pre-Hipertensi ∧ BMI Overweight", "Sedang",
         min(sis['pre_hipertensi'], bmi['overweight'])),
        ("R15", "Tua ∧ BMI Obesitas", "Tinggi",
         min(u['tua'], bmi['obesitas'])),
    ]
    return rule_defs


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFUZZIFIKASI
# ─────────────────────────────────────────────────────────────────────────────

def defuzzifikasi_sugeno(hasil_rules):
    """Rata-rata terbobot (Weighted Average) — Singleton: Rendah=20, Sedang=50, Tinggi=80."""
    bobot = {'rendah': 20, 'sedang': 50, 'tinggi': 80}
    num, den = 0.0, 0.0
    for kategori, kekuatan in hasil_rules:
        num += kekuatan * bobot[kategori]
        den += kekuatan
    return num / den if den != 0 else 20.0


def defuzzifikasi_mamdani(hasil_rules):
    """Centroid Diskrit (Integral Riemann) dengan 1000 titik sampel."""
    x_sampel = np.linspace(0, 100, 1000)
    num, den = 0.0, 0.0
    for x in x_sampel:
        mu_rendah = fungsi_trapesium(x, 0, 0, 20, 40)
        mu_sedang = fungsi_segitiga(x, 30, 50, 70)
        mu_tinggi = fungsi_trapesium(x, 60, 80, 100, 100)

        area_aktif = 0.0
        for kategori, kekuatan in hasil_rules:
            if kategori == 'rendah':
                mu_potong = min(kekuatan, mu_rendah)
            elif kategori == 'sedang':
                mu_potong = min(kekuatan, mu_sedang)
            else:
                mu_potong = min(kekuatan, mu_tinggi)
            area_aktif = max(area_aktif, mu_potong)

        num += x * area_aktif
        den += area_aktif
    return num / den if den != 0 else 20.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PREDIKSI SATU PASIEN
# ─────────────────────────────────────────────────────────────────────────────

def prediksi_pasien(usia, sistolik, diastolik, kolesterol, bmi, metode='sugeno'):
    """
    Melakukan prediksi risiko kardiovaskular untuk satu pasien.

    Parameters
    ----------
    usia       : float  (tahun)
    sistolik   : float  (mmHg)
    diastolik  : float  (mmHg)
    kolesterol : int    (1=Normal, 2=Di Atas Normal, 3=Tinggi)
    bmi        : float  (kg/m²)
    metode     : str    'sugeno' atau 'mamdani'

    Returns
    -------
    skor       : float  0–100 (skor risiko)
    label      : str    'Berisiko Tinggi' / 'Berisiko Sedang' / 'Berisiko Rendah'
    fuzzy_vals : dict   derajat keanggotaan setiap variabel input
    rules      : list   detail 15 rule base
    """
    u   = hitung_fuzzy_usia(usia)
    sis = hitung_fuzzy_sistolik(sistolik)
    dia = hitung_fuzzy_diastolik(diastolik)
    kol = hitung_fuzzy_kolesterol(kolesterol)
    bm  = hitung_fuzzy_bmi(bmi)

    hasil_rules = jalankan_inferensi_15_rules(u, sis, dia, kol, bm)
    rule_detail = get_rule_details(u, sis, dia, kol, bm)

    if metode == 'mamdani':
        skor = defuzzifikasi_mamdani(hasil_rules)
    else:
        skor = defuzzifikasi_sugeno(hasil_rules)

    if skor >= 60:
        label = 'Berisiko Tinggi'
    elif skor >= 40:
        label = 'Berisiko Sedang'
    else:
        label = 'Berisiko Rendah'

    fuzzy_vals = {
        'Usia': u,
        'Sistolik': sis,
        'Diastolik': dia,
        'Kolesterol': kol,
        'BMI': bm,
    }

    return skor, label, fuzzy_vals, rule_detail