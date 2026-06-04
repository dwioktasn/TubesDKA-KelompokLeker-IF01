"""
app.py — Aplikasi Streamlit
===========================
Sistem Prediksi Risiko Kardiovaskular menggunakan
Fuzzy Logic Mamdani & Sugeno (From Scratch)

Kelompok Leker | Dasar Kecerdasan Artificial
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend untuk Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from fuzzy_engine import (
    hitung_fuzzy_usia, hitung_fuzzy_sistolik, hitung_fuzzy_diastolik,
    hitung_fuzzy_kolesterol, hitung_fuzzy_bmi,
    fungsi_segitiga, fungsi_trapesium,
    prediksi_pasien,
)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fuzzy Cardio Risk — Kelompok Leker",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS KUSTOM
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Nunito:wght@400;600;700;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 2px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #334155;
    border-radius: 10px;
    background: #0f172a;
}

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: WARNA RISIKO
# ─────────────────────────────────────────────────────────────────────────────

def warna_risiko(label: str):
    return {
        'Berisiko Tinggi':  ('#ef4444', '🔴'),
        'Berisiko Sedang':  ('#f59e0b', '🟡'),
        'Berisiko Rendah':  ('#22c55e', '🟢'),
    }.get(label, ('#94a3b8', '⚪'))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PLOT FUNGSI KEANGGOTAAN
# ─────────────────────────────────────────────────────────────────────────────

def plot_membership(ax, x_range, curves, title, xlabel, nilai_aktual=None):
    colors = ['#38bdf8', '#fb923c', '#f87171']
    ax.set_facecolor('#0f172a')
    ax.figure.set_facecolor('#1e293b')
    for (label, y_vals), color in zip(curves, colors):
        ax.plot(x_range, y_vals, label=label, color=color, linewidth=2)
    if nilai_aktual is not None:
        ax.axvline(x=nilai_aktual, color='#facc15', linestyle='--',
                   linewidth=1.5, label=f'Input: {nilai_aktual:.1f}')
    ax.set_title(title, fontsize=10, color='#e2e8f0', pad=6,
                 fontfamily='monospace')
    ax.set_xlabel(xlabel, fontsize=8, color='#94a3b8')
    ax.set_ylabel('μ(x)', fontsize=8, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#e2e8f0', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.25, color='#334155')
    ax.set_ylim(-0.05, 1.15)


def gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#1e293b')
    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    # ─ Usia
    x = np.linspace(0, 100, 500)
    plot_membership(axes[0, 0], x, [
        ('Muda',      [fungsi_trapesium(v, 0, 0, 35, 45)    for v in x]),
        ('Paruh Baya',[fungsi_segitiga(v, 35, 50, 65)        for v in x]),
        ('Tua',       [fungsi_trapesium(v, 55, 65, 100, 100) for v in x]),
    ], 'Usia', 'Tahun', usia)

    # ─ Sistolik
    x = np.linspace(80, 200, 500)
    plot_membership(axes[0, 1], x, [
        ('Normal',         [fungsi_trapesium(v, 80, 80, 110, 125) for v in x]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 115, 130, 140)      for v in x]),
        ('Hipertensi',     [fungsi_trapesium(v, 135, 150, 200, 200)for v in x]),
    ], 'Tekanan Darah Sistolik', 'mmHg', sistolik)

    # ─ Diastolik
    x = np.linspace(40, 120, 500)
    plot_membership(axes[0, 2], x, [
        ('Normal',         [fungsi_trapesium(v, 40, 40, 70, 80) for v in x]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 75, 85, 90)       for v in x]),
        ('Hipertensi',     [fungsi_trapesium(v, 85, 95, 120, 120)for v in x]),
    ], 'Tekanan Darah Diastolik', 'mmHg', diastolik)

    # ─ Kolesterol
    x = np.linspace(0, 4, 500)
    plot_membership(axes[1, 0], x, [
        ('Normal',         [fungsi_segitiga(v, 0, 1, 2) for v in x]),
        ('Di Atas Normal', [fungsi_segitiga(v, 1, 2, 3) for v in x]),
        ('Tinggi',         [fungsi_segitiga(v, 2, 3, 4) for v in x]),
    ], 'Kolesterol', 'Kategori (1/2/3)', kolesterol)

    # ─ BMI
    x = np.linspace(10, 50, 500)
    plot_membership(axes[1, 1], x, [
        ('Normal',     [fungsi_trapesium(v, 10, 10, 18.5, 25) for v in x]),
        ('Overweight', [fungsi_segitiga(v, 23, 27, 30)         for v in x]),
        ('Obesitas',   [fungsi_trapesium(v, 29, 35, 50, 50)    for v in x]),
    ], 'BMI', 'kg/m²', bmi)

    # ─ Output (kosongkan subplot terakhir)
    axes[1, 2].set_facecolor('#0f172a')
    axes[1, 2].set_title('Output Risiko', fontsize=10, color='#e2e8f0',
                          pad=6, fontfamily='monospace')
    x = np.linspace(0, 100, 500)
    axes[1, 2].plot(x, [fungsi_trapesium(v, 0, 0, 20, 40)    for v in x],
                    color='#22c55e', linewidth=2, label='Rendah')
    axes[1, 2].plot(x, [fungsi_segitiga(v, 30, 50, 70)        for v in x],
                    color='#f59e0b', linewidth=2, label='Sedang')
    axes[1, 2].plot(x, [fungsi_trapesium(v, 60, 80, 100, 100) for v in x],
                    color='#ef4444', linewidth=2, label='Tinggi')
    axes[1, 2].set_xlabel('Skor Risiko (%)', fontsize=8, color='#94a3b8')
    axes[1, 2].set_ylabel('μ(x)', fontsize=8, color='#94a3b8')
    axes[1, 2].tick_params(colors='#94a3b8', labelsize=7)
    for spine in axes[1, 2].spines.values():
        spine.set_edgecolor('#334155')
    axes[1, 2].legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
                      labelcolor='#e2e8f0', loc='upper right')
    axes[1, 2].grid(True, linestyle='--', alpha=0.25, color='#334155')
    axes[1, 2].set_ylim(-0.05, 1.15)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────

def gambar_gauge(skor: float, label: str):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    theta = np.linspace(np.pi, 0, 300)
    r = 1.0

    # Background arc
    ax.plot(theta, [r] * 300, color='#334155', linewidth=12, solid_capstyle='round')

    # Colored arc (0–33 hijau, 33–66 kuning, 66–100 merah)
    colors_seg = [
        (np.linspace(np.pi,   2*np.pi/3, 100), '#22c55e'),
        (np.linspace(2*np.pi/3, np.pi/3, 100), '#f59e0b'),
        (np.linspace(np.pi/3,  0,        100), '#ef4444'),
    ]
    for seg, col in colors_seg:
        ax.plot(seg, [r] * 100, color=col, linewidth=12, solid_capstyle='butt')

    # Needle
    angle = np.pi * (1 - skor / 100)
    ax.annotate('', xy=(angle, 0.75), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#facc15',
                                lw=2.5, mutation_scale=18))

    # Center dot
    ax.plot(0, 0, 'o', color='#facc15', markersize=8, zorder=10)

    warna, _ = warna_risiko(label)
    ax.text(np.pi / 2, 0.15, f'{skor:.1f}%', ha='center', va='center',
            fontsize=18, fontweight='bold', color=warna, fontfamily='monospace')
    ax.text(np.pi / 2, -0.25, label, ha='center', va='center',
            fontsize=9, color='#e2e8f0')

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PASIEN
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫀 Fuzzy Cardio Risk")
    st.markdown("*Kelompok Leker — DKA*")
    st.markdown("---")

    st.markdown("### ⚙️ Metode Inferensi")
    metode = st.radio(
        "Pilih metode defuzzifikasi:",
        options=['Sugeno (Cepat)', 'Mamdani (Akurat)'],
        index=0,
        help="Sugeno: Weighted Average · Mamdani: Centroid Diskrit (lebih lambat)"
    )
    metode_key = 'sugeno' if 'Sugeno' in metode else 'mamdani'

    st.markdown("---")
    st.markdown("### 📋 Data Klinis Pasien")

    usia = st.slider("Usia (tahun)", 18, 90, 45,
                     help="Usia pasien dalam tahun")
    sistolik = st.slider("Tekanan Sistolik (mmHg)", 80, 200, 120,
                         help="Tekanan darah sistolik (angka atas)")
    diastolik = st.slider("Tekanan Diastolik (mmHg)", 40, 120, 80,
                          help="Tekanan darah diastolik (angka bawah)")
    kolesterol = st.select_slider(
        "Kolesterol",
        options=[1, 2, 3],
        value=1,
        format_func=lambda v: {1: '1 — Normal', 2: '2 — Di Atas Normal', 3: '3 — Tinggi'}[v],
        help="Kategori kolesterol sesuai dataset Kaggle"
    )
    tinggi_cm = st.slider("Tinggi Badan (cm)", 140, 210, 165)
    berat_kg  = st.slider("Berat Badan (kg)", 40, 150, 70)
    bmi       = round(berat_kg / (tinggi_cm / 100) ** 2, 2)
    st.metric("BMI Terhitung", f"{bmi:.2f} kg/m²",
              delta=f"{'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obesitas'}")

    st.markdown("---")
    hitung = st.button("🔍 Hitung Risiko", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🫀 Sistem Prediksi Risiko Kardiovaskular")
st.markdown(
    "**Fuzzy Logic Mamdani & Sugeno — From Scratch** · "
    "Kelompok Leker · Dasar Kecerdasan Artificial"
)
st.markdown("---")

# Tab navigasi
tab_prediksi, tab_membership, tab_rules, tab_info = st.tabs([
    "📊 Prediksi", "📈 Fungsi Keanggotaan", "📋 Rule Base", "ℹ️ Tentang"
])


# ── TAB 1: PREDIKSI ──────────────────────────────────────────────────────────
with tab_prediksi:
    if hitung or 'last_result' in st.session_state:

        # Hitung (atau ambil dari cache)
        if hitung:
            with st.spinner(
                "⏳ Menjalankan Integral Riemann Diskrit (1000 titik)…"
                if metode_key == 'mamdani' else "⚡ Menghitung Weighted Average…"
            ):
                t0 = time.perf_counter()
                skor, label, fuzzy_vals, rule_detail = prediksi_pasien(
                    usia, sistolik, diastolik, kolesterol, bmi, metode=metode_key
                )
                elapsed = time.perf_counter() - t0
            st.session_state['last_result'] = (skor, label, fuzzy_vals, rule_detail, elapsed, metode_key)
        else:
            skor, label, fuzzy_vals, rule_detail, elapsed, metode_key = st.session_state['last_result']

        warna, ikon = warna_risiko(label)

        # ── HASIL UTAMA
        col_gauge, col_info = st.columns([1, 2])

        with col_gauge:
            fig_gauge = gambar_gauge(skor, label)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        with col_info:
            st.markdown(f"### Hasil Prediksi")
            st.markdown(
                f"<h2 style='color:{warna};font-family:monospace'>"
                f"{ikon} {label}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Skor Risiko:** `{skor:.2f}%`")
            st.markdown(f"**Metode:** `{metode}`")
            st.markdown(f"**Waktu Komputasi:** `{elapsed*1000:.1f} ms`")
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Usia", f"{usia} thn")
            c2.metric("Sistolik", f"{sistolik} mmHg")
            c3.metric("Diastolik", f"{diastolik} mmHg")
            c4, c5, c6 = st.columns(3)
            c4.metric("Kolesterol", {1: 'Normal', 2: 'Di Atas Normal', 3: 'Tinggi'}[kolesterol])
            c5.metric("BMI", f"{bmi:.2f}")
            c6.metric("Berat / Tinggi", f"{berat_kg} kg / {tinggi_cm} cm")

        st.markdown("---")

        # ── DERAJAT KEANGGOTAAN
        st.markdown("### 🔢 Derajat Keanggotaan Input")
        col_maps = st.columns(5)
        var_names = ['Usia', 'Sistolik', 'Diastolik', 'Kolesterol', 'BMI']
        labels_map = {
            'Usia':       ['muda', 'paruh_baya', 'tua'],
            'Sistolik':   ['normal', 'pre_hipertensi', 'hipertensi'],
            'Diastolik':  ['normal', 'pre_hipertensi', 'hipertensi'],
            'Kolesterol': ['normal', 'di_atas_normal', 'tinggi'],
            'BMI':        ['normal', 'overweight', 'obesitas'],
        }
        for col, vname in zip(col_maps, var_names):
            with col:
                st.markdown(f"**{vname}**")
                for lbl in labels_map[vname]:
                    val = fuzzy_vals[vname][lbl]
                    bar_color = '#ef4444' if val > 0.6 else '#f59e0b' if val > 0.2 else '#22c55e'
                    st.markdown(
                        f"<small>{lbl.replace('_', ' ').title()}</small> "
                        f"<span style='color:{bar_color};font-weight:bold'>{val:.3f}</span>",
                        unsafe_allow_html=True
                    )
                    st.progress(float(val))

        st.markdown("---")

        # ── TOP RULES AKTIF
        st.markdown("### ⚡ Rule Aktif (α > 0)")
        aktif = [(no, kondisi, output, alpha)
                 for no, kondisi, output, alpha in rule_detail if alpha > 0]
        if aktif:
            import pandas as pd
            df_aktif = pd.DataFrame(aktif, columns=['Rule', 'Kondisi', 'Output', 'α (Kekuatan)'])
            df_aktif['α (Kekuatan)'] = df_aktif['α (Kekuatan)'].map('{:.4f}'.format)
            df_aktif = df_aktif.sort_values('α (Kekuatan)', ascending=False)
            st.dataframe(df_aktif, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada rule yang aktif. Coba ubah nilai input.")

    else:
        st.info("👈 Masukkan data klinis pasien di sidebar, lalu tekan **Hitung Risiko**.")
        st.markdown("""
        #### Cara Penggunaan
        1. Pilih **metode inferensi** di sidebar (Sugeno lebih cepat, Mamdani lebih detail)
        2. Masukkan **data klinis** pasien menggunakan slider
        3. Tekan tombol **🔍 Hitung Risiko**
        4. Lihat hasil prediksi, derajat keanggotaan, dan rule yang aktif
        """)


# ── TAB 2: FUNGSI KEANGGOTAAN ─────────────────────────────────────────────────
with tab_membership:
    st.markdown("### 📈 Visualisasi Fungsi Keanggotaan")
    st.markdown(
        "Garis **kuning putus-putus** menunjukkan nilai input saat ini. "
        "Kurva menggunakan kombinasi **Segitiga** dan **Trapesium**."
    )
    fig_mf = gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi)
    st.pyplot(fig_mf, use_container_width=True)
    plt.close(fig_mf)

    with st.expander("📖 Penjelasan Parameter Kurva"):
        st.markdown("""
| Variabel | Himpunan | Tipe | Parameter |
|---|---|---|---|
| **Usia** | Muda | Trapesium | (0, 0, 35, 45) |
| | Paruh Baya | Segitiga | (35, 50, 65) |
| | Tua | Trapesium | (55, 65, 100, 100) |
| **Sistolik** | Normal | Trapesium | (80, 80, 110, 125) |
| | Pre-Hipertensi | Segitiga | (115, 130, 140) |
| | Hipertensi | Trapesium | (135, 150, 200, 200) |
| **Diastolik** | Normal | Trapesium | (40, 40, 70, 80) |
| | Pre-Hipertensi | Segitiga | (75, 85, 90) |
| | Hipertensi | Trapesium | (85, 95, 120, 120) |
| **Kolesterol** | Normal | Segitiga | (0, 1, 2) |
| | Di Atas Normal | Segitiga | (1, 2, 3) |
| | Tinggi | Segitiga | (2, 3, 4) |
| **BMI** | Normal | Trapesium | (10, 10, 18.5, 25) |
| | Overweight | Segitiga | (23, 27, 30) |
| | Obesitas | Trapesium | (29, 35, 50, 50) |
        """)


# ── TAB 3: RULE BASE ──────────────────────────────────────────────────────────
with tab_rules:
    st.markdown("### 📋 15 Rule Base")
    st.markdown(
        "Tabel berikut menampilkan 15 rule utama dari total teoritis **243 kombinasi** "
        "(3⁵). Nilai α akan diisi setelah klik **Hitung Risiko**."
    )

    all_rules = [
        ("R1",  "Tua ∧ Sistolik Hipertensi",               "Tinggi"),
        ("R2",  "Kol Tinggi ∧ BMI Obesitas",               "Tinggi"),
        ("R3",  "Sistolik Normal ∧ Diastolik Normal",       "Sedang"),
        ("R4",  "Muda ∧ BMI Normal",                       "Rendah"),
        ("R5",  "Kol Normal ∧ Sistolik Normal",            "Rendah"),
        ("R6",  "Paruh Baya ∧ Sistolik Pre-Hipertensi",    "Sedang"),
        ("R7",  "Diastolik Hipertensi ∧ BMI Obesitas",     "Tinggi"),
        ("R8",  "Kol Di Atas Normal ∧ BMI Overweight",     "Sedang"),
        ("R9",  "Tua ∧ Kol Tinggi",                        "Tinggi"),
        ("R10", "Muda ∧ Diastolik Pre-Hipertensi",         "Sedang"),
        ("R11", "Sistolik Hipertensi ∧ Kol Tinggi",        "Tinggi"),
        ("R12", "Paruh Baya ∧ BMI Normal",                 "Sedang"),
        ("R13", "Diastolik Normal ∧ Kol Normal",           "Rendah"),
        ("R14", "Sistolik Pre-Hipertensi ∧ BMI Overweight","Sedang"),
        ("R15", "Tua ∧ BMI Obesitas",                      "Tinggi"),
    ]

    # Tambah α jika sudah dihitung
    if 'last_result' in st.session_state:
        _, _, _, rule_detail, _, _ = st.session_state['last_result']
        alpha_map = {r[0]: r[3] for r in rule_detail}
        data = [(no, cond, out, f"{alpha_map.get(no, 0):.4f}")
                for no, cond, out in all_rules]
        cols = ['Rule', 'Kondisi (IF ... THEN)', 'Output', 'α']
    else:
        data = all_rules
        cols = ['Rule', 'Kondisi (IF ... THEN)', 'Output']

    import pandas as pd
    df_rules = pd.DataFrame(data, columns=cols)

    def warna_output(val):
        return {'Tinggi': 'color:#ef4444;font-weight:bold',
                'Sedang': 'color:#f59e0b;font-weight:bold',
                'Rendah': 'color:#22c55e;font-weight:bold'}.get(val, '')

    st.dataframe(
        df_rules.style.applymap(warna_output, subset=['Output']),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col_m, col_s = st.columns(2)
    with col_m:
        st.markdown("#### 🔵 Defuzzifikasi Mamdani (Centroid)")
        st.latex(r"Z^* = \frac{\sum_{i=1}^{1000} z_i \cdot \mu_{\text{komposisi}}(z_i)}{\sum_{i=1}^{1000} \mu_{\text{komposisi}}(z_i)}")
    with col_s:
        st.markdown("#### 🟢 Defuzzifikasi Sugeno (Weighted Average)")
        st.latex(r"Z^* = \frac{\sum_{i=1}^{15} \alpha_i \cdot k_i}{\sum_{i=1}^{15} \alpha_i}")
        st.markdown("Singleton: Rendah = 20, Sedang = 50, Tinggi = 80")


# ── TAB 4: TENTANG ────────────────────────────────────────────────────────────
with tab_info:
    st.markdown("### ℹ️ Tentang Proyek")
    st.markdown("""
    **Tugas Besar — Dasar Kecerdasan Artificial**
    
    *Implementasi Fuzzy Logic Mamdani dan Sugeno untuk Prediksi Risiko Penyakit 
    Kardiovaskular: Perbandingan Performa dan Analisis Hasil*
    
    ---
    
    #### 👥 Kelompok Leker
    | Nama | NIM |
    |------|-----|
    | Savila Nur Fadilla | 103112400031 |
    | Dwi Okta Suryaningrum | 103112400066 |
    | Sheila Stephanie Anindya | 103112400086 |
    
    ---
    
    #### 🗃️ Dataset
    - **Cardiovascular Disease Dataset** — Kaggle (`sulianova/cardiovascular-disease-dataset`)
    - 70.000 data total → 10.000 sampel → **9.799 data bersih** (setelah filtering outlier)
    
    #### 📊 Hasil Pengujian (9.799 data)
    | Metode | Akurasi | Waktu Eksekusi |
    |--------|---------|----------------|
    | **Fuzzy Sugeno** | **65.61%** (6.429 data benar) | **~1.94 detik** |
    | Fuzzy Mamdani | 63.67% (6.239 data benar) | ~266 detik (~4 menit) |
    
    #### 🧮 Arsitektur Sistem
    - **5 variabel input**: Usia, Tekanan Darah Sistolik, Tekanan Darah Diastolik, Kolesterol, BMI
    - **15 rule base** (dari 243 kombinasi teoritis 3⁵)
    - **Fungsi keanggotaan**: Segitiga & Trapesium (*from scratch*)
    - **Inferensi**: Operator AND = min()  |  Agregasi = max()
    - **Defuzzifikasi Mamdani**: Centroid Diskrit (Integral Riemann, 1000 titik)
    - **Defuzzifikasi Sugeno**: Weighted Average (Singleton: 20/50/80)
    
    ---
    
    > *"Sistem ini dibangun murni dari nol menggunakan Python dasar, tanpa bantuan 
    > library fuzzy apapun (skfuzzy, simpful, dll)."*
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:12px'>"
        "🫀 Fuzzy Cardio Risk · Kelompok Leker · 2025</p>",
        unsafe_allow_html=True
    )