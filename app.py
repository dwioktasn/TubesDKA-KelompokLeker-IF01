import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend untuk Streamlit
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from fuzzy_engine import (
    hitung_fuzzy_usia,
    hitung_fuzzy_sistolik,
    hitung_fuzzy_diastolik,
    hitung_fuzzy_kolesterol,
    hitung_fuzzy_bmi,
    fungsi_segitiga,
    fungsi_trapesium,
    prediksi_pasien,
    jalankan_inferensi_15_rules,
    defuzzifikasi_mamdani,
    defuzzifikasi_sugeno
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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Nunito:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Mono', monospace; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); border-right: 2px solid #334155; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stMetric"] { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px 20px; }
[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 10px; background: #0f172a; }
hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def warna_risiko(label: str):
    return {
        'Berisiko Tinggi':  ('#ef4444', '🔴'),
        'Berisiko Sedang':  ('#f59e0b', '🟡'),
        'Berisiko Rendah':  ('#22c55e', '🟢'),
    }.get(label, ('#94a3b8', '⚪'))

def plot_membership(ax, x_range, curves, title, xlabel, nilai_aktual=None):
    colors = ['#38bdf8', '#fb923c', '#f87171']
    ax.set_facecolor('#0f172a')
    ax.figure.set_facecolor('#1e293b')
    for (label, y_vals), color in zip(curves, colors):
        ax.plot(x_range, y_vals, label=label, color=color, linewidth=2)
    if nilai_aktual is not None:
        ax.axvline(x=nilai_aktual, color='#facc15', linestyle='--', linewidth=1.5, label=f'Input: {nilai_aktual:.1f}')
    ax.set_title(title, fontsize=10, color='#e2e8f0', pad=6, fontfamily='monospace')
    ax.set_xlabel(xlabel, fontsize=8, color='#94a3b8')
    ax.set_ylabel('μ(x)', fontsize=8, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.25, color='#334155')
    ax.set_ylim(-0.05, 1.15)

def gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#1e293b')
    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    x_u = np.linspace(0, 100, 500)
    plot_membership(axes[0, 0], x_u, [
        ('Muda', [fungsi_trapesium(v, 0, 0, 35, 45) for v in x_u]),
        ('Paruh Baya', [fungsi_segitiga(v, 35, 50, 65) for v in x_u]),
        ('Tua', [fungsi_trapesium(v, 55, 65, 100, 100) for v in x_u]),
    ], 'Usia', 'Tahun', usia)

    x_s = np.linspace(80, 200, 500)
    plot_membership(axes[0, 1], x_s, [
        ('Normal', [fungsi_trapesium(v, 80, 80, 110, 125) for v in x_s]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 115, 130, 140) for v in x_s]),
        ('Hipertensi', [fungsi_trapesium(v, 135, 150, 200, 200) for v in x_s]),
    ], 'Sistolik', 'mmHg', sistolik)

    x_d = np.linspace(40, 120, 500)
    plot_membership(axes[0, 2], x_d, [
        ('Normal', [fungsi_trapesium(v, 40, 40, 70, 80) for v in x_d]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 75, 85, 90) for v in x_d]),
        ('Hipertensi', [fungsi_trapesium(v, 85, 95, 120, 120) for v in x_d]),
    ], 'Diastolik', 'mmHg', diastolik)

    x_k = np.linspace(0, 4, 500)
    plot_membership(axes[1, 0], x_k, [
        ('Normal', [fungsi_segitiga(v, 0, 1, 2) for v in x_k]),
        ('Di Atas Normal', [fungsi_segitiga(v, 1, 2, 3) for v in x_k]),
        ('Tinggi', [fungsi_segitiga(v, 2, 3, 4) for v in x_k]),
    ], 'Kolesterol', 'Kategori', kolesterol)

    x_b = np.linspace(10, 50, 500)
    plot_membership(axes[1, 1], x_b, [
        ('Normal', [fungsi_trapesium(v, 10, 10, 18.5, 25) for v in x_b]),
        ('Overweight', [fungsi_segitiga(v, 23, 27, 30) for v in x_b]),
        ('Obesitas', [fungsi_trapesium(v, 29, 35, 50, 50) for v in x_b]),
    ], 'BMI', 'kg/m²', bmi)

    axes[1, 2].set_facecolor('#0f172a')
    x_out = np.linspace(0, 100, 500)
    axes[1, 2].plot(x_out, [fungsi_trapesium(v, 0, 0, 20, 40) for v in x_out], color='#22c55e', label='Rendah')
    axes[1, 2].plot(x_out, [fungsi_segitiga(v, 30, 50, 70) for v in x_out], color='#f59e0b', label='Sedang')
    axes[1, 2].plot(x_out, [fungsi_trapesium(v, 60, 80, 100, 100) for v in x_out], color='#ef4444', label='Tinggi')
    axes[1, 2].set_title('Output Risiko', color='#e2e8f0', fontsize=10)
    axes[1, 2].legend(fontsize=7)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫀 Fuzzy Cardio Risk")
    metode_pilihan = st.radio("Metode Defuzzifikasi Aktif:", ['Sugeno (Cepat)', 'Mamdani (Akurat)'])
    metode_key = 'sugeno' if 'Sugeno' in metode_pilihan else 'mamdani'
    
    st.markdown("---")
    usia = st.slider("Usia", 18, 90, 45)
    sistolik = st.slider("Sistolik", 80, 200, 120)
    diastolik = st.slider("Diastolik", 40, 120, 80)
    kolesterol = st.select_slider("Kolesterol", options=[1, 2, 3], value=1, 
                                  format_func=lambda v: {1: 'Normal', 2: 'Sedang', 3: 'Tinggi'}[v])
    tinggi_cm = st.slider("Tinggi (cm)", 140, 210, 165)
    berat_kg  = st.slider("Berat (kg)", 40, 150, 70)
    bmi = round(berat_kg / (tinggi_cm / 100) ** 2, 2)
    st.metric("BMI", f"{bmi}")
    hitung = st.button("🔍 Hitung Risiko", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

tab_prediksi, tab_membership, tab_rules = st.tabs(["📊 Prediksi & Grafik", "📈 Fungsi Keanggotaan", "📋 Rule Base"])

with tab_prediksi:
    if hitung or 'last_result' in st.session_state:
        if hitung:
            # Hitung satu kali untuk metode yang dipilih
            skor, label, fuzzy_vals, rule_detail = prediksi_pasien(usia, sistolik, diastolik, kolesterol, bmi, metode=metode_key)
            st.session_state['last_result'] = (skor, label, fuzzy_vals, rule_detail, metode_key)
        else:
            skor, label, fuzzy_vals, rule_detail, metode_key = st.session_state['last_result']

        warna, ikon = warna_risiko(label)
        
        # --- Bagian Metric ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(f"Skor {metode_key.title()}", f"{skor:.2f}%", label)
            st.progress(max(0.0, min(skor / 100, 1.0)))
        with col2:
            st.markdown(f"""<div style="padding:20px;border-radius:15px;background:{warna}22;border:2px solid {warna};">
                        <h2 style="color:{warna};margin:0;">{ikon} {label}</h2></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- Bagian Grafik Berdampingan ---
        st.subheader("📈 Visualisasi Perbandingan Defuzzifikasi")
        col_mam, col_sug = st.columns(2)

        # Hitung data Mamdani & Sugeno secara manual untuk grafik
        rules_inf = jalankan_inferensi_15_rules(fuzzy_vals['Usia'], fuzzy_vals['Sistolik'], 
                                               fuzzy_vals['Diastolik'], fuzzy_vals['Kolesterol'], fuzzy_vals['BMI'])
        
        with col_mam:
            st.write("**Mamdani (Area Agregasi)**")
            x_s = np.linspace(0, 100, 500)
            agregasi = []
            for x in x_s:
                m_r, m_s, m_t = fungsi_trapesium(x,0,0,20,40), fungsi_segitiga(x,30,50,70), fungsi_trapesium(x,60,80,100,100)
                val = 0
                for kat, alpha in rules_inf:
                    mu = min(alpha, m_r if kat=='rendah' else m_s if kat=='sedang' else m_t)
                    val = max(val, mu)
                agregasi.append(val)
            
            fig_m, ax_m = plt.subplots(figsize=(6,4))
            ax_m.set_facecolor('#0f172a')
            fig_m.patch.set_facecolor('#1e293b')
            ax_m.plot(x_s, agregasi, color='cyan', linewidth=2)
            ax_m.fill_between(x_s, agregasi, color='cyan', alpha=0.3)
            # Ambil skor mamdani murni untuk garis
            s_mamdani = defuzzifikasi_mamdani(rules_inf)
            ax_m.axvline(s_mamdani, color='red', linestyle='--', label=f'Centroid: {s_mamdani:.2f}')
            ax_m.tick_params(colors='#94a3b8')
            ax_m.legend()
            st.pyplot(fig_m)

        with col_sug:
            st.write("**Sugeno (Kontribusi Rule)**")
            df_sug = pd.DataFrame(rule_detail, columns=['Rule', 'Kondisi', 'Output', 'Alpha'])
            bobot_map = {'rendah': 20, 'sedang': 50, 'tinggi': 80}
            df_sug['Kontribusi'] = df_sug.apply(lambda r: r['Alpha'] * bobot_map[r['Output'].lower()], axis=1)
            
            fig_s, ax_s = plt.subplots(figsize=(6,4))
            ax_s.set_facecolor('#0f172a')
            fig_s.patch.set_facecolor('#1e293b')
            ax_s.bar(df_sug['Rule'], df_sug['Kontribusi'], color='orange')
            ax_s.set_title("Alpha x Singleton", color='white', fontsize=10)
            ax_s.tick_params(colors='#94a3b8', labelsize=8)
            st.pyplot(fig_s)

        st.markdown("---")

        # --- Bagian Dijejerin (Perbandingan Skor Akhir) ---
        st.subheader("📊 Perbandingan Skor Akhir")
        s_mamdani = defuzzifikasi_mamdani(rules_inf)
        s_sugeno = defuzzifikasi_sugeno(rules_inf)

        fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
        ax_comp.set_facecolor('#0f172a')
        fig_comp.patch.set_facecolor('#1e293b')
        
        metodes = ['Mamdani', 'Sugeno']
        skors = [s_mamdani, s_sugeno]
        bars = ax_comp.barh(metodes, skors, color=['cyan', 'orange'])
        
        ax_comp.set_xlim(0, 100)
        ax_comp.tick_params(colors='#94a3b8')
        for bar in bars:
            width = bar.get_width()
            ax_comp.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                         color='white', va='center', fontweight='bold')
        
        st.pyplot(fig_comp)

        # --- Derajat Keanggotaan ---
        st.markdown("### 🔢 Derajat Keanggotaan Input")
        cols = st.columns(5)
        for col, vname in zip(cols, fuzzy_vals.keys()):
            with col:
                st.write(f"**{vname}**")
                for k, v in fuzzy_vals[vname].items():
                    st.caption(f"{k}: {v:.3f}")
                    st.progress(float(v))

with tab_membership:
    st.pyplot(gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi))

with tab_rules:
    df_rules = pd.DataFrame(rule_detail, columns=['Rule', 'Kondisi', 'Output', 'Alpha'])
    st.dataframe(df_rules, use_container_width=True)