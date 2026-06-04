import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #0a0f1e;
    color: #e2e8f0;
}
h1, h2, h3, h4, .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1527 0%, #111827 100%) !important;
    border-right: 1.5px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { margin-top: 4px; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1f3d 0%, #162035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 22px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #3b82f6; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1527;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.85rem;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #1d4ed8) !important;
    color: #f1f5f9 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1527;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; margin: 20px 0; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #1d4ed8, #06b6d4) !important;
    border-radius: 99px;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #0891b2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(29,78,216,0.35);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(29,78,216,0.5);
}

/* ── Caption ── */
.stCaption { color: #64748b !important; font-size: 0.75rem; }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Risk box ── */
.risk-box {
    border-radius: 16px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
}

/* ── Membership card ── */
.mem-card {
    background: #0d1527;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BG_DARK   = '#070e1f'
BG_CARD   = '#0d1527'
BG_LIGHT  = '#162035'
BORDER    = '#1e3a5f'
TEXT_MAIN = '#e2e8f0'
TEXT_DIM  = '#64748b'
TEXT_MID  = '#94a3b8'

C_BLUE    = '#38bdf8'
C_ORANGE  = '#fb923c'
C_RED     = '#f87171'
C_GREEN   = '#4ade80'
C_YELLOW  = '#facc15'
C_PURPLE  = '#a78bfa'
C_CYAN    = '#06b6d4'

PLT_COLORS = [C_BLUE, C_ORANGE, C_RED]

def apply_ax_style(ax, title='', xlabel='', ylabel='μ(x)'):
    ax.set_facecolor(BG_DARK)
    ax.set_title(title, fontsize=10, color=TEXT_MAIN, pad=8, fontfamily='monospace')
    ax.set_xlabel(xlabel, fontsize=8, color=TEXT_MID)
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT_MID)
    ax.tick_params(colors=TEXT_MID, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, linestyle='--', alpha=0.2, color=BORDER)
    ax.set_ylim(-0.05, 1.15)

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
# HELPER: PLOT MEMBERSHIP SINGLE VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_membership(ax, x_range, curves, title, xlabel, nilai_aktual=None):
    ax.set_facecolor(BG_DARK)
    ax.figure.set_facecolor(BG_CARD)
    for (label, y_vals), color in zip(curves, PLT_COLORS):
        ax.fill_between(x_range, y_vals, alpha=0.08, color=color)
        ax.plot(x_range, y_vals, label=label, color=color, linewidth=2.2)
    if nilai_aktual is not None:
        ax.axvline(x=nilai_aktual, color=C_YELLOW, linestyle='--', linewidth=1.8,
                   label=f'Input: {nilai_aktual:.1f}', zorder=5)
    apply_ax_style(ax, title, xlabel)
    ax.legend(fontsize=7, facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=TEXT_MAIN,
              loc='upper right', framealpha=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GAMBAR 6 PANEL MEMBERSHIP
# ─────────────────────────────────────────────────────────────────────────────

def gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(BG_CARD)
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # ── Usia ──
    x_u = np.linspace(0, 100, 500)
    plot_membership(axes[0][0], x_u, [
        ('Muda',      [fungsi_trapesium(v, 0, 0, 35, 45)    for v in x_u]),
        ('Paruh Baya',[fungsi_segitiga(v, 35, 50, 65)       for v in x_u]),
        ('Tua',       [fungsi_trapesium(v, 55, 65, 100, 100) for v in x_u]),
    ], 'Usia', 'Tahun', usia)

    # ── Sistolik ──
    x_s = np.linspace(80, 200, 500)
    plot_membership(axes[0][1], x_s, [
        ('Normal',        [fungsi_trapesium(v, 80, 80, 110, 125)  for v in x_s]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 115, 130, 140)     for v in x_s]),
        ('Hipertensi',    [fungsi_trapesium(v, 135, 150, 200, 200) for v in x_s]),
    ], 'Sistolik', 'mmHg', sistolik)

    # ── Diastolik ──
    x_d = np.linspace(40, 120, 500)
    plot_membership(axes[0][2], x_d, [
        ('Normal',        [fungsi_trapesium(v, 40, 40, 70, 80)  for v in x_d]),
        ('Pre-Hipertensi', [fungsi_segitiga(v, 75, 85, 90)      for v in x_d]),
        ('Hipertensi',    [fungsi_trapesium(v, 85, 95, 120, 120) for v in x_d]),
    ], 'Diastolik', 'mmHg', diastolik)

    # ── Kolesterol ──
    x_k = np.linspace(0, 4, 500)
    plot_membership(axes[1][0], x_k, [
        ('Normal',        [fungsi_segitiga(v, 0, 1, 2) for v in x_k]),
        ('Di Atas Normal',[fungsi_segitiga(v, 1, 2, 3) for v in x_k]),
        ('Tinggi',        [fungsi_segitiga(v, 2, 3, 4) for v in x_k]),
    ], 'Kolesterol', 'Kategori', kolesterol)

    # ── BMI ──
    x_b = np.linspace(10, 50, 500)
    plot_membership(axes[1][1], x_b, [
        ('Normal',    [fungsi_trapesium(v, 10, 10, 18.5, 25) for v in x_b]),
        ('Overweight',[fungsi_segitiga(v, 23, 27, 30)        for v in x_b]),
        ('Obesitas',  [fungsi_trapesium(v, 29, 35, 50, 50)   for v in x_b]),
    ], 'BMI', 'kg/m²', bmi)

    # ── Output Risiko ──
    ax_out = axes[1][2]
    ax_out.set_facecolor(BG_DARK)
    fig.set_facecolor(BG_CARD)
    x_out = np.linspace(0, 100, 500)
    kurva_output = [
        ('Rendah', [fungsi_trapesium(v, 0, 0, 20, 40)   for v in x_out], '#22c55e'),
        ('Sedang', [fungsi_segitiga(v, 30, 50, 70)       for v in x_out], '#f59e0b'),
        ('Tinggi', [fungsi_trapesium(v, 60, 80, 100, 100) for v in x_out], '#ef4444'),
    ]
    for lbl, yv, col in kurva_output:
        ax_out.fill_between(x_out, yv, alpha=0.1, color=col)
        ax_out.plot(x_out, yv, label=lbl, color=col, linewidth=2.2)
    apply_ax_style(ax_out, 'Output Risiko', 'Skor (%)')
    ax_out.legend(fontsize=7, facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=TEXT_MAIN,
                  loc='upper right', framealpha=0.9)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GRAFIK MAMDANI AREA AGREGASI
# ─────────────────────────────────────────────────────────────────────────────

def plot_mamdani(rules_inf, skor_mamdani):
    x_s = np.linspace(0, 100, 500)
    y_r = np.array([fungsi_trapesium(x, 0, 0, 20, 40)    for x in x_s])
    y_m = np.array([fungsi_segitiga(x, 30, 50, 70)       for x in x_s])
    y_t = np.array([fungsi_trapesium(x, 60, 80, 100, 100) for x in x_s])

    agregasi = np.zeros(len(x_s))
    for kat, alpha in rules_inf:
        if kat == 'rendah':
            agregasi = np.maximum(agregasi, np.minimum(alpha, y_r))
        elif kat == 'sedang':
            agregasi = np.maximum(agregasi, np.minimum(alpha, y_m))
        else:
            agregasi = np.maximum(agregasi, np.minimum(alpha, y_t))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_facecolor(BG_DARK)
    fig.patch.set_facecolor(BG_CARD)

    # Area per kelas (under-curve backdrop)
    ax.fill_between(x_s, y_r, alpha=0.07, color='#22c55e')
    ax.fill_between(x_s, y_m, alpha=0.07, color='#f59e0b')
    ax.fill_between(x_s, y_t, alpha=0.07, color='#ef4444')

    ax.plot(x_s, y_r, color='#22c55e', linewidth=1, linestyle=':', alpha=0.5, label='MF Rendah')
    ax.plot(x_s, y_m, color='#f59e0b', linewidth=1, linestyle=':', alpha=0.5, label='MF Sedang')
    ax.plot(x_s, y_t, color='#ef4444', linewidth=1, linestyle=':', alpha=0.5, label='MF Tinggi')

    # Agregasi utama
    ax.fill_between(x_s, agregasi, alpha=0.35, color=C_CYAN)
    ax.plot(x_s, agregasi, color=C_CYAN, linewidth=2.5, label='Agregasi')

    ax.axvline(skor_mamdani, color=C_YELLOW, linestyle='--', linewidth=2,
               label=f'Centroid: {skor_mamdani:.2f}%', zorder=6)

    apply_ax_style(ax, 'Mamdani — Area Agregasi', 'Skor Risiko (%)')
    ax.legend(fontsize=7.5, facecolor=BG_LIGHT, edgecolor=BORDER, labelcolor=TEXT_MAIN,
              framealpha=0.9, loc='upper left')
    fig.tight_layout(pad=1.2)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GRAFIK SUGENO BAR KONTRIBUSI
# ─────────────────────────────────────────────────────────────────────────────

def plot_sugeno(rule_detail):
    bobot_map = {'rendah': 20, 'sedang': 50, 'tinggi': 80}
    color_map  = {'rendah': '#22c55e', 'sedang': '#f59e0b', 'tinggi': '#ef4444'}

    df = pd.DataFrame(rule_detail, columns=['Rule', 'Kondisi', 'Output', 'Alpha'])
    df['Kontribusi'] = df.apply(
        lambda r: r['Alpha'] * bobot_map.get(r['Output'].lower(), 50), axis=1)
    bar_colors = [color_map.get(r.lower(), '#94a3b8') for r in df['Output']]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_facecolor(BG_DARK)
    fig.patch.set_facecolor(BG_CARD)

    bars = ax.bar(df['Rule'], df['Kontribusi'], color=bar_colors,
                  width=0.6, edgecolor=BORDER, linewidth=0.5, zorder=3)

    for bar, val in zip(bars, df['Kontribusi']):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', color=TEXT_MAIN,
                    fontsize=7, fontweight='bold')

    apply_ax_style(ax, 'Sugeno — Kontribusi Rule (α × Singleton)', 'Rule', ylabel='Kontribusi')
    ax.set_ylim(0, max(df['Kontribusi'].max() * 1.2, 5))
    ax.tick_params(axis='x', labelrotation=0, labelsize=7)

    patches = [mpatches.Patch(color=color_map[k], label=k.title()) for k in color_map]
    ax.legend(handles=patches, fontsize=7.5, facecolor=BG_LIGHT, edgecolor=BORDER,
              labelcolor=TEXT_MAIN, framealpha=0.9, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color=BORDER, zorder=0)
    fig.tight_layout(pad=1.2)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GRAFIK PERBANDINGAN SKOR
# ─────────────────────────────────────────────────────────────────────────────

def plot_perbandingan(s_mamdani, s_sugeno):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_facecolor(BG_DARK)
    fig.patch.set_facecolor(BG_CARD)

    metodes = ['Mamdani', 'Sugeno']
    skors   = [s_mamdani, s_sugeno]
    colors  = [C_CYAN, C_ORANGE]

    bars = ax.barh(metodes, skors, color=colors, height=0.42,
                   edgecolor=BORDER, linewidth=0.5)

    # Background track
    ax.barh(metodes, [100, 100], color=BG_LIGHT, height=0.42,
            zorder=0, edgecolor=BORDER, linewidth=0.4)

    for bar, color in zip(bars, colors):
        w = bar.get_width()
        ax.barh(bar.get_y() + bar.get_height() / 2, w,
                height=0.42, color=color, left=0, zorder=2)
        ax.text(min(w + 2, 98), bar.get_y() + bar.get_height() / 2,
                f'{w:.2f}%', color=TEXT_MAIN, va='center',
                fontweight='bold', fontsize=11, zorder=3)

    ax.set_xlim(0, 108)
    ax.set_xlabel('Skor Risiko (%)', fontsize=9, color=TEXT_MID)
    apply_ax_style(ax, 'Perbandingan Skor Defuzzifikasi', 'Skor (%)', ylabel='')
    ax.set_ylim(-0.6, 1.6)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    fig.tight_layout(pad=1.2)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px;'>
      <span style='font-size:2.4rem;'>🫀</span>
      <h2 style='font-family:Space Mono,monospace;margin:4px 0 0;font-size:1rem;color:#38bdf8;letter-spacing:1px;'>
        FUZZY CARDIO<br>RISK SYSTEM
      </h2>
      <p style='color:#475569;font-size:0.72rem;margin:4px 0 0;'>Kelompok Leker</p>
    </div>
    <hr style='border-color:#1e3a5f;margin:10px 0 16px;'/>
    """, unsafe_allow_html=True)

    metode_pilihan = st.radio(
        "Metode Defuzzifikasi",
        ['Sugeno (Cepat)', 'Mamdani (Akurat)'],
        help="Pilih metode defuzzifikasi yang digunakan untuk output utama"
    )
    metode_key = 'sugeno' if 'Sugeno' in metode_pilihan else 'mamdani'

    st.markdown("<div class='section-header' style='margin-top:16px;'>Parameter Input</div>", unsafe_allow_html=True)

    usia      = st.slider("🎂 Usia (tahun)", 18, 90, 45)
    sistolik  = st.slider("📈 Sistolik (mmHg)", 80, 200, 120)
    diastolik = st.slider("📉 Diastolik (mmHg)", 40, 120, 80)
    kolesterol = st.select_slider(
        "🧪 Kolesterol",
        options=[1, 2, 3], value=1,
        format_func=lambda v: {1: '1 — Normal', 2: '2 — Sedang', 3: '3 — Tinggi'}[v]
    )
    tinggi_cm = st.slider("📏 Tinggi (cm)", 140, 210, 165)
    berat_kg  = st.slider("⚖️ Berat (kg)", 40, 150, 70)
    bmi = round(berat_kg / (tinggi_cm / 100) ** 2, 2)

    bmi_label = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obesitas"
    bmi_color = "#38bdf8" if bmi < 18.5 else "#4ade80" if bmi < 25 else "#f59e0b" if bmi < 30 else "#ef4444"
    st.markdown(f"""
    <div style='background:#0d1527;border:1px solid #1e3a5f;border-radius:10px;padding:12px 16px;margin:8px 0;'>
      <span style='color:#94a3b8;font-size:0.78rem;'>BMI Terhitung</span><br>
      <span style='font-size:1.6rem;font-weight:900;color:{bmi_color};font-family:Space Mono,monospace;'>{bmi}</span>
      <span style='color:{bmi_color};font-size:0.8rem;margin-left:8px;'>({bmi_label})</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    hitung = st.button("🔍 Hitung Risiko", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:28px 0 12px;'>
  <h1 style='font-family:Space Mono,monospace;font-size:1.55rem;margin:0;
             background:linear-gradient(90deg,#38bdf8,#818cf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;'>
    🫀 Fuzzy Cardio Risk Prediction
  </h1>
  <p style='color:#64748b;margin:6px 0 0;font-size:0.88rem;'>
    Sistem prediksi risiko kardiovaskular berbasis Logika Fuzzy Mamdani & Sugeno
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_prediksi, tab_membership, tab_rules = st.tabs([
    "📊  Prediksi & Grafik",
    "📈  Fungsi Keanggotaan",
    "📋  Rule Base"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI & GRAFIK
# ══════════════════════════════════════════════════════════

with tab_prediksi:
    if hitung or 'last_result' in st.session_state:
        if hitung:
            skor, label, fuzzy_vals, rule_detail = prediksi_pasien(
                usia, sistolik, diastolik, kolesterol, bmi, metode=metode_key)
            st.session_state['last_result'] = (skor, label, fuzzy_vals, rule_detail, metode_key)
        else:
            skor, label, fuzzy_vals, rule_detail, metode_key = st.session_state['last_result']

        warna, ikon = warna_risiko(label)

        # ── Risk header ──────────────────────────────────────────────────────
        col_skor, col_label = st.columns([1, 2])
        with col_skor:
            st.metric(
                label=f"Skor ({metode_key.title()})",
                value=f"{skor:.2f}%",
                delta=label
            )
            st.progress(max(0.0, min(skor / 100, 1.0)))

        with col_label:
            st.markdown(f"""
            <div class='risk-box' style='background:{warna}18;border:2px solid {warna};'>
              <span style='font-size:2.8rem;'>{ikon}</span>
              <div>
                <div style='color:{warna};font-family:Space Mono,monospace;
                            font-size:1.5rem;font-weight:700;margin:0;'>
                  {label}
                </div>
                <div style='color:#64748b;font-size:0.82rem;margin-top:4px;'>
                  Metode: {metode_key.title()} · BMI: {bmi} ({
                    "Normal" if 18.5 <= bmi < 25 else
                    "Overweight" if 25 <= bmi < 30 else
                    "Obesitas" if bmi >= 30 else "Underweight"})
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Hitung rules inferensi ────────────────────────────────────────────
        rules_inf = jalankan_inferensi_15_rules(
            fuzzy_vals['Usia'], fuzzy_vals['Sistolik'],
            fuzzy_vals['Diastolik'], fuzzy_vals['Kolesterol'], fuzzy_vals['BMI']
        )
        s_mamdani = defuzzifikasi_mamdani(rules_inf)
        s_sugeno  = defuzzifikasi_sugeno(rules_inf)

        # ── Visualisasi Perbandingan Defuzzifikasi ────────────────────────────
        st.markdown("<div class='section-header'>Visualisasi Perbandingan Defuzzifikasi</div>", unsafe_allow_html=True)

        col_m, col_s = st.columns(2, gap="medium")
        with col_m:
            st.markdown("<p style='color:#64748b;font-size:0.8rem;margin-bottom:6px;'>Mamdani — Area agregasi hasil inferensi</p>", unsafe_allow_html=True)
            st.pyplot(plot_mamdani(rules_inf, s_mamdani))

        with col_s:
            st.markdown("<p style='color:#64748b;font-size:0.8rem;margin-bottom:6px;'>Sugeno — Kontribusi tiap rule (α × singleton)</p>", unsafe_allow_html=True)
            st.pyplot(plot_sugeno(rule_detail))

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Perbandingan Skor Akhir ───────────────────────────────────────────
        st.markdown("<div class='section-header'>Perbandingan Skor Akhir</div>", unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Skor Mamdani", f"{s_mamdani:.2f}%")
        with m_col2:
            st.metric("Skor Sugeno", f"{s_sugeno:.2f}%")
        with m_col3:
            selisih = abs(s_mamdani - s_sugeno)
            st.metric("Selisih", f"{selisih:.2f}%",
                      delta="Konsisten ✓" if selisih < 5 else "Perlu Review ⚠")

        st.pyplot(plot_perbandingan(s_mamdani, s_sugeno))

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Derajat Keanggotaan Input ─────────────────────────────────────────
        st.markdown("<div class='section-header'>Derajat Keanggotaan Input</div>", unsafe_allow_html=True)

        cols_mem = st.columns(5)
        var_icons = {'Usia': '🎂', 'Sistolik': '📈', 'Diastolik': '📉', 'Kolesterol': '🧪', 'BMI': '⚖️'}
        for col, vname in zip(cols_mem, fuzzy_vals.keys()):
            with col:
                st.markdown(f"""
                <div style='background:#0d1527;border:1px solid #1e3a5f;border-radius:10px;
                            padding:12px;margin-bottom:8px;'>
                  <div style='font-size:0.75rem;color:#94a3b8;margin-bottom:6px;
                              font-family:Space Mono,monospace;'>
                    {var_icons.get(vname, '')} {vname}
                  </div>
                """, unsafe_allow_html=True)
                for k, v in fuzzy_vals[vname].items():
                    bar_w = int(v * 100)
                    st.markdown(f"""
                    <div style='margin-bottom:6px;'>
                      <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#64748b;'>
                        <span>{k}</span><span style='color:#e2e8f0;font-weight:700;'>{v:.3f}</span>
                      </div>
                      <div style='background:#162035;border-radius:99px;height:5px;margin-top:3px;'>
                        <div style='background:linear-gradient(90deg,#1d4ed8,#06b6d4);
                                    width:{bar_w}%;height:5px;border-radius:99px;'></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder saat belum dihitung
        st.markdown("""
        <div style='text-align:center;padding:80px 40px;background:#0d1527;
                    border:1px dashed #1e3a5f;border-radius:16px;'>
          <div style='font-size:4rem;margin-bottom:16px;'>🫀</div>
          <h3 style='font-family:Space Mono,monospace;color:#38bdf8;'>Belum Ada Hasil</h3>
          <p style='color:#475569;font-size:0.9rem;'>
            Atur parameter di sidebar, lalu tekan <strong style='color:#e2e8f0;'>🔍 Hitung Risiko</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — FUNGSI KEANGGOTAAN
# ══════════════════════════════════════════════════════════

with tab_membership:
    st.markdown("<div class='section-header'>Visualisasi Kurva Keanggotaan — Semua Variabel</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.82rem;margin-bottom:16px;'>"
        "Garis kuning putus-putus menunjukkan nilai input aktual saat ini.</p>",
        unsafe_allow_html=True
    )
    fig_mem = gambar_semua_membership(usia, sistolik, diastolik, kolesterol, bmi)
    st.pyplot(fig_mem, use_container_width=True)

    # Info singkat per variabel
    with st.expander("ℹ️ Keterangan Variabel & Range"):
        tabel_info = pd.DataFrame([
            ["Usia",       "Tahun",  "18 – 90",   "Muda (≤45), Paruh Baya (35–65), Tua (≥55)"],
            ["Sistolik",   "mmHg",   "80 – 200",  "Normal (≤125), Pre-HTN (115–140), HTN (≥135)"],
            ["Diastolik",  "mmHg",   "40 – 120",  "Normal (≤80), Pre-HTN (75–90), HTN (≥85)"],
            ["Kolesterol", "Kategori","1 – 3",    "Normal (1), Di Atas Normal (2), Tinggi (3)"],
            ["BMI",        "kg/m²",  "10 – 50",   "Normal (≤25), Overweight (23–30), Obesitas (≥29)"],
        ], columns=["Variabel", "Satuan", "Range Input", "Kategori Fuzzy"])
        st.dataframe(tabel_info, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — RULE BASE
# ══════════════════════════════════════════════════════════

with tab_rules:
    st.markdown("<div class='section-header'>Rule Base — 15 Rules Fuzzy</div>", unsafe_allow_html=True)

    if 'last_result' in st.session_state:
        _, _, _, rule_detail, _ = st.session_state['last_result']
        df_rules = pd.DataFrame(rule_detail, columns=['Rule', 'Kondisi', 'Output', 'Alpha'])

        # Tambah kolom highlight
        def tag_output(val):
            colors = {'rendah': '#22c55e', 'sedang': '#f59e0b', 'tinggi': '#ef4444'}
            c = colors.get(val.lower(), '#94a3b8')
            return f'<span style="color:{c};font-weight:700;">{val.title()}</span>'

        df_display = df_rules.copy()
        df_display['Alpha'] = df_display['Alpha'].map(lambda v: f"{v:.4f}")

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rule":    st.column_config.NumberColumn("No. Rule", width="small"),
                "Kondisi": st.column_config.TextColumn("Kondisi IF-THEN", width="large"),
                "Output":  st.column_config.TextColumn("Output Kelas", width="medium"),
                "Alpha":   st.column_config.TextColumn("Derajat Aktivasi (α)", width="medium"),
            }
        )

        # Ringkasan rule aktif
        aktif = df_rules[df_rules['Alpha'].astype(float) > 0]
        st.markdown(f"""
        <div style='background:#0d1527;border:1px solid #1e3a5f;border-radius:10px;
                    padding:16px 20px;margin-top:16px;'>
          <span style='color:#94a3b8;font-size:0.82rem;'>
            ✅ <strong style='color:#e2e8f0;'>{len(aktif)}</strong> dari
            <strong style='color:#e2e8f0;'>{len(df_rules)}</strong> rule aktif (α > 0)
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Tekan **🔍 Hitung Risiko** di sidebar untuk melihat detail rule yang aktif.")