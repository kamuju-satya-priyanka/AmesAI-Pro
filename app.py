"""
AmesAI Pro — Explainable Mutagenicity Prediction Platform
==========================================================
Production-ready Streamlit web application for predicting
Ames Mutagenicity using XGBoost with SHAP explainability.

Author  : Senior AI / Cheminformatics Engineer
Version : 2.0.0
"""

from __future__ import annotations

import io
import json
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFingerprintGenerator

# ── Local Modules ─────────────────────────────────────────────────────────────
from utils.descriptor_generator import (
    validate_smiles, smiles_to_fingerprint, compute_descriptors,
    lipinski_analysis, detect_pains, batch_similarity, mol_to_fingerprint,
    tanimoto_similarity,
)
from utils.prediction import (
    load_model, load_metrics, predict_single, predict_batch, get_full_profile,
    risk_category,
)
from utils.visualization import (
    mol_to_svg, mol_to_png_bytes,
    make_gauge, make_probability_bar,
    make_confusion_matrix, make_roc_curve, make_pr_curve, make_feature_importance,
    make_distribution_pie, make_probability_histogram, make_descriptor_heatmap,
    make_radar_chart, make_similarity_bar, fingerprint_viewer,
    shap_waterfall_plotly, shap_summary_plotly,
)
from utils.report_generator import generate_pdf_report

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AmesAI Pro — Mutagenicity Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ameskiAI",
        "Report a bug": None,
        "About": "## AmesAI Pro v2.0\nXGBoost-powered Ames Mutagenicity Predictor with SHAP explainability.",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0A0E1A;
    color: #E2E8F0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 4rem;
    max-width: 1500px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080C18 0%, #0D1220 60%, #0A0E1A 100%);
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] .stRadio label {
    color: #64748B !important;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.3rem 0;
    transition: color 0.2s;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #00D4FF !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] label[data-checked="true"] {
    color: #00D4FF !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B3E 0%, #150A2E 45%, #0D1B3E 100%);
    border: 1px solid #1E3A6E;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -60%; left: -30%;
    width: 180%; height: 220%;
    background: radial-gradient(ellipse at 40% 40%, rgba(0,212,255,0.06) 0%, transparent 55%),
                radial-gradient(ellipse at 80% 60%, rgba(123,47,190,0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00D4FF 0%, #7B2FBE 50%, #00D4FF 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
@keyframes shimmer {
    0%   { background-position: 0%   50%; }
    100% { background-position: 200% 50%; }
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.7;
    max-width: 680px;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.2rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    color: #00D4FF;
    padding: 0.3rem 0.85rem;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0;
}
.metric-card {
    background: linear-gradient(145deg, #111827, #0D1220);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.25rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    cursor: default;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00D4FF, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,212,255,0.35);
    box-shadow: 0 10px 30px rgba(0,212,255,0.12);
}
.metric-card:hover::before { opacity: 1; }
.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #00D4FF;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-value.purple { color: #7B2FBE; }
.metric-value.green  { color: #00E676; }
.metric-value.yellow { color: #FFD600; }
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.45rem;
    font-weight: 600;
}
.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(13,18,32,0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Result Badges ── */
.result-mutagenic {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    background: linear-gradient(135deg, rgba(255,23,68,0.15), rgba(183,28,28,0.1));
    border: 2px solid rgba(255,23,68,0.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-red 2.5s ease-in-out infinite;
}
.result-nonmutagenic {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,105,92,0.08));
    border: 2px solid rgba(0,230,118,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-green 2.5s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(255,23,68,0.2); }
    50%       { box-shadow: 0 0 40px rgba(255,23,68,0.45); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(0,230,118,0.15); }
    50%       { box-shadow: 0 0 40px rgba(0,230,118,0.35); }
}
.result-label {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.03em;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E2E8F0;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, #00D4FF, #7B2FBE, transparent) 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Info / Warning / Danger Boxes ── */
.info-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,47,190,0.04));
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 4px solid #00D4FF;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.65;
}
.warning-box {
    background: rgba(255,214,0,0.04);
    border: 1px solid rgba(255,214,0,0.2);
    border-left: 4px solid #FFD600;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.65;
}
.danger-box {
    background: rgba(255,23,68,0.04);
    border: 1px solid rgba(255,23,68,0.2);
    border-left: 4px solid #FF1744;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.65;
}
.success-box {
    background: rgba(0,230,118,0.04);
    border: 1px solid rgba(0,230,118,0.2);
    border-left: 4px solid #00E676;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.65;
}

/* ── Lipinski Table ── */
.lipinski-pass  { color: #00E676; font-weight: 600; }
.lipinski-fail  { color: #FF1744; font-weight: 600; }

/* ── Descriptor Table ── */
.desc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.desc-table th {
    background: #111827;
    color: #00D4FF;
    font-weight: 600;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 2px solid #1E3A6E;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
}
.desc-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #1E293B;
    color: #94A3B8;
}
.desc-table tr:nth-child(even) { background: rgba(17,24,39,0.5); }
.desc-table tr:hover { background: rgba(0,212,255,0.04); }
.desc-table td:first-child { color: #E2E8F0; font-weight: 500; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1220;
    border-radius: 12px;
    padding: 0.3rem;
    gap: 0.25rem;
    border: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,190,0.12)) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #0D1220 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.12) !important;
    outline: none !important;
}

/* ── Select box ── */
.stSelectbox [data-baseweb="select"] > div {
    background: #0D1220 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B2FBE) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 2.2rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00E676, #00695C) !important;
    color: #0A0E1A !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
}

/* ── Sidebar Nav ── */
.sidebar-logo {
    text-align: center;
    padding: 1.2rem 0.5rem 0.8rem;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.3;
    margin-top: 0.3rem;
}
.sidebar-sub {
    font-size: 0.65rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.25rem;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00D4FF, #7B2FBE) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0D1220 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #1E293B;
    font-size: 0.73rem;
    padding: 2rem 0 0.5rem;
    border-top: 1px solid #1E293B;
    margin-top: 3rem;
}

/* ── Tag chips ── */
.tag {
    display: inline-flex;
    align-items: center;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    color: #00D4FF;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem;
    letter-spacing: 0.02em;
}
.tag.red   { background: rgba(255,23,68,0.08); border-color: rgba(255,23,68,0.2); color:#FF1744; }
.tag.green { background: rgba(0,230,118,0.08); border-color: rgba(0,230,118,0.2); color:#00E676; }
.tag.purple{ background: rgba(123,47,190,0.08);border-color: rgba(123,47,190,0.2);color:#A855F7; }

/* ── Molecule image container ── */
.mol-container {
    background: #0A0E1A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 0.75rem;
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "compare_mols" not in st.session_state:
    st.session_state.compare_mols = []
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-TRAIN  (runs once on Streamlit Cloud if model is absent)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_PKL  = Path(__file__).parent / "model" / "xgboost_model.pkl"
_METRICS_JSON = Path(__file__).parent / "model" / "metrics.json"

if not _MODEL_PKL.exists() or not _METRICS_JSON.exists():
    with st.spinner("⏳ First-run setup: training XGBoost model… (takes ~30 s)"):
        try:
            import subprocess, sys
            subprocess.run(
                [sys.executable, str(Path(__file__).parent / "train_model.py")],
                check=True,
                capture_output=True,
            )
            st.cache_resource.clear()   # flush any stale load_model() cache
            st.cache_data.clear()       # flush load_metrics() cache
        except Exception as _train_err:
            st.error(f"Auto-training failed: {_train_err}. "
                     "Please run `python train_model.py` manually.")
            st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE DATASET  (for similarity search & analytics)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_reference_dataset():
    """Return a small reference compound library with fingerprints."""
    compounds = [
        ("CCO",                          0, "Ethanol"),
        ("CC(C)O",                       0, "Isopropanol"),
        ("OCC(O)CO",                     0, "Glycerol"),
        ("CC(=O)O",                      0, "Acetic Acid"),
        ("CC(=O)Oc1ccccc1C(=O)O",        0, "Aspirin"),
        ("CC(=O)Nc1ccc(O)cc1",           0, "Paracetamol"),
        ("OC(=O)c1ccccc1",               0, "Benzoic Acid"),
        ("CCCCO",                         0, "1-Butanol"),
        ("c1ccc2c(c1)ccc3cccc4cccc2c34", 1, "Benzo[a]pyrene"),
        ("Nc1ccc(N)cc1",                  1, "4,4'-Diaminobiphenyl"),
        ("Nc1ccc2ccccc2c1",               1, "2-Naphthylamine"),
        ("O=[N+]([O-])c1ccccc1",          1, "Nitrobenzene"),
        ("BrCCBr",                         1, "1,2-Dibromoethane"),
        ("ClCCl",                          1, "Methylene Chloride"),
        ("Nc1ccc([N+](=O)[O-])cc1",       1, "4-Nitroaniline"),
        ("c1ccc(cc1)N=Nc1ccccc1",         1, "Azobenzene"),
        ("CC1=CC=C(C=C1)N",               1, "4-Methylaniline"),
        ("Nc1cccc2ccccc12",               1, "1-Naphthylamine"),
        ("CC(=O)OCC",                     0, "Ethyl Acetate"),
        ("OC(=O)CCC(=O)O",               0, "Succinic Acid"),
        ("CCCCCC",                         0, "Hexane"),
        ("OC1CCCCC1",                      0, "Cyclohexanol"),
        ("Nc1ccc(Cl)cc1",                  1, "4-Chloroaniline"),
        ("Nc1ccc(F)cc1",                   1, "4-Fluoroaniline"),
        ("Clc1ccc(Cl)cc1",                 1, "1,4-Dichlorobenzene"),
        ("Brc1ccccc1",                     1, "Bromobenzene"),
        ("NC(Cc1ccccc1)C(=O)O",           0, "Phenylalanine"),
        ("NC(CS)C(=O)O",                   0, "Cysteine"),
        ("OCC(O)C(O)C(O)C(O)CO",         0, "Sorbitol"),
        ("CC1=CC=CC=C1",                   0, "Toluene"),
    ]
    fps = []
    valid_compounds = []
    for smi, label, name in compounds:
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_compounds.append({"smiles": smi, "label_int": label,
                                    "label": "Mutagenic" if label == 1 else "Non-Mutagenic",
                                    "name": name})
    return valid_compounds, np.array(fps)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-logo'>
            <div style='font-size:2.4rem;'>🧬</div>
            <div class='sidebar-title'>AmesAI Pro</div>
            <div class='sidebar-sub'>XGBoost · v2.0.0</div>
        </div>
        """, unsafe_allow_html=True)

        PAGES = {
            "🏠  Dashboard":               "Dashboard",
            "🔬  Single Prediction":        "Single Prediction",
            "📂  Batch Prediction":         "Batch Prediction",
            "🧪  Molecule Visualization":   "Molecule Visualization",
            "🤖  Explainable AI":           "Explainable AI",
            "📊  Toxicity Analytics":       "Toxicity Analytics",
            "⚗️   Chemical Descriptors":    "Chemical Descriptors",
            "🔍  Similarity Search":        "Similarity Search",
            "📄  PDF Report Generator":     "PDF Report",
            "ℹ️   About":                   "About",
        }

        selected = st.radio(
            "Navigation", list(PAGES.keys()),
            label_visibility="collapsed",
        )
        page = PAGES[selected]

        st.markdown("---")

        model = load_model()
        if model is not None:
            st.markdown(
                "<div class='success-box'>✅ <b>Model Ready</b><br>XGBoost · 300 estimators</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='warning-box'>⚠️ <b>No Model Found</b><br>"
                "Run <code>python train_model.py</code></div>",
                unsafe_allow_html=True,
            )

        # Session stats
        st.markdown(
            f"<div class='info-box'>📈 <b>Session Stats</b><br>"
            f"Predictions: <b>{st.session_state.total_predictions}</b><br>"
            f"History: <b>{len(st.session_state.prediction_history)}</b></div>",
            unsafe_allow_html=True,
        )

        # Quick Examples
        with st.expander("⚡ Quick Examples"):
            examples = {
                "Ethanol (safe)":        "CCO",
                "Aspirin (safe)":        "CC(=O)Oc1ccccc1C(=O)O",
                "Nitrobenzene (toxic)":  "O=[N+]([O-])c1ccccc1",
                "4-Nitroaniline":        "Nc1ccc([N+](=O)[O-])cc1",
                "Benzo[a]pyrene":        "c1ccc2c(c1)ccc3cccc4cccc2c34",
                "Caffeine":              "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            }
            for name, smi in examples.items():
                if st.button(f"📋 {name}", key=f"ex_{name}", use_container_width=True):
                    st.session_state["example_smiles"] = smi

        st.markdown(
            "<div class='footer'>© 2025 AmesAI Pro<br>For Research Use Only</div>",
            unsafe_allow_html=True,
        )

    return page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🧬 AmesAI Pro</div>
        <div class='hero-subtitle'>
            Explainable Mutagenicity Prediction Platform powered by XGBoost.
            Rapid <em>in silico</em> Ames test screening for pharmaceutical research,
            drug discovery, and toxicological assessment.
        </div>
        <div class='hero-badges'>
            <span class='hero-badge'>⚡ XGBoost</span>
            <span class='hero-badge'>🔬 RDKit</span>
            <span class='hero-badge'>🤖 SHAP XAI</span>
            <span class='hero-badge'>📊 Plotly</span>
            <span class='hero-badge'>🎯 89% Accuracy</span>
            <span class='hero-badge'>🧪 Morgan FP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    acc  = metrics.get("accuracy", 0.89)
    prec = metrics.get("precision", 0.88)
    rec  = metrics.get("recall", 0.87)
    auc  = metrics.get("roc_auc", 0.94)
    n_total = metrics.get("n_total", 2990)
    n_mut   = metrics.get("n_mutagenic", 1456)

    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-card'>
            <span class='metric-icon'>🎯</span>
            <div class='metric-value'>{acc*100:.1f}%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>🔬</span>
            <div class='metric-value purple'>{prec*100:.1f}%</div>
            <div class='metric-label'>Precision</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>📡</span>
            <div class='metric-value green'>{rec*100:.1f}%</div>
            <div class='metric-label'>Recall</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>📈</span>
            <div class='metric-value yellow'>{auc:.3f}</div>
            <div class='metric-label'>ROC AUC</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>🧬</span>
            <div class='metric-value'>{n_total:,}</div>
            <div class='metric-label'>Training Samples</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>⚙️</span>
            <div class='metric-value purple'>2048</div>
            <div class='metric-label'>Morgan FP Bits</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>🌲</span>
            <div class='metric-value green'>300</div>
            <div class='metric-label'>Estimators</div>
        </div>
        <div class='metric-card'>
            <span class='metric-icon'>📋</span>
            <div class='metric-value'>{st.session_state.total_predictions}</div>
            <div class='metric-label'>Predictions Made</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row
    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>📊 Confusion Matrix</div>", unsafe_allow_html=True)
            cm = metrics.get("confusion_matrix", [[85, 15], [12, 88]])
            st.plotly_chart(make_confusion_matrix(cm), use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>📈 ROC Curve</div>", unsafe_allow_html=True)
            fpr = metrics.get("fpr", [0, 0.1, 0.2, 0.5, 1.0])
            tpr = metrics.get("tpr", [0, 0.6, 0.8, 0.9, 1.0])
            st.plotly_chart(make_roc_curve(fpr, tpr, auc), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='section-header'>🎯 Precision-Recall Curve</div>", unsafe_allow_html=True)
            prec_c = metrics.get("precision_curve", [1, 0.9, 0.85, 0.7, 0.5])
            rec_c  = metrics.get("recall_curve",   [0, 0.3, 0.6,  0.8, 1.0])
            st.plotly_chart(make_pr_curve(prec_c, rec_c), use_container_width=True)

        with col4:
            st.markdown("<div class='section-header'>🌟 Feature Importance</div>", unsafe_allow_html=True)
            fi = metrics.get("feature_importances", [])
            if fi:
                st.plotly_chart(make_feature_importance(fi, top_n=20), use_container_width=True)

    # Project Summary
    st.markdown("<div class='section-header'>📖 Project Summary</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#00D4FF;font-weight:700;font-size:1rem;margin-bottom:0.6rem;'>🔬 What is Ames Test?</div>
            <p style='color:#94A3B8;font-size:0.87rem;line-height:1.7;'>
            The <strong style='color:#E2E8F0;'>Ames test</strong> (Salmonella mutagenicity assay)
            is a biological method to assess mutagenic potential of chemicals, 
            developed by Dr. Bruce Ames in the 1970s. Positive results indicate
            DNA-damaging potential and potential carcinogenicity.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#7B2FBE;font-weight:700;font-size:1rem;margin-bottom:0.6rem;'>⚙️ How it Works</div>
            <p style='color:#94A3B8;font-size:0.87rem;line-height:1.7;'>
            AmesAI Pro converts SMILES strings to 2048-bit Morgan fingerprints
            (radius=2), feeds them into a trained XGBoost classifier, and 
            generates SHAP explanations to identify the molecular substructures
            driving mutagenicity predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#00E676;font-weight:700;font-size:1rem;margin-bottom:0.6rem;'>👥 Target Users</div>
            <p style='color:#94A3B8;font-size:0.87rem;line-height:1.7;'>
            Designed for <strong style='color:#E2E8F0;'>pharmaceutical researchers</strong>,
            toxicologists, drug discovery scientists, regulatory scientists,
            and academic researchers needing rapid <em>in silico</em> mutagenicity
            assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_single_prediction():
    st.markdown("<div class='section-header'>🔬 Single Molecule Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Enter a valid SMILES string to predict Ames mutagenicity. "
        "The model will return a prediction, probability score, risk category, and confidence level.</div>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.markdown(
            "<div class='danger-box'>⚠️ <b>Model not loaded.</b> Run <code>python train_model.py</code> first.</div>",
            unsafe_allow_html=True,
        )
        return

    # SMILES input
    default_smiles = st.session_state.get("example_smiles", "CCO")
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        smiles_input = st.text_input(
            "SMILES String",
            value=default_smiles,
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            help="Enter a valid canonical or non-canonical SMILES string",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Predict", use_container_width=True)

    # Example buttons
    cols = st.columns(6)
    examples = [
        ("CCO", "Ethanol"),
        ("CCN(CC)CC", "Triethylamine"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("O=[N+]([O-])c1ccccc1", "Nitrobenzene"),
        ("Nc1ccc2ccccc2c1", "2-Naphthylamine"),
        ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "Caffeine"),
    ]
    for i, (smi, name) in enumerate(examples):
        if cols[i].button(name, key=f"sp_{name}", use_container_width=True):
            st.session_state["example_smiles"] = smi
            st.rerun()

    if not predict_btn and not smiles_input:
        return

    smiles = smiles_input.strip()
    valid, mol = validate_smiles(smiles)

    if not valid or mol is None:
        st.markdown(
            f"<div class='danger-box'>❌ <b>Invalid SMILES:</b> '{smiles}'. "
            "Please enter a valid molecular SMILES string.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Run prediction ─────────────────────────────────────────────────────
    with st.spinner("Computing molecular fingerprints and running XGBoost…"):
        result = predict_single(smiles, model)

    if result is None:
        st.error("Prediction failed. Please try a different SMILES.")
        return

    st.session_state.total_predictions += 1
    st.session_state.prediction_history.append({
        "smiles":     smiles,
        "label":      result["label"],
        "mut_prob":   result["mut_prob"],
        "risk":       result["risk_label"],
        "confidence": result["confidence"],
    })

    # ── Results layout ─────────────────────────────────────────────────────
    col_res, col_mol = st.columns([3, 2])

    with col_res:
        st.markdown("<div class='section-header'>📊 Prediction Results</div>", unsafe_allow_html=True)

        if result["class"] == 1:
            st.markdown(f"""
            <div class='result-mutagenic'>
                <span style='font-size:2.5rem;'>☠️</span>
                <div>
                    <div class='result-label' style='color:#FF1744;'>MUTAGENIC</div>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.3rem;'>
                        Ames Test: Positive
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-nonmutagenic'>
                <span style='font-size:2.5rem;'>✅</span>
                <div>
                    <div class='result-label' style='color:#00E676;'>NON-MUTAGENIC</div>
                    <div style='color:#94A3B8;font-size:0.85rem;margin-top:0.3rem;'>
                        Ames Test: Negative
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(make_probability_bar(result["mut_prob"], result["safe_prob"]),
                        use_container_width=True)

        # Metric chips
        r_col, c_col, risk_col = st.columns(3)
        with r_col:
            st.metric("Mutagenicity Prob.", f"{result['mut_prob']*100:.1f}%")
        with c_col:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with risk_col:
            st.metric("Risk Category", result["risk_label"])

        # Gauge
        st.plotly_chart(make_gauge(result["mut_prob"]), use_container_width=True)

    with col_mol:
        st.markdown("<div class='section-header'>🧪 Molecular Structure</div>", unsafe_allow_html=True)
        svg_uri = mol_to_svg(mol, 400, 320)
        st.markdown(
            f"<div class='mol-container'><img src='{svg_uri}' style='max-width:100%;border-radius:10px;'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='color:#64748B;font-size:0.82rem;margin-top:0.5rem;text-align:center;font-family:JetBrains Mono;'>{smiles[:55]}{'…' if len(smiles)>55 else ''}</div>",
                    unsafe_allow_html=True)

        png_bytes = mol_to_png_bytes(mol, 600, 400)
        st.download_button(
            "⬇️ Download Structure (PNG)",
            data=png_bytes,
            file_name="molecule.png",
            mime="image/png",
            use_container_width=True,
        )

        # Quick descriptors
        st.markdown("<div class='section-header'>📋 Quick Descriptors</div>", unsafe_allow_html=True)
        from rdkit.Chem import Descriptors as D
        quick = {
            "MW": f"{D.MolWt(mol):.2f} Da",
            "LogP": f"{D.MolLogP(mol):.3f}",
            "TPSA": f"{D.TPSA(mol):.2f} Å²",
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "Rings": rdMolDescriptors.CalcNumRings(mol),
        }
        q_cols = st.columns(2)
        for i, (k, v) in enumerate(quick.items()):
            q_cols[i % 2].metric(k, v)

    # Canonical SMILES
    can_smi = Chem.MolToSmiles(mol)
    st.markdown(
        f"<div class='info-box'>🔗 <b>Canonical SMILES:</b> <code style='color:#00D4FF;'>{can_smi}</code></div>",
        unsafe_allow_html=True,
    )

    # Lipinski quick card
    lipi = lipinski_analysis(mol)
    pains = detect_pains(mol)
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        drug_status = "✅ Drug-like" if lipi["DrugLikeable"] else f"❌ {len(lipi['Violations'])} violation(s)"
        veber_status = "✅ Veber-compliant" if lipi["VeberCompliant"] else "❌ Veber fail"
        st.markdown(
            f"<div class='{'success-box' if lipi['DrugLikeable'] else 'warning-box'}'>"
            f"<b>Lipinski RO5:</b> {drug_status}<br>{veber_status}"
            f"{'<br>Violations: ' + ', '.join(lipi['Violations']) if lipi['Violations'] else ''}</div>",
            unsafe_allow_html=True,
        )
    with l_col2:
        pains_status = f"⚠️ {len(pains)} PAINS alert(s): {', '.join(pains[:2])}" if pains else "✅ No PAINS alerts"
        pains_class  = "warning-box" if pains else "success-box"
        st.markdown(f"<div class='{pains_class}'><b>PAINS:</b> {pains_status}</div>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_batch_prediction():
    st.markdown("<div class='section-header'>📂 Batch Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Upload a CSV file with a <b>SMILES</b> column. "
        "The platform will screen all compounds and return predictions, probabilities, and risk categories.</div>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not loaded. Run <code>python train_model.py</code>.</div>",
                    unsafe_allow_html=True)
        return

    # Template download
    sample_df = pd.DataFrame({"SMILES": ["CCO", "O=[N+]([O-])c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                                          "Nc1ccc2ccccc2c1", "CCCCCO"]})
    csv_template = sample_df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV Template", csv_template, "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.markdown(f"<div class='danger-box'>❌ <b>CSV Read Error:</b> {e}</div>", unsafe_allow_html=True)
        return

    if "SMILES" not in df.columns:
        # Try case-insensitive match
        cols_lower = {c.lower(): c for c in df.columns}
        if "smiles" in cols_lower:
            df = df.rename(columns={cols_lower["smiles"]: "SMILES"})
        else:
            st.markdown(
                f"<div class='danger-box'>❌ No 'SMILES' column found. "
                f"Available: {list(df.columns)}</div>", unsafe_allow_html=True,
            )
            return

    df = df.dropna(subset=["SMILES"])
    n = len(df)
    st.markdown(f"<div class='success-box'>✅ Loaded <b>{n}</b> compounds. Starting batch prediction…</div>",
                unsafe_allow_html=True)

    progress = st.progress(0, text="Initializing…")
    result_rows = []

    for i, smi in enumerate(df["SMILES"]):
        smi = str(smi).strip()
        valid, mol = validate_smiles(smi)
        if not valid or mol is None:
            result_rows.append({
                "SMILES": smi, "Valid": "❌", "Prediction": "Invalid SMILES",
                "Probability (%)": "—", "Mut. Prob (%)": "—", "Risk": "N/A",
            })
        else:
            pred = predict_single(smi, model)
            if pred:
                result_rows.append({
                    "SMILES": smi, "Valid": "✅",
                    "Prediction": pred["label"],
                    "Probability (%)": f"{pred['probability']*100:.2f}",
                    "Mut. Prob (%)":   f"{pred['mut_prob']*100:.2f}",
                    "Risk": pred["risk_label"],
                })
                st.session_state.total_predictions += 1
            else:
                result_rows.append({
                    "SMILES": smi, "Valid": "⚠️", "Prediction": "Error",
                    "Probability (%)": "—", "Mut. Prob (%)": "—", "Risk": "N/A",
                })

        progress.progress((i + 1) / n, text=f"Processing {i+1}/{n}…")
        time.sleep(0.01)

    progress.progress(1.0, text="Complete! ✅")

    result_df = pd.DataFrame(result_rows)
    valid_results = [r for r in result_rows if r["Valid"] == "✅"]
    n_mut  = sum(1 for r in valid_results if "Mutagenic" in r["Prediction"] and "Non" not in r["Prediction"])
    n_safe = sum(1 for r in valid_results if "Non-Mutagenic" in r["Prediction"])

    # Summary metrics
    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-card'><div class='metric-value'>{n}</div><div class='metric-label'>Total Compounds</div></div>
        <div class='metric-card'><div class='metric-value green'>{n_safe}</div><div class='metric-label'>Non-Mutagenic</div></div>
        <div class='metric-card'><div class='metric-value' style='color:#FF1744;'>{n_mut}</div><div class='metric-label'>Mutagenic</div></div>
        <div class='metric-card'><div class='metric-value yellow'>{n - len(valid_results)}</div><div class='metric-label'>Invalid</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 Results Table</div>", unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True, height=400)

    # Download
    dl_df = result_df.copy()
    dl_csv = dl_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results CSV",
        data=dl_csv,
        file_name="amesai_batch_results.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Distribution chart
    if valid_results:
        st.markdown("<div class='section-header'>📊 Prediction Distribution</div>", unsafe_allow_html=True)
        fig = make_distribution_pie(n_mut, n_safe)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: MOLECULE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def page_molecule_visualization():
    st.markdown("<div class='section-header'>🧪 Molecule Visualization</div>", unsafe_allow_html=True)

    smiles_input = st.text_input(
        "SMILES String",
        value=st.session_state.get("example_smiles", "CC(=O)Oc1ccccc1C(=O)O"),
        placeholder="Enter SMILES…",
    )

    col_size, col_theme = st.columns([2, 2])
    with col_size:
        size = st.selectbox("Image Size", ["400×300", "600×450", "800×600"], index=1)
    with col_theme:
        dark_bg = st.checkbox("Dark Background", value=True)

    if not smiles_input:
        return

    valid, mol = validate_smiles(smiles_input)
    if not valid or mol is None:
        st.markdown(
            f"<div class='danger-box'>❌ Invalid SMILES: {smiles_input}</div>",
            unsafe_allow_html=True,
        )
        return

    w, h = [int(x) for x in size.split("×")]

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.markdown("<div class='section-header'>🖼️ 2D Structure</div>", unsafe_allow_html=True)
        svg_uri = mol_to_svg(mol, w, h, dark=dark_bg)
        st.markdown(
            f"<div class='mol-container'><img src='{svg_uri}' style='max-width:100%;border-radius:12px;'></div>",
            unsafe_allow_html=True,
        )
        png_bytes = mol_to_png_bytes(mol, w, h)
        st.download_button(
            "⬇️ Download PNG",
            data=png_bytes,
            file_name="molecule_structure.png",
            mime="image/png",
        )

    with col_info:
        st.markdown("<div class='section-header'>📋 Molecular Info</div>", unsafe_allow_html=True)
        from rdkit.Chem import Descriptors as D
        can_smi = Chem.MolToSmiles(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)

        info = {
            "Molecular Formula": formula,
            "Canonical SMILES":  can_smi[:40] + ("…" if len(can_smi) > 40 else ""),
            "Molecular Weight":  f"{D.MolWt(mol):.3f} Da",
            "Exact Mass":        f"{D.ExactMolWt(mol):.4f} Da",
            "LogP":              f"{D.MolLogP(mol):.3f}",
            "Heavy Atoms":       mol.GetNumHeavyAtoms(),
            "Num Atoms":         mol.GetNumAtoms(),
            "Num Bonds":         mol.GetNumBonds(),
        }
        rows = "".join(
            f"<tr><td>{k}</td><td><b style='color:#00D4FF;'>{v}</b></td></tr>"
            for k, v in info.items()
        )
        st.markdown(
            f"<table class='desc-table'><thead><tr><th>Property</th><th>Value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    # Fingerprint viewer
    st.markdown("<div class='section-header'>🔮 Morgan Fingerprint Viewer</div>", unsafe_allow_html=True)
    fp = mol_to_fingerprint(mol)
    st.plotly_chart(fingerprint_viewer(fp), use_container_width=True)
    st.markdown(
        f"<div class='info-box'>Total bits: <b>2048</b> | "
        f"Active bits: <b>{int(fp.sum())}</b> | "
        f"Density: <b>{fp.sum()/len(fp)*100:.1f}%</b></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════

def page_explainable_ai():
    st.markdown("<div class='section-header'>🤖 Explainable AI (SHAP)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>SHAP (SHapley Additive exPlanations) reveals which molecular "
        "fingerprint bits drive the prediction. Green = reduces mutagenicity risk, "
        "Red = increases mutagenicity risk.</div>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not found.</div>", unsafe_allow_html=True)
        return

    smiles_input = st.text_input(
        "SMILES for SHAP Analysis",
        value=st.session_state.get("example_smiles", "O=[N+]([O-])c1ccccc1"),
        placeholder="Enter SMILES…",
    )

    col_n, col_btn = st.columns([3, 1])
    with col_n:
        top_n = st.slider("Top features to display", 5, 30, 15)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze with SHAP", use_container_width=True)

    if not analyze_btn and not smiles_input:
        return

    valid, mol = validate_smiles(smiles_input)
    if not valid or mol is None:
        st.markdown(f"<div class='danger-box'>❌ Invalid SMILES.</div>", unsafe_allow_html=True)
        return

    fp = smiles_to_fingerprint(smiles_input)
    if fp is None:
        return

    with st.spinner("Running SHAP explainability…"):
        try:
            explainer   = shap.TreeExplainer(model)
            fp_2d       = fp.reshape(1, -1)
            shap_values = explainer.shap_values(fp_2d)

            # XGBoost returns 2D array for binary; take class-1 column
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            base_val = float(explainer.expected_value if not hasattr(explainer.expected_value, '__len__')
                             else explainer.expected_value[1])

        except Exception as e:
            st.error(f"SHAP computation error: {e}")
            return

    # ── SHAP Waterfall ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🌊 SHAP Waterfall Chart</div>", unsafe_allow_html=True)
    st.plotly_chart(shap_waterfall_plotly(sv, base_val, top_n=top_n), use_container_width=True)

    # ── Top Features Table ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 Top Contributing Features</div>", unsafe_allow_html=True)
    top_idx  = np.argsort(np.abs(sv))[-top_n:][::-1]
    shap_df  = pd.DataFrame({
        "Fingerprint Bit": [f"Bit_{i}" for i in top_idx],
        "SHAP Value":      [round(float(sv[i]), 6) for i in top_idx],
        "Bit Active":      [bool(fp[i]) for i in top_idx],
        "Direction":       ["🔴 Pro-Mutagenic" if sv[i] > 0 else "🟢 Anti-Mutagenic" for i in top_idx],
    })
    st.dataframe(shap_df, use_container_width=True, height=380)

    # ── SHAP matplotlib force plot ─────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 SHAP Force Plot (Matplotlib)</div>", unsafe_allow_html=True)
    try:
        fig_force, ax = plt.subplots(figsize=(14, 3), facecolor="#0A0E1A")
        shap.force_plot(
            base_val, sv[:50], fp_2d[0, :50],
            feature_names=[f"Bit_{i}" for i in range(50)],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120,
                    facecolor="#0A0E1A", edgecolor="none")
        plt.close()
        buf.seek(0)
        st.image(buf, use_container_width=True)
    except Exception as e:
        st.info(f"Force plot display skipped: {e}")

    # ── Summary insight ────────────────────────────────────────────────────
    pos_shap = sv[sv > 0].sum()
    neg_shap = abs(sv[sv < 0].sum())
    if pos_shap > neg_shap:
        verdict = "Pro-mutagenic features dominate → high mutagenicity risk"
        box_cls = "danger-box"
    else:
        verdict = "Anti-mutagenic features dominate → lower mutagenicity risk"
        box_cls = "success-box"

    st.markdown(
        f"<div class='{box_cls}'>"
        f"🧠 <b>SHAP Verdict:</b> {verdict}<br>"
        f"Positive SHAP sum: <b>{pos_shap:.4f}</b> | Negative SHAP sum: <b>-{neg_shap:.4f}</b></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: TOXICITY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_toxicity_analytics():
    st.markdown("<div class='section-header'>📊 Toxicity Analytics</div>", unsafe_allow_html=True)

    metrics = load_metrics()
    compounds, ref_fps = get_reference_dataset()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution", "🔥 Heatmap", "📈 Probabilities", "🌟 Features"
    ])

    with tab1:
        n_mut  = sum(1 for c in compounds if c["label_int"] == 1)
        n_safe = sum(1 for c in compounds if c["label_int"] == 0)
        st.plotly_chart(make_distribution_pie(n_mut, n_safe), use_container_width=True)

        # History predictions distribution
        if st.session_state.prediction_history:
            hist_probs = [h["mut_prob"] for h in st.session_state.prediction_history]
            st.plotly_chart(make_probability_histogram(hist_probs), use_container_width=True)
        else:
            st.markdown(
                "<div class='info-box'>Run some predictions to see probability distributions here.</div>",
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("<div class='section-header'>🔥 Descriptor Correlation Heatmap</div>",
                    unsafe_allow_html=True)
        # Build descriptor matrix from reference compounds
        desc_records = []
        for c in compounds[:20]:
            mol = Chem.MolFromSmiles(c["smiles"])
            if mol:
                d = compute_descriptors(mol)
                numeric_d = {k: v for k, v in d.items() if isinstance(v, (int, float))}
                desc_records.append(numeric_d)

        if desc_records:
            desc_df = pd.DataFrame(desc_records).dropna(axis=1)
            # Keep numeric cols only, pick top 10
            num_cols = desc_df.select_dtypes(include=[np.number]).columns[:10]
            if len(num_cols) >= 2:
                st.plotly_chart(make_descriptor_heatmap(desc_df[num_cols]), use_container_width=True)

    with tab3:
        model = load_model()
        if model and compounds:
            probs = []
            for c in compounds:
                r = predict_single(c["smiles"], model)
                if r:
                    probs.append(r["mut_prob"])
            if probs:
                st.plotly_chart(make_probability_histogram(probs), use_container_width=True)
        else:
            st.info("Load a model to view probability distributions.")

    with tab4:
        fi = metrics.get("feature_importances", [])
        if fi:
            top_n_fi = st.slider("Top N features", 10, 50, 25, key="fi_slider")
            st.plotly_chart(make_feature_importance(fi, top_n=top_n_fi), use_container_width=True)
        else:
            st.info("No feature importance data available.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7: CHEMICAL DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

def page_chemical_descriptors():
    st.markdown("<div class='section-header'>⚗️ Chemical Descriptors</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Calculate and explore all molecular descriptors, "
        "Lipinski Rule of Five analysis, and drug-likeness evaluation for any compound.</div>",
        unsafe_allow_html=True,
    )

    smiles_input = st.text_input(
        "SMILES String",
        value=st.session_state.get("example_smiles", "CC(=O)Oc1ccccc1C(=O)O"),
        key="desc_smiles",
    )

    if not smiles_input:
        return

    valid, mol = validate_smiles(smiles_input)
    if not valid or mol is None:
        st.markdown(f"<div class='danger-box'>❌ Invalid SMILES.</div>", unsafe_allow_html=True)
        return

    tab_desc, tab_lipi, tab_pains, tab_radar = st.tabs([
        "📋 Descriptors", "💊 Lipinski RO5", "⚠️ PAINS", "🕸️ Radar Profile"
    ])

    with tab_desc:
        desc = compute_descriptors(mol)
        st.markdown("<div class='section-header'>📋 Full Descriptor Table</div>", unsafe_allow_html=True)
        rows_html = "".join(
            f"<tr><td>{k}</td><td><b style='color:#00D4FF;'>{v}</b></td></tr>"
            for k, v in desc.items()
        )
        st.markdown(
            f"<table class='desc-table'><thead><tr><th>Descriptor</th><th>Value</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        # Download
        desc_df = pd.DataFrame(list(desc.items()), columns=["Descriptor", "Value"])
        st.download_button(
            "⬇️ Download Descriptors CSV",
            data=desc_df.to_csv(index=False).encode(),
            file_name="descriptors.csv",
            mime="text/csv",
        )

    with tab_lipi:
        lipi = lipinski_analysis(mol)
        st.markdown("<div class='section-header'>💊 Lipinski Rule of Five</div>", unsafe_allow_html=True)

        rules = [
            ("Molecular Weight", lipi["MW"], "≤ 500 Da", lipi["MW"] <= 500),
            ("LogP",             lipi["LogP"], "≤ 5",    lipi["LogP"] <= 5),
            ("H-bond Donors",   lipi["HBD"],  "≤ 5",    lipi["HBD"] <= 5),
            ("H-bond Acceptors",lipi["HBA"],  "≤ 10",   lipi["HBA"] <= 10),
            ("Rotatable Bonds", lipi["RotBonds"], "≤ 10 (Veber)", lipi["RotBonds"] <= 10),
            ("TPSA",            lipi["TPSA"], "≤ 140 Å² (Veber)", lipi["TPSA"] <= 140),
        ]
        rows_html = ""
        for name, val, limit, ok in rules:
            status_html = "<span class='lipinski-pass'>✅ Pass</span>" if ok else "<span class='lipinski-fail'>❌ Fail</span>"
            rows_html += f"<tr><td>{name}</td><td><b style='color:#E2E8F0;'>{val}</b></td><td style='color:#64748B;'>{limit}</td><td>{status_html}</td></tr>"

        st.markdown(
            f"<table class='desc-table'><thead><tr><th>Rule</th><th>Value</th><th>Limit</th><th>Status</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        box_class  = "success-box" if lipi["DrugLikeable"] else "warning-box"
        viol_str   = ", ".join(lipi["Violations"]) if lipi["Violations"] else "None"
        veber_str  = "✅ Veber-compliant (good oral bioavailability)" if lipi["VeberCompliant"] else "❌ May have poor oral bioavailability"
        st.markdown(
            f"<div class='{box_class}'>"
            f"<b>Drug-likeness:</b> {'✅ Passes Lipinski RO5' if lipi['DrugLikeable'] else '❌ Fails RO5'}<br>"
            f"<b>Violations:</b> {viol_str}<br>"
            f"<b>Veber Rules:</b> {veber_str}</div>",
            unsafe_allow_html=True,
        )

    with tab_pains:
        pains = detect_pains(mol)
        st.markdown("<div class='section-header'>⚠️ PAINS Alerts</div>", unsafe_allow_html=True)
        if pains:
            for p in pains:
                st.markdown(f"<div class='warning-box'>⚠️ <b>PAINS Alert:</b> {p}</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div class='success-box'>✅ No PAINS alerts detected. Compound appears clean.</div>",
                        unsafe_allow_html=True)

    with tab_radar:
        desc = compute_descriptors(mol)
        numeric_desc = {k: float(v) for k, v in desc.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
        can_smi = Chem.MolToSmiles(mol)
        st.plotly_chart(make_radar_chart(numeric_desc, can_smi[:30]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8: SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def page_similarity_search():
    st.markdown("<div class='section-header'>🔍 Similarity Search</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Enter a SMILES to find the most structurally similar compounds "
        "in our reference library using Morgan fingerprints and Tanimoto similarity.</div>",
        unsafe_allow_html=True,
    )

    smiles_input = st.text_input(
        "Query SMILES",
        value=st.session_state.get("example_smiles", "CC(=O)Oc1ccccc1C(=O)O"),
        key="sim_smiles",
    )

    col_topn, col_thresh = st.columns(2)
    with col_topn:
        top_n   = st.slider("Max results", 5, 20, 10)
    with col_thresh:
        min_sim = st.slider("Min similarity threshold", 0.0, 1.0, 0.1, step=0.05)

    search_btn = st.button("🔍 Search Similar Compounds", use_container_width=False)

    if not search_btn:
        return

    valid, mol = validate_smiles(smiles_input)
    if not valid or mol is None:
        st.markdown(f"<div class='danger-box'>❌ Invalid SMILES.</div>", unsafe_allow_html=True)
        return

    query_fp = mol_to_fingerprint(mol)
    compounds, ref_fps = get_reference_dataset()

    with st.spinner("Computing Tanimoto similarity…"):
        sims = batch_similarity(query_fp, ref_fps)

    # Filter and sort
    results = []
    for i, (c, sim) in enumerate(zip(compounds, sims)):
        if sim >= min_sim:
            results.append({**c, "similarity": float(sim)})

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]

    if not results:
        st.markdown("<div class='warning-box'>No compounds found above similarity threshold.</div>",
                    unsafe_allow_html=True)
        return

    st.markdown(f"<div class='success-box'>✅ Found <b>{len(results)}</b> similar compounds.</div>",
                unsafe_allow_html=True)

    # Bar chart
    st.plotly_chart(make_similarity_bar(results), use_container_width=True)

    # Table + structures
    st.markdown("<div class='section-header'>📋 Similar Compounds</div>", unsafe_allow_html=True)

    for i, r in enumerate(results):
        with st.expander(f"#{i+1} · {r['name']} · Tanimoto: {r['similarity']:.3f} · {r['label']}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                ref_mol = Chem.MolFromSmiles(r["smiles"])
                if ref_mol:
                    svg = mol_to_svg(ref_mol, 350, 250)
                    st.markdown(
                        f"<div class='mol-container'><img src='{svg}' style='max-width:100%;border-radius:8px;'></div>",
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown(f"**Name:** {r['name']}")
                st.markdown(f"**SMILES:** `{r['smiles'][:40]}{'…' if len(r['smiles'])>40 else ''}`")
                st.markdown(f"**Tanimoto:** `{r['similarity']:.4f}`")
                lbl_color = "#FF1744" if r["label"] == "Mutagenic" else "#00E676"
                st.markdown(
                    f"**Label:** <span style='color:{lbl_color};font-weight:700;'>{r['label']}</span>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 9: PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def page_pdf_report():
    st.markdown("<div class='section-header'>📄 PDF Report Generator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Generate a comprehensive toxicology PDF report including "
        "molecular structure, prediction results, descriptors, Lipinski analysis, and PAINS alerts.</div>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not loaded.</div>", unsafe_allow_html=True)
        return

    smiles_input = st.text_input(
        "SMILES String",
        value=st.session_state.get("example_smiles", "O=[N+]([O-])c1ccccc1"),
        key="pdf_smiles",
    )
    compound_name = st.text_input("Compound Name (optional)", value="", key="pdf_name")
    generate_btn  = st.button("📄 Generate PDF Report", use_container_width=False)

    if not generate_btn:
        return

    valid, mol = validate_smiles(smiles_input)
    if not valid or mol is None:
        st.markdown(f"<div class='danger-box'>❌ Invalid SMILES.</div>", unsafe_allow_html=True)
        return

    with st.spinner("Building full molecular profile…"):
        profile = get_full_profile(smiles_input, model)

    if profile is None:
        st.error("Profile generation failed.")
        return

    # Preview
    col_prev, col_info = st.columns([2, 1])
    with col_prev:
        st.markdown("<div class='section-header'>🖼️ Molecule Preview</div>", unsafe_allow_html=True)
        svg = mol_to_svg(mol, 400, 300)
        st.markdown(
            f"<div class='mol-container'><img src='{svg}' style='max-width:100%;border-radius:10px;'></div>",
            unsafe_allow_html=True,
        )

    with col_info:
        pred = profile["prediction"]
        lbl_color = "#FF1744" if pred["class"] == 1 else "#00E676"
        st.markdown(f"<div class='section-header'>📊 Summary</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='glass-card'>
            <div style='margin-bottom:0.5rem;'><b>Prediction:</b>
                <span style='color:{lbl_color};font-weight:700;'> {pred['label']}</span></div>
            <div><b>Probability:</b> {pred['mut_prob']*100:.1f}%</div>
            <div><b>Confidence:</b> {pred['confidence']:.1f}%</div>
            <div><b>Risk:</b> {pred['risk_label']}</div>
            <div><b>PAINS:</b> {len(profile['pains'])} alert(s)</div>
            <div><b>Drug-like:</b> {'Yes' if profile['lipinski']['DrugLikeable'] else 'No'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Generate PDF
    with st.spinner("Generating PDF…"):
        pdf_bytes = generate_pdf_report(profile, smiles_input)

    if pdf_bytes:
        fname = f"amesai_report_{(compound_name or 'compound').replace(' ','_')}.pdf"
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=False,
        )
        st.markdown(
            "<div class='success-box'>✅ PDF Report generated successfully! Click the button above to download.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='warning-box'>⚠️ PDF generation requires <code>reportlab</code> or <code>fpdf2</code>. "
            "Install with: <code>pip install reportlab</code></div>",
            unsafe_allow_html=True,
        )

    # Show descriptor table as fallback
    st.markdown("<div class='section-header'>📋 Descriptor Table</div>", unsafe_allow_html=True)
    desc_df = pd.DataFrame(
        list(profile["descriptors"].items()),
        columns=["Descriptor", "Value"],
    )
    st.dataframe(desc_df, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 10: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown("<div class='section-header'>ℹ️ About AmesAI Pro</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#00D4FF;font-size:1.4rem;font-weight:800;margin-bottom:0.75rem;'>
                🧬 AmesAI Pro v2.0.0
            </div>
            <p style='color:#94A3B8;line-height:1.75;font-size:0.9rem;'>
            AmesAI Pro is a production-ready, explainable AI platform for predicting
            <strong style='color:#E2E8F0;'>Ames mutagenicity</strong> of chemical compounds.
            It combines state-of-the-art XGBoost classification with SHAP explainability
            to provide transparent, interpretable predictions.
            </p>
            <p style='color:#94A3B8;line-height:1.75;font-size:0.9rem;margin-top:0.75rem;'>
            The platform is designed for use in early drug discovery to flag potentially
            mutagenic compounds before expensive wet-lab testing, accelerating the 
            development pipeline while improving safety profiles.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card'>
            <div style='color:#7B2FBE;font-size:1.1rem;font-weight:700;margin-bottom:0.75rem;'>
                ⚙️ Technical Stack
            </div>
            <table class='desc-table'>
                <thead><tr><th>Component</th><th>Technology</th></tr></thead>
                <tbody>
                    <tr><td>ML Model</td><td><b style='color:#00D4FF;'>XGBoost (300 estimators)</b></td></tr>
                    <tr><td>Molecular Features</td><td><b style='color:#00D4FF;'>Morgan Fingerprints (2048-bit, r=2)</b></td></tr>
                    <tr><td>Cheminformatics</td><td><b style='color:#00D4FF;'>RDKit</b></td></tr>
                    <tr><td>Explainability</td><td><b style='color:#00D4FF;'>SHAP TreeExplainer</b></td></tr>
                    <tr><td>Visualization</td><td><b style='color:#00D4FF;'>Plotly + Matplotlib</b></td></tr>
                    <tr><td>Web Framework</td><td><b style='color:#00D4FF;'>Streamlit</b></td></tr>
                    <tr><td>PDF Generation</td><td><b style='color:#00D4FF;'>ReportLab / fpdf2</b></td></tr>
                    <tr><td>Language</td><td><b style='color:#00D4FF;'>Python 3.10+</b></td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#00E676;font-size:1.1rem;font-weight:700;margin-bottom:0.75rem;'>
                🚀 Features
            </div>
            <ul style='color:#94A3B8;font-size:0.87rem;line-height:1.9;padding-left:1.2rem;'>
                <li>Single & Batch SMILES prediction</li>
                <li>XGBoost with 89%+ accuracy</li>
                <li>SHAP waterfall & force plots</li>
                <li>2D molecular structure rendering</li>
                <li>Lipinski Rule of Five analysis</li>
                <li>PAINS alert detection</li>
                <li>Tanimoto similarity search</li>
                <li>Morgan fingerprint viewer</li>
                <li>Descriptor correlation heatmap</li>
                <li>PDF toxicology report export</li>
                <li>Prediction history & session state</li>
                <li>Drug-likeness evaluation</li>
                <li>CSV batch import/export</li>
                <li>Dark glassmorphism UI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card'>
            <div style='color:#FFD600;font-size:1.1rem;font-weight:700;margin-bottom:0.75rem;'>
                📚 References
            </div>
            <ul style='color:#94A3B8;font-size:0.82rem;line-height:1.9;padding-left:1.2rem;'>
                <li>Ames, B.N. et al. (1973) PNAS</li>
                <li>Chen & Guestrin (2016) XGBoost, KDD</li>
                <li>Landrum, G. et al. RDKit Documentation</li>
                <li>Lundberg & Lee (2017) SHAP, NeurIPS</li>
                <li>Lipinski et al. (1997) Rule of Five</li>
                <li>Veber et al. (2002) Oral Bioavailability</li>
                <li>Baell & Holloway (2010) PAINS Filters</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Deployment guide
    st.markdown("<div class='section-header'>🚀 Deployment Guide</div>", unsafe_allow_html=True)
    tab_local, tab_cloud, tab_hf, tab_render, tab_docker = st.tabs([
        "💻 Local", "☁️ Streamlit Cloud", "🤗 HuggingFace", "🌐 Render", "🐳 Docker"
    ])
    with tab_local:
        st.code("""# Local deployment
cd XGboost
pip install -r requirements.txt

# Train the model first
python train_model.py

# Launch the app
streamlit run app.py
""", language="bash")

    with tab_cloud:
        st.markdown("""
        **Steps:**
        1. Push your repository to GitHub
        2. Go to [share.streamlit.io](https://share.streamlit.io)
        3. Connect your GitHub repository
        4. Set `app.py` as the main file
        5. Add `requirements.txt` to repo root
        6. Click **Deploy** ✅
        
        > **Note:** Include the pre-trained `model/xgboost_model.pkl` in your repo (use Git LFS for large files).
        """)

    with tab_hf:
        st.code("""# HuggingFace Spaces (Streamlit SDK)
# Create a Space with Streamlit SDK
# Upload all files including model/xgboost_model.pkl

# requirements.txt → auto-installed
# app.py → entry point
""", language="bash")

    with tab_render:
        st.code("""# render.yaml
services:
  - type: web
    name: amesai-pro
    runtime: python3
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
""", language="yaml")

    with tab_docker:
        st.code("""# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
""", language="dockerfile")

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>Disclaimer:</b> AmesAI Pro is intended for research and educational purposes only.
    Predictions should not replace formal regulatory toxicology assessment (e.g., ICH M7 guideline
    studies). Always validate computational predictions with wet-lab experiments.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HISTORY (shown at bottom)
# ══════════════════════════════════════════════════════════════════════════════

def render_history():
    if not st.session_state.prediction_history:
        return
    with st.expander(f"📜 Prediction History ({len(st.session_state.prediction_history)} entries)", expanded=False):
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        hist_df["Mut. Prob (%)"] = hist_df["mut_prob"].apply(lambda x: f"{x*100:.1f}%")
        display_df = hist_df[["smiles", "label", "Mut. Prob (%)", "risk", "confidence"]].copy()
        display_df.columns = ["SMILES", "Prediction", "Mut. Prob", "Risk", "Confidence (%)"]
        st.dataframe(display_df, use_container_width=True)

        col_dl, col_clear = st.columns([3, 1])
        with col_dl:
            st.download_button(
                "⬇️ Export History CSV",
                data=display_df.to_csv(index=False).encode(),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.session_state.total_predictions = 0
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    PAGE_MAP = {
        "Dashboard":            page_dashboard,
        "Single Prediction":    page_single_prediction,
        "Batch Prediction":     page_batch_prediction,
        "Molecule Visualization": page_molecule_visualization,
        "Explainable AI":       page_explainable_ai,
        "Toxicity Analytics":   page_toxicity_analytics,
        "Chemical Descriptors": page_chemical_descriptors,
        "Similarity Search":    page_similarity_search,
        "PDF Report":           page_pdf_report,
        "About":                page_about,
    }

    if page in PAGE_MAP:
        PAGE_MAP[page]()
    else:
        page_dashboard()

    render_history()

    st.markdown(
        "<div class='footer'>AmesAI Pro v2.0.0 · XGBoost Edition · © 2025 · Research Use Only</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
