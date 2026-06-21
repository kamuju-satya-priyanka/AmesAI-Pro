"""
prediction.py
=============
Model loading, single-molecule, and batch prediction utilities.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.descriptor_generator import (
    smiles_to_fingerprint,
    validate_smiles,
    lipinski_analysis,
    detect_pains,
    compute_descriptors,
)

warnings.filterwarnings("ignore")

MODEL_PATH   = Path(__file__).parent.parent / "model" / "xgboost_model.pkl"
METRICS_PATH = Path(__file__).parent.parent / "model" / "metrics.json"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load XGBoost model (cached across sessions)."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Load pre-computed performance metrics JSON."""
    import json
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def risk_category(mut_prob: float) -> tuple[str, str]:
    """Return (risk_label, color_hex) based on mutagenicity probability."""
    if mut_prob < 0.30:
        return "Low Risk", "#00E676"
    elif mut_prob < 0.60:
        return "Moderate Risk", "#FFD600"
    else:
        return "High Risk", "#FF1744"


def predict_single(smiles: str, model) -> Optional[dict]:
    """
    Run single-molecule prediction.

    Returns
    -------
    dict with keys:
        class, label, probability, mut_prob, safe_prob,
        risk_label, risk_color, fp, fp_2d
    """
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None

    fp_2d = fp.reshape(1, -1)
    pred_class = int(model.predict(fp_2d)[0])
    pred_proba = model.predict_proba(fp_2d)[0]
    mut_prob   = float(pred_proba[1])
    safe_prob  = float(pred_proba[0])
    probability = float(pred_proba[pred_class])
    risk_lbl, risk_col = risk_category(mut_prob)

    return {
        "class":       pred_class,
        "label":       "Mutagenic" if pred_class == 1 else "Non-Mutagenic",
        "probability": probability,
        "mut_prob":    mut_prob,
        "safe_prob":   safe_prob,
        "risk_label":  risk_lbl,
        "risk_color":  risk_col,
        "fp":          fp,
        "fp_2d":       fp_2d,
        "confidence":  round(probability * 100, 1),
    }


def predict_batch(df: pd.DataFrame, model, smiles_col: str = "SMILES") -> pd.DataFrame:
    """
    Batch prediction on a DataFrame with a SMILES column.

    Returns the DataFrame augmented with Prediction, Probability, Risk columns.
    """
    results = []
    for smi in df[smiles_col]:
        smi = str(smi).strip()
        valid, mol = validate_smiles(smi)
        if not valid or mol is None:
            results.append({
                "SMILES":      smi,
                "Valid":       False,
                "Prediction":  "Invalid SMILES",
                "Probability": None,
                "Mut_Prob":    None,
                "Risk":        "N/A",
            })
            continue

        pred = predict_single(smi, model)
        if pred is None:
            results.append({
                "SMILES":      smi,
                "Valid":       False,
                "Prediction":  "Prediction Error",
                "Probability": None,
                "Mut_Prob":    None,
                "Risk":        "N/A",
            })
        else:
            results.append({
                "SMILES":      smi,
                "Valid":       True,
                "Prediction":  pred["label"],
                "Probability": round(pred["probability"] * 100, 2),
                "Mut_Prob":    round(pred["mut_prob"] * 100, 2),
                "Risk":        pred["risk_label"],
            })

    return pd.DataFrame(results)


def get_full_profile(smiles: str, model) -> Optional[dict]:
    """
    Full molecular profile: prediction + descriptors + Lipinski + PAINS.
    Used by the PDF report generator and multi-molecule comparison.
    """
    valid, mol = validate_smiles(smiles)
    if not valid or mol is None:
        return None

    pred = predict_single(smiles, model)
    if pred is None:
        return None

    desc   = compute_descriptors(mol)
    lipi   = lipinski_analysis(mol)
    pains  = detect_pains(mol)

    return {
        "smiles":      smiles,
        "prediction":  pred,
        "descriptors": desc,
        "lipinski":    lipi,
        "pains":       pains,
        "mol":         mol,
    }
