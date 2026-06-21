"""
visualization.py
================
All Plotly / RDKit / Matplotlib visualization helpers for AmesAI Pro.
"""
from __future__ import annotations

import base64
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit.Chem import Draw

warnings.filterwarnings("ignore")

# ── Color Palette ──────────────────────────────────────────────────────────────
CYAN    = "#00D4FF"
PURPLE  = "#7B2FBE"
GREEN   = "#00E676"
RED     = "#FF1744"
YELLOW  = "#FFD600"
BG_DARK = "#0A0E1A"
BG_CARD = "#111827"
BORDER  = "#1E293B"

PLOTLY_DARK = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_CARD,
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="#1E293B", bordercolor=CYAN, font_color="#E2E8F0"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  Molecule Rendering
# ══════════════════════════════════════════════════════════════════════════════

def mol_to_svg(mol, width: int = 400, height: int = 300, dark: bool = True) -> str:
    """Render RDKit mol to SVG data-URI (base64-encoded)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts   = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.padding = 0.12
    if dark:
        opts.backgroundColour = (10/255, 14/255, 26/255, 1)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def mol_to_png_bytes(mol, width: int = 600, height: int = 400) -> bytes:
    """Render molecule to PNG bytes for download."""
    from rdkit.Chem import Draw
    img = Draw.MolToImage(mol, size=(400,400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Risk / Probability Gauges
# ══════════════════════════════════════════════════════════════════════════════

def make_gauge(mut_prob: float) -> go.Figure:
    """Animated toxicity risk gauge."""
    risk_pct = mut_prob * 100
    if risk_pct < 30:
        color, label = GREEN,  "LOW RISK ✅"
    elif risk_pct < 60:
        color, label = YELLOW, "MODERATE ⚠️"
    else:
        color, label = RED,    "HIGH RISK 🚨"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "JetBrains Mono"}},
        title={"text": f"Mutagenicity Risk<br><span style='font-size:13px;color:{color};font-weight:700'>{label}</span>",
               "font": {"size": 15, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155",
                     "tickfont": {"color": "#64748B"}, "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": BG_CARD,
            "bordercolor": BORDER,
            "borderwidth": 2,
            "steps": [
                {"range": [0,  30], "color": "rgba(0,230,118,0.08)"},
                {"range": [30, 60], "color": "rgba(255,214,0,0.08)"},
                {"range": [60, 100], "color": "rgba(255,23,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.80,
                "value": risk_pct,
            },
        },
    ))
    fig.update_layout(height=320, **PLOTLY_DARK)
    return fig


def make_probability_bar(mut_prob: float, safe_prob: float) -> go.Figure:
    """Stacked horizontal probability breakdown bar."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[safe_prob * 100], y=["Probability"],
        orientation="h", name="Non-Mutagenic",
        marker_color=GREEN,
        text=[f"Non-Mutagenic: {safe_prob*100:.1f}%"],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#0A0E1A", size=13, family="Inter"),
    ))
    fig.add_trace(go.Bar(
        x=[mut_prob * 100], y=["Probability"],
        orientation="h", name="Mutagenic",
        marker_color=RED,
        text=[f"Mutagenic: {mut_prob*100:.1f}%"],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=13, family="Inter"),
    ))
    fig.update_layout(
        barmode="stack", height=110,
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="right", x=1, font=dict(size=11)),
        **PLOTLY_DARK,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard Charts
# ══════════════════════════════════════════════════════════════════════════════

def make_confusion_matrix(cm: list) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["Non-Mutagenic", "Mutagenic"]
    z = np.array(cm)
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="<b>%{text}</b>",
        textfont=dict(size=22, color="white"),
        colorscale=[[0, BG_CARD], [0.5, PURPLE], [1.0, CYAN]],
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16, color="#E2E8F0")),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
        **PLOTLY_DARK,
    )
    return fig


def make_roc_curve(fpr: list, tpr: list, auc: float) -> go.Figure:
    """ROC curve with AUC fill."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"XGBoost (AUC = {auc:.3f})",
        line=dict(color=CYAN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Baseline",
        line=dict(color="#334155", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(size=16, color="#E2E8F0")),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        **PLOTLY_DARK,
    )
    return fig


def make_pr_curve(precision: list, recall: list) -> go.Figure:
    """Precision-Recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name="Precision-Recall",
        line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(123,47,190,0.08)",
    ))
    fig.update_layout(
        title=dict(text="Precision-Recall Curve", font=dict(size=16, color="#E2E8F0")),
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
        **PLOTLY_DARK,
    )
    return fig


def make_feature_importance(importances: list, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of top-N XGBoost feature importances."""
    imp   = np.array(importances)
    idx   = np.argsort(imp)[-top_n:]
    vals  = imp[idx]
    names = [f"Bit_{i}" for i in idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names,
        orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, PURPLE], [1, CYAN]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importances", font=dict(size=16, color="#E2E8F0")),
        xaxis_title="Importance Score",
        height=600,
        yaxis=dict(tickfont=dict(size=10, family="JetBrains Mono")),
        **PLOTLY_DARK,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Analytics Charts
# ══════════════════════════════════════════════════════════════════════════════

def make_distribution_pie(n_mutagenic: int, n_non: int) -> go.Figure:
    """Donut chart: Mutagenic vs Non-Mutagenic distribution."""
    fig = go.Figure(go.Pie(
        labels=["Mutagenic", "Non-Mutagenic"],
        values=[n_mutagenic, n_non],
        hole=0.6,
        marker=dict(colors=[RED, GREEN], line=dict(color=BG_DARK, width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Dataset Class Distribution", font=dict(size=16, color="#E2E8F0")),
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05),
        **PLOTLY_DARK,
    )
    fig.add_annotation(
        text=f"<b>{n_mutagenic + n_non}</b><br><span style='font-size:11px'>Compounds</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#E2E8F0"),
    )
    return fig


def make_probability_histogram(mut_probs: list) -> go.Figure:
    """Histogram of mutagenicity probability scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mut_probs,
        nbinsx=30,
        name="Mut. Probability",
        marker=dict(
            color=mut_probs,
            colorscale=[[0, GREEN], [0.5, YELLOW], [1.0, RED]],
            line=dict(color=BG_DARK, width=0.8),
        ),
        opacity=0.85,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=YELLOW,
                  annotation_text="Decision Threshold", annotation_position="top")
    fig.update_layout(
        title=dict(text="Mutagenicity Probability Distribution", font=dict(size=16, color="#E2E8F0")),
        xaxis_title="Mutagenicity Probability",
        yaxis_title="Count",
        height=380,
        **PLOTLY_DARK,
    )
    return fig


def make_descriptor_heatmap(desc_df: "pd.DataFrame") -> go.Figure:
    """Correlation heatmap for molecular descriptors."""
    corr = desc_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, RED], [0.5, BG_CARD], [1.0, CYAN]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Descriptor Correlation Heatmap", font=dict(size=16, color="#E2E8F0")),
        height=500,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        **PLOTLY_DARK,
    )
    return fig


def make_radar_chart(desc_values: dict, label: str) -> go.Figure:
    """Spider/radar chart of normalized descriptor values."""
    keys = list(desc_values.keys())[:8]
    vals = [float(desc_values[k]) if desc_values[k] is not None else 0 for k in keys]
    max_v = [max(v, 1e-9) for v in vals]
    norm  = [v / m for v, m in zip(vals, max_v)]
    norm.append(norm[0])
    cats = keys + [keys[0]]

    fig = go.Figure(go.Scatterpolar(
        r=norm, theta=cats, fill="toself",
        fillcolor=f"rgba(0,212,255,0.12)",
        line=dict(color=CYAN, width=2),
        name=label,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_CARD,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER,
                            tickfont=dict(size=9, color="#475569")),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=10, color="#94A3B8")),
        ),
        title=dict(text="Descriptor Profile", font=dict(size=16, color="#E2E8F0")),
        showlegend=False,
        height=400,
        paper_bgcolor=BG_DARK,
        font=dict(family="Inter", color="#94A3B8"),
    )
    return fig


def make_similarity_bar(results: list[dict]) -> go.Figure:
    """Horizontal bar chart for similarity search results."""
    smiles = [r["smiles"][:25] + "…" if len(r["smiles"]) > 25 else r["smiles"] for r in results]
    scores = [r["similarity"] for r in results]
    colors = [GREEN if r.get("label") == "Non-Mutagenic" else RED for r in results]

    fig = go.Figure(go.Bar(
        x=scores, y=smiles,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="<b>%{y}</b><br>Tanimoto: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top Similar Compounds (Tanimoto Similarity)", font=dict(size=16, color="#E2E8F0")),
        xaxis=dict(range=[0, 1], title="Tanimoto Similarity"),
        height=420,
        **PLOTLY_DARK,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def shap_waterfall_plotly(shap_values: np.ndarray, base_value: float,
                           top_n: int = 15) -> go.Figure:
    """Interactive SHAP waterfall chart using Plotly."""
    idx   = np.argsort(np.abs(shap_values))[-top_n:][::-1]
    names = [f"Bit_{i}" for i in idx]
    vals  = shap_values[idx]
    colors = [GREEN if v < 0 else RED for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names,
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color=BG_DARK, width=0.5)),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BORDER, line_width=2)
    fig.update_layout(
        title=dict(text=f"SHAP Waterfall — Top {top_n} Features<br>"
                        f"<span style='font-size:12px;color:#64748B'>Base value: {base_value:.4f}</span>",
                   font=dict(size=15, color="#E2E8F0")),
        xaxis_title="SHAP Value",
        height=500,
        **PLOTLY_DARK,
    )
    return fig


def shap_summary_plotly(shap_matrix: np.ndarray, fp_matrix: np.ndarray,
                         top_n: int = 20) -> go.Figure:
    """Interactive SHAP beeswarm-style summary plot."""
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    idx   = np.argsort(mean_abs)[-top_n:]
    names = [f"Bit_{i}" for i in idx]
    mu    = mean_abs[idx]

    fig = go.Figure(go.Bar(
        x=mu, y=names,
        orientation="h",
        marker=dict(color=mu, colorscale=[[0, PURPLE], [1, CYAN]],
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|", thickness=10,
                                  tickfont=dict(size=9, color="#64748B"))),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"SHAP Feature Importance — Top {top_n}",
                   font=dict(size=15, color="#E2E8F0")),
        xaxis_title="Mean |SHAP Value|",
        height=550,
        **PLOTLY_DARK,
    )
    return fig


def fingerprint_viewer(fp: np.ndarray, width: int = 800) -> go.Figure:
    """Visualise Morgan fingerprint as an interactive bit-pattern heatmap."""
    side = int(np.ceil(np.sqrt(len(fp))))
    padded = np.zeros(side * side)
    padded[: len(fp)] = fp
    grid = padded.reshape(side, side)

    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale=[[0, BG_CARD], [1, CYAN]],
        showscale=False,
        hovertemplate="Row: %{y}, Col: %{x}<br>Bit: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Morgan Fingerprint Bit Pattern ({len(fp)} bits)",
                   font=dict(size=14, color="#E2E8F0")),
        height=350,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        **PLOTLY_DARK,
    )
    return fig
