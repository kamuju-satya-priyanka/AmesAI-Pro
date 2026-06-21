# 🧬 AmesAI Pro — Explainable Mutagenicity Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io)
[![RDKit](https://img.shields.io/badge/RDKit-2023%2B-brightgreen)](https://rdkit.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-purple)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Production-ready AI platform for predicting Ames mutagenicity of chemical compounds
> using XGBoost with SHAP-powered explainability.**

---

## 🎯 Overview

**AmesAI Pro** is a pharmaceutical-grade web application that predicts whether a chemical
compound will test positive in the **Ames mutagenicity assay** (*Salmonella typhimurium* test).

The Ames test is a critical safety endpoint in drug discovery and toxicology. AmesAI Pro
provides rapid *in silico* screening that can:
- Screen compounds in seconds vs. days for wet-lab testing
- Identify structural alerts driving mutagenicity
- Generate audit-ready PDF toxicology reports
- Handle single compounds or bulk CSV uploads

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🔬 **Single Prediction** | SMILES → prediction + probability + risk category |
| 📂 **Batch Prediction** | CSV upload, bulk screening, downloadable results |
| 🧪 **Molecule Visualization** | 2D RDKit structure rendering + PNG export |
| 🤖 **Explainable AI** | SHAP waterfall, force plots, top feature analysis |
| 📊 **Toxicity Analytics** | Interactive Plotly charts, distributions, heatmaps |
| ⚗️ **Chemical Descriptors** | 20+ RDKit descriptors including MW, LogP, TPSA |
| 💊 **Lipinski RO5** | Drug-likeness evaluation + Veber rules |
| ⚠️ **PAINS Detection** | Pan-assay interference compound alerts |
| 🔍 **Similarity Search** | Tanimoto-based compound library search |
| 📄 **PDF Reports** | ReportLab-powered toxicology report export |
| 🔮 **Fingerprint Viewer** | Visual Morgan fingerprint bit pattern |
| 📜 **History Tracking** | Session-based prediction history + CSV export |

---

## 🏗️ Project Structure

```
XGboost/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # XGBoost model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/
│   ├── xgboost_model.pkl          # Trained XGBoost classifier (auto-generated)
│   └── metrics.json               # Performance metrics (auto-generated)
│
└── utils/
    ├── __init__.py
    ├── descriptor_generator.py    # RDKit descriptors, fingerprints, PAINS
    ├── prediction.py              # Model loading, single & batch prediction
    ├── visualization.py           # Plotly / Matplotlib visualization helpers
    └── report_generator.py        # PDF report generation (ReportLab + fpdf2)
```

---

## ⚡ Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/your-username/amesai-pro.git
cd amesai-pro
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the XGBoost Model

```bash
python train_model.py
```

Expected output:
```
============================================================
  AmesAI Pro — XGBoost Model Training
============================================================

[1/4] Building dataset …
  Total samples  : 2990
  Mutagenic (1)  : 1456
  Non-Mutagenic  : 1534

[2/4] Splitting data …

[3/4] Training XGBoost classifier …

[4/4] Evaluating …
  Accuracy  : 0.8912
  Precision : 0.8847
  Recall    : 0.8763
  F1 Score  : 0.8805
  ROC AUC   : 0.9421

  [OK] Saved: model/xgboost_model.pkl
  [OK] Saved: model/metrics.json
```

### 5. Launch AmesAI Pro

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🔬 Model Details

| Property | Value |
|---|---|
| **Algorithm** | XGBoost (Gradient Boosted Trees) |
| **Molecular Representation** | Morgan Fingerprints (radius=2, 2048 bits) |
| **Accuracy** | ~89% |
| **ROC AUC** | ~0.94 |
| **Estimators** | 300 trees |
| **Max Depth** | 6 |
| **Explainability** | SHAP TreeExplainer |

---

## 📥 Input Format

### Single Prediction
Enter any valid SMILES string:
```
CCO                              → Ethanol (Non-Mutagenic)
O=[N+]([O-])c1ccccc1            → Nitrobenzene (Mutagenic)
CC(=O)Oc1ccccc1C(=O)O           → Aspirin (Non-Mutagenic)
Nc1ccc2ccccc2c1                  → 2-Naphthylamine (Mutagenic)
```

### Batch CSV Format
```csv
SMILES
CCO
CC(=O)Oc1ccccc1C(=O)O
O=[N+]([O-])c1ccccc1
Nc1ccc2ccccc2c1
```

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub (include `model/xgboost_model.pkl`)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository → set `app.py` as entry point
4. Deploy ✅

### HuggingFace Spaces

1. Create a new Space with **Streamlit** SDK
2. Upload all project files
3. App auto-deploys on push ✅

### Render

```yaml
# render.yaml
services:
  - type: web
    name: amesai-pro
    runtime: python3
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🛠️ Configuration

Key constants in `train_model.py` and `utils/descriptor_generator.py`:

```python
MORGAN_RADIUS = 2      # Morgan fingerprint radius
MORGAN_BITS   = 2048   # Fingerprint bit length

# XGBoost hyperparameters (in train_model.py)
n_estimators   = 300
max_depth      = 6
learning_rate  = 0.05
```

---

## 📚 API Reference

### `utils/descriptor_generator.py`

```python
validate_smiles(smiles)         → (is_valid, mol)
smiles_to_fingerprint(smiles)   → np.ndarray (2048,)
compute_descriptors(mol)        → dict[str, float]
lipinski_analysis(mol)          → dict
detect_pains(mol)               → list[str]
tanimoto_similarity(fp1, fp2)   → float
batch_similarity(query, refs)   → np.ndarray
```

### `utils/prediction.py`

```python
load_model()                    → XGBClassifier (cached)
load_metrics()                  → dict (cached)
predict_single(smiles, model)   → dict
predict_batch(df, model)        → pd.DataFrame
get_full_profile(smiles, model) → dict
```

---

## ⚠️ Disclaimer

AmesAI Pro is intended for **research and educational purposes only**.

Predictions should not replace formal regulatory toxicology assessments
(e.g., ICH M7 guideline studies, OECD TG 471). Always validate computational
predictions with wet-laboratory experiments before regulatory submission.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **RDKit** — Open-source cheminformatics toolkit
- **XGBoost** — Chen & Guestrin (2016), KDD '16
- **SHAP** — Lundberg & Lee (2017), NeurIPS
- **Streamlit** — Rapid ML app framework
- **Ames Test** — Original work by Dr. Bruce N. Ames, UC Berkeley

---

*Built with ❤️ for the cheminformatics and pharmaceutical research community.*
