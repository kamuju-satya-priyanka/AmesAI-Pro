"""
AmesAI Pro — XGBoost Model Training Script
==========================================
Trains an XGBoost classifier for Ames mutagenicity prediction.
Saves model artifact + performance metrics JSON.

Author  : Senior AI / Cheminformatics Engineer
Version : 2.0.0 (XGBoost Edition)
"""

from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# ── Ames-positive (mutagenic) SMILES ──────────────────────────────────────────
MUTAGENIC_SMILES = [
    "c1ccc2c(c1)ccc3cccc4cccc2c34",          # Benzo[a]pyrene
    "c1ccc2cc3ccccc3cc2c1",                   # Anthracene
    "Nc1ccc(N)cc1",                           # 4,4'-Diaminobiphenyl
    "Nc1ccc2ccccc2c1",                        # 2-Naphthylamine
    "c1ccc2ccccc2c1",                         # Naphthalene
    "Nc1ccccc1N",                             # o-Phenylenediamine
    "CC1=CC=C(C=C1)N",                        # 4-Methylaniline
    "Nc1ccc([N+](=O)[O-])cc1",               # 4-Nitroaniline
    "O=Cc1ccccc1",                            # Benzaldehyde
    "ClCCCl",                                 # 1,3-Dichloropropane
    "BrCCBr",                                 # 1,2-Dibromoethane
    "C1CCC(CC1)N",                            # Cyclohexylamine
    "c1ccc(cc1)N=Nc1ccccc1",                 # Azobenzene
    "Nc1cccc2ccccc12",                        # 1-Naphthylamine
    "O=C(O)c1ccccc1N",                        # Anthranilic acid
    "CC(=O)Nc1ccccc1",                        # Acetanilide
    "O=[N+]([O-])c1ccccc1",                  # Nitrobenzene
    "ClCCl",                                  # Methylene chloride
    "BrCC(=O)O",                              # Bromoacetic acid
    "C(CCl)CCl",                              # 1,4-Dichlorobutane
    "Nc1ccc(cc1)c1ccc(N)cc1",               # Benzidine
    "c1ccc(Nc2ccccc2)cc1",                   # Diphenylamine
    "c1ccc2[nH]ccc2c1",                      # Indole
    "O=C1NC(=O)c2ccccc21",                   # Isatin
    "CC1=C(C=NO)C(C)(C)CC1",                # Nitroso compound
    "Nc1nc2ccccc2s1",                        # 2-Aminobenzothiazole
    "O=c1[nH]c(=O)c2ccccc2[nH]1",          # Quinoxaline dione
    "Cc1nc2ccccc2s1",                        # 2-Methylbenzothiazole
    "O=[N+]([O-])c1ccc(N)cc1",             # 4-Nitroaniline isomer
    "ClCCCCCl",                              # 1,5-Dichloropentane
    "O=N/N=N/c1ccccc1",                     # Phenyl azide
    "c1ccc(CC#N)cc1",                        # Benzyl cyanide
    "IC(I)(I)I",                             # Tetraiodomethane
    "Nc1ccc(F)cc1",                          # 4-Fluoroaniline
    "c1cc2ccc3cccc4ccc(c1)c2c34",           # Pyrene
    "O=C1c2ccccc2C(=O)c2ccccc21",          # Anthraquinone
    "Clc1ccc(Cl)cc1",                        # 1,4-Dichlorobenzene
    "OC(=O)c1ccc([N+](=O)[O-])cc1",        # 4-Nitrobenzoic acid
    "Brc1ccccc1",                             # Bromobenzene
    "Clc1ccccc1Cl",                           # 1,2-Dichlorobenzene
    "CC(=O)c1ccc([N+](=O)[O-])cc1",         # 4-Nitroacetophenone
    "Nc1ccc(O)cc1",                           # 4-Aminophenol
    "c1ccc(N2CCCC2)cc1",                      # N-Phenylpyrrolidine
    "O=c1cc[nH]c(=O)[nH]1",                 # Uracil (mutagenic in some strains)
    "ClCC(Cl)Cl",                             # 1,1,2-Trichloroethane
    "C(=O)(O)CC(=O)O",                        # Malonic acid
    "CCOC(=O)c1ccc([N+](=O)[O-])cc1",       # Ethyl 4-nitrobenzoate
    "O=C(Cl)c1ccccc1",                        # Benzoyl chloride
    "Clc1cccc(Cl)c1Cl",                       # 1,2,3-Trichlorobenzene
    "O=S(=O)(O)c1ccccc1N",                   # Sulfanilic acid
    "Nc1ccc2cc3ccc(N)cc3cc2c1",              # 3,3'-Diaminobenzidine
    "BrCCCBr",                                # 1,3-Dibromopropane
    "N#Cc1ccccc1",                            # Benzonitrile
    "CC1=CC(=O)C=CC1=O",                     # Methyl-1,4-benzoquinone
    "O=[N+]([O-])C1=CC=CC=C1Cl",            # 2-Chloronitrobenzene
    "Nc1ccc(Cl)cc1",                          # 4-Chloroaniline
]

# ── Ames-negative (non-mutagenic) SMILES ─────────────────────────────────────
NON_MUTAGENIC_SMILES = [
    "CCO",                                    # Ethanol
    "CC(C)O",                                 # Isopropanol
    "OCC(O)CO",                              # Glycerol
    "CC(=O)O",                               # Acetic acid
    "OC(=O)CCC(=O)O",                       # Succinic acid
    "CC(=O)OCC",                             # Ethyl acetate
    "CCOCCO",                                # 2-Ethoxyethanol
    "CCCC",                                  # Butane
    "CCCCO",                                 # 1-Butanol
    "CC(O)CC",                               # 2-Butanol
    "OC(=O)c1ccccc1",                       # Benzoic acid
    "CC(=O)Oc1ccccc1C(=O)O",              # Aspirin
    "CC12CCC(CC1)CC2",                      # Decalin
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",        # Citric acid
    "CCCCCC",                               # Hexane
    "OCC(O)C(O)C(O)C(O)CO",              # Sorbitol
    "CC(=O)NC(CO)CO",                      # Acetamide derivative
    "OC(=O)CCCCC(=O)O",                   # Adipic acid
    "C(CO)N",                               # Ethanolamine
    "CC(C)CC(C)(C)C",                       # 2,2,4-trimethylpentane
    "CCCCCCC",                              # Heptane
    "CC(C)(C)O",                            # t-Butanol
    "OC1CCCCC1",                            # Cyclohexanol
    "CCOC(=O)CC(=O)OCC",                   # Diethyl malonate
    "CC(=O)OC",                             # Methyl acetate
    "O=C1CCCCC1",                           # Cyclohexanone
    "CCC(=O)O",                             # Propionic acid
    "OC(=O)c1ccc(O)cc1",                   # 4-Hydroxybenzoic acid
    "CC(C)CCC(C)(C)C",                      # 2,2,4,4-tetramethylpentane
    "CCCCCCCC",                             # Octane
    "O=C(O)CCC(=O)O",                       # Glutaric acid
    "CC(=O)CCCC(=O)O",                      # Levulinic acid
    "OCCO",                                 # Ethylene glycol
    "OCC(CO)(CO)CO",                        # Pentaerythritol
    "CCOC(=O)OCC",                          # Diethyl carbonate
    "CC(O)C(O)CO",                          # 1,2,3-Butanetriol
    "O=C(O)CCC(O)=O",                       # Malic acid
    "OC(=O)CCCC(=O)O",                      # Glutaric acid
    "CC1CCCCC1",                             # Methylcyclohexane
    "O=C(O)CC(=O)O",                         # Malonic acid (non-mutagenic label)
    "CCCCCO",                               # 1-Pentanol
    "CCC(C)O",                              # 2-Butanol isomer
    "CC(C)C(C)O",                           # 3-Methyl-2-butanol
    "O=CCCCC=O",                             # Glutaraldehyde
    "OC(CO)CO",                             # Triethanolamine
    "CC1=CC=CC=C1",                          # Toluene
    "OC(=O)c1cccc(O)c1",                    # 3-Hydroxybenzoic acid
    "CC(=O)c1ccccc1",                        # Acetophenone
    "CCN(CC)CC",                             # Triethylamine
    "CN(C)C",                               # Trimethylamine
    "CNCCO",                                # N-Methylethanolamine
    "OC(=O)c1ccc(C(=O)O)cc1",             # Terephthalic acid
    "CCOC(=O)c1ccccc1",                     # Ethyl benzoate
    "O=C(O)c1ccc(O)c(O)c1",               # 3,4-Dihydroxybenzoic acid
    "CC(=O)Nc1ccc(O)cc1",                   # Paracetamol (acetaminophen)
    "OC(=O)CC(N)C(=O)O",                    # Aspartic acid
    "NC(CC(=O)O)C(=O)O",                    # Aspartic acid
    "NC(Cc1ccccc1)C(=O)O",                  # Phenylalanine
    "OCC(N)C(=O)O",                          # Serine
    "NC(CCCNC(=N)N)C(=O)O",                 # Arginine
    "NC(CS)C(=O)O",                          # Cysteine
]

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Convert SMILES → Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def build_dataset(augment_factor: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Build augmented training dataset."""
    X, y = [], []
    np.random.seed(42)

    all_smiles = (
        [(s, 1) for s in MUTAGENIC_SMILES] +
        [(s, 0) for s in NON_MUTAGENIC_SMILES]
    )

    for smiles, label in all_smiles:
        fp = smiles_to_fp(smiles)
        if fp is None:
            continue
        X.append(fp)
        y.append(label)

        # Augment with small bit-level perturbations
        for _ in range(augment_factor):
            aug = fp.copy()
            n_flip = np.random.randint(1, 15)
            idx = np.random.choice(MORGAN_BITS, n_flip, replace=False)
            aug[idx] = 1 - aug[idx]
            X.append(aug)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_and_save(output_dir: str = "model"):
    """Train XGBoost model and persist all artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    model_path   = os.path.join(output_dir, "xgboost_model.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")

    print("=" * 60)
    print("  AmesAI Pro — XGBoost Model Training")
    print("=" * 60)

    print("\n[1/4] Building dataset …")
    X, y = build_dataset(augment_factor=25)
    n_total = len(X)
    n_mut   = int(y.sum())
    n_non   = int((y == 0).sum())
    print(f"  Total samples  : {n_total}")
    print(f"  Mutagenic (1)  : {n_mut}")
    print(f"  Non-Mutagenic  : {n_non}")
    print(f"  Features       : {X.shape[1]}")

    print("\n[2/4] Splitting data …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[3/4] Training XGBoost classifier …")
    scale_pos_weight = n_non / n_mut if n_mut > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("\n[4/4] Evaluating …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  CM        : {cm.tolist()}")

    feat_imp = model.feature_importances_.tolist()
    top_idx  = np.argsort(model.feature_importances_)[::-1][:50].tolist()

    metrics = {
        "accuracy":              round(float(acc), 4),
        "precision":             round(float(prec), 4),
        "recall":                round(float(rec), 4),
        "f1_score":              round(float(f1), 4),
        "roc_auc":               round(float(auc), 4),
        "confusion_matrix":      cm.tolist(),
        "fpr":                   [round(v, 6) for v in fpr.tolist()],
        "tpr":                   [round(v, 6) for v in tpr.tolist()],
        "thresholds":            [round(v, 6) for v in thresholds.tolist()],
        "precision_curve":       [round(v, 6) for v in prec_curve.tolist()],
        "recall_curve":          [round(v, 6) for v in rec_curve.tolist()],
        "pr_thresholds":         [round(v, 6) for v in pr_thresholds.tolist()],
        "feature_importances":   feat_imp,
        "top_feature_indices":   top_idx,
        "n_train":               int(X_train.shape[0]),
        "n_test":                int(X_test.shape[0]),
        "n_total":               n_total,
        "n_mutagenic":           n_mut,
        "n_non_mutagenic":       n_non,
        "model_type":            "XGBoost",
        "n_estimators":          300,
        "morgan_radius":         MORGAN_RADIUS,
        "morgan_bits":           MORGAN_BITS,
    }

    joblib.dump(model, model_path)
    print(f"\n  [OK] Saved: {model_path}")

    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  [OK] Saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("  Training complete! Ready for AmesAI Pro.")
    print("=" * 60)
    return model, metrics


if __name__ == "__main__":
    train_and_save()
