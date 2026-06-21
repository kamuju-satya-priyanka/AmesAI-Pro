"""
descriptor_generator.py
========================
Compute RDKit molecular descriptors and Morgan fingerprints.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048


def validate_smiles(smiles: str) -> tuple[bool, object | None]:
    """Validate a SMILES string. Returns (is_valid, mol_or_None)."""
    if not smiles or not smiles.strip():
        return False, None
    mol = Chem.MolFromSmiles(smiles.strip())
    return (mol is not None), mol


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert SMILES → Morgan fingerprint vector (float32)."""
    valid, mol = validate_smiles(smiles)
    if not valid or mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def mol_to_fingerprint(mol) -> np.ndarray:
    """Convert RDKit mol object → Morgan fingerprint."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def compute_descriptors(mol) -> dict:
    """Compute a comprehensive set of molecular descriptors."""
    return {
        "Molecular Weight (Da)": round(Descriptors.MolWt(mol), 3),
        "Exact Mol Wt (Da)":     round(Descriptors.ExactMolWt(mol), 4),
        "LogP (Crippen)":        round(Descriptors.MolLogP(mol), 3),
        "TPSA (Å²)":             round(Descriptors.TPSA(mol), 3),
        "H-bond Donors":         int(rdMolDescriptors.CalcNumHBD(mol)),
        "H-bond Acceptors":      int(rdMolDescriptors.CalcNumHBA(mol)),
        "Rotatable Bonds":       int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Aromatic Rings":        int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Ring Count":            int(rdMolDescriptors.CalcNumRings(mol)),
        "Heavy Atom Count":      int(mol.GetNumHeavyAtoms()),
        "Fraction Csp3":         round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 4),
        "Stereo Centers":        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "Heteroatom Count":      int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "Aliphatic Rings":       int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "Aliphatic Carbocycles": int(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)),
        "Aromatic Hetereocycles":int(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)),
        "Num Atoms":             int(mol.GetNumAtoms()),
        "Num Bonds":             int(mol.GetNumBonds()),
        "Molar Refractivity":    round(Descriptors.MolMR(mol), 3),
        "BalabanJ":              round(Descriptors.BalabanJ(mol), 4) if mol.GetNumBonds() > 0 else 0,
    }


def lipinski_analysis(mol) -> dict:
    """Evaluate Lipinski Rule of Five compliance."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = []
    if mw   > 500: violations.append("MW > 500 Da")
    if logp  > 5:  violations.append("LogP > 5")
    if hbd  >  5:  violations.append("HBD > 5")
    if hba  > 10:  violations.append("HBA > 10")

    drug_like = len(violations) == 0

    # Veber rules
    veber_ok = (rb <= 10) and (tpsa <= 140)

    return {
        "MW":              round(mw,   3),
        "LogP":            round(logp, 3),
        "HBD":             hbd,
        "HBA":             hba,
        "RotBonds":        rb,
        "TPSA":            round(tpsa, 3),
        "Violations":      violations,
        "DrugLikeable":    drug_like,
        "VeberCompliant":  veber_ok,
        "NumViolations":   len(violations),
    }


def detect_pains(mol) -> list[str]:
    """Detect PAINS (Pan-assay interference compounds) alerts."""
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        matches = catalog.GetMatches(mol)
        return [m.GetDescription() for m in matches]
    except Exception:
        return []


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto coefficient between two binary fingerprint arrays."""
    a = np.sum(fp1)
    b = np.sum(fp2)
    c = np.sum(fp1 * fp2)
    denom = a + b - c
    return float(c / denom) if denom > 0 else 0.0


def batch_similarity(query_fp: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """Vectorised Tanimoto against a reference matrix. Returns similarity scores."""
    q = query_fp.astype(np.float32)
    R = ref_fps.astype(np.float32)
    c = R @ q                              # dot products
    a = np.sum(q)
    b = R.sum(axis=1)
    denom = a + b - c
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, c / denom, 0.0)
    return sims
