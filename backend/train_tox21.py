"""Train endpoint-specific educational classifiers on the public Tox21 benchmark.

This trains models for specific Tox21 assay labels, NOT a universal "toxicity" score.
Review the dataset's terms, endpoint meanings, split strategy, and limitations before
using outputs. Models trained here are research baselines and are not safety tools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
FINGERPRINT_BITS = 2048
RADIUS = 2
SEED = 42


def featurize(smiles: str) -> np.ndarray | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, RADIUS, nBits=FINGERPRINT_BITS
    )
    bits = np.zeros((FINGERPRINT_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    return bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "tox21.csv.gz")
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    args.data.parent.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        print(f"Downloading public Tox21 benchmark from {DATA_URL}")
        urlretrieve(DATA_URL, args.data)

    frame = pd.read_csv(args.data)
    if "smiles" not in frame.columns:
        raise ValueError("Expected a 'smiles' column in the Tox21 CSV.")
    target_columns = [column for column in frame.columns if column != "smiles"]
    if not target_columns:
        raise ValueError("No assay label columns were found.")

    print(f"Rows: {len(frame)}; assay endpoints: {len(target_columns)}")
    features, valid_indices = [], []
    for index, smiles in enumerate(frame["smiles"]):
        vector = featurize(smiles)
        if vector is not None:
            features.append(vector)
            valid_indices.append(index)
    X = np.asarray(features, dtype=np.float32)
    valid = frame.iloc[valid_indices].reset_index(drop=True)
    metrics = {}

    for endpoint in target_columns:
        y_raw = pd.to_numeric(valid[endpoint], errors="coerce").to_numpy()
        known = np.isfinite(y_raw) & np.isin(y_raw, [0, 1])
        X_endpoint, y = X[known], y_raw[known].astype(int)
        if len(np.unique(y)) != 2 or min(np.bincount(y)) < 5:
            print(f"Skipping {endpoint}: insufficient examples in both classes")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_endpoint,
            y,
            test_size=args.test_size,
            random_state=SEED,
            stratify=y,
        )
        model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        result = {
            "endpoint": endpoint,
            "n_known": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "positive_fraction": float(y.mean()),
            "roc_auc_random_split": float(roc_auc_score(y_test, probabilities)),
            "average_precision_random_split": float(
                average_precision_score(y_test, probabilities)
            ),
            "balanced_accuracy_random_split": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "warning": (
                "Random stratified split is a baseline, not external validation; "
                "chemical-series leakage may inflate performance."
            ),
        }
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in endpoint)
        joblib.dump(
            {
                "model": model,
                "endpoint": endpoint,
                "feature": "RDKit Morgan fingerprint",
                "radius": RADIUS,
                "n_bits": FINGERPRINT_BITS,
                "seed": SEED,
                "dataset": "Tox21",
                "metrics": result,
            },
            args.models / f"{safe_name}.joblib",
        )
        metrics[endpoint] = result
        print(
            f"{endpoint}: ROC-AUC={result['roc_auc_random_split']:.3f}; "
            f"PR-AUC={result['average_precision_random_split']:.3f}"
        )

    report = {
        "dataset": "Tox21",
        "source": DATA_URL,
        "featurization": {"type": "RDKit Morgan fingerprint", "radius": RADIUS, "bits": FINGERPRINT_BITS},
        "split": "random stratified; baseline only",
        "seed": SEED,
        "metrics": metrics,
        "limitations": [
            "Tox21 endpoints are assay-specific biological activity labels, not universal toxicity.",
            "Random splits may leak related chemical scaffolds across train and test sets.",
            "Metrics are not external validation or regulatory qualification.",
            "Do not use for human/animal safety decisions or to claim a compound is safe.",
        ],
    }
    (args.models / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved models and evaluation report under {args.models.resolve()}")


if __name__ == "__main__":
    main()
