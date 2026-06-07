"""
Train the ghost-hotel likelihood classifier.

Trains a HistGradientBoostingClassifier on the 25 Sep 2025 scored snapshot using
ONLY the live-scrapeable features defined in pipeline/features.py, and saves an
explainable model bundle to data/models/ghost_clf.joblib.

Labels (user decision — "treat enforcement/priority cases as positives"):
  positive (1) = priority_tier in {"Priority", "Review"}
  negative (0) = priority_tier == "Low"
  Monitor / NaN tiers are dropped (ambiguous middle).
Negatives are down-sampled to 4x positives (random_state=42) for balance.

NOTE: the labels are the heuristic F1-F8 priority tiers, NOT independent council
confirmations. This model approximates "what the heuristic flags using only
live-scrapeable listing-page features" and is a triage aid, not ground truth.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "scored", "scored_listings.parquet")
MODEL_DIR = os.path.join(ROOT, "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ghost_clf.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

RANDOM_STATE = 42
NEG_RATIO = 4  # negatives sampled to NEG_RATIO x positives

NOTE = (
    "Labels are the heuristic F1-F8 priority tiers, NOT independent council "
    "confirmations. This model approximates 'what the heuristic flags using only "
    "live-scrapeable listing-page features'. It is an explainable triage aid for "
    "listings absent from the snapshot, not independent ground truth."
)
TRAINED_ON = "scored_listings.parquet (25 Sep 2025 snapshot)"


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Priority/Review (1) vs Low (0); drop Monitor/NaN; balance negatives."""
    tier = df["priority_tier"]
    pos = df[tier.isin(["Priority", "Review"])].copy()
    neg = df[tier == "Low"].copy()
    pos["y"] = 1
    neg["y"] = 0

    n_take = min(len(neg), NEG_RATIO * len(pos))
    neg = neg.sample(n=n_take, random_state=RANDOM_STATE)

    out = pd.concat([pos, neg], axis=0)
    # Shuffle deterministically so order is not class-blocked.
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    labeled = build_labels(df)
    y = labeled["y"].to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"Labels: {n_pos:,} positives (Priority+Review), {n_neg:,} negatives (Low, 4x-balanced)")

    # Step 2: shared feature matrix.
    X, cat_maps = build_X(labeled)

    # Step 3: stratified 80/20 split + fit.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Split: {len(X_train):,} train / {len(X_test):,} test")

    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Step 4: evaluation on test.
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    clf_report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cm_list = cm.tolist()  # [[tn, fp], [fn, tp]]

    # Step 5: permutation importance on the test set.
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE
    )
    importances = sorted(
        (
            {"feature": feat, "importance": float(imp)}
            for feat, imp in zip(features.FEATURE_ORDER, perm.importances_mean)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )

    # Step 6: per-feature training-median baselines.
    feature_baselines = {
        col: float(np.nanmedian(X_train[col].to_numpy(dtype="float64")))
        for col in features.FEATURE_ORDER
    }

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "confusion_at_0.5": cm_list,
        "classification_report": clf_report,
    }

    # Step 7: save the bundle.
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "cat_maps": cat_maps,
        "feature_order": features.FEATURE_ORDER,
        "feature_baselines": feature_baselines,
        "importances": importances,
        "metrics": metrics,
        "trained_on": TRAINED_ON,
        "note": NOTE,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle -> {MODEL_PATH}")

    # Step 8: metrics.json (metrics + note + top 10 importances).
    metrics_out = dict(metrics)
    metrics_out["note"] = NOTE
    metrics_out["trained_on"] = TRAINED_ON
    metrics_out["top_importances"] = importances[:10]
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    # Step 9: human summary.
    tn, fp = cm_list[0]
    fn, tp = cm_list[1]
    print("\n=== Ghost-hotel classifier summary ===")
    print(f"ROC AUC          : {roc_auc:.4f}")
    print(f"PR AUC (AP)      : {pr_auc:.4f}")
    print(f"Train / Test     : {len(X_train):,} / {len(X_test):,}")
    print(f"Pos / Neg        : {n_pos:,} / {n_neg:,}")
    print(f"Confusion @0.5   : [[tn={tn}, fp={fp}], [fn={fn}, tp={tp}]]")
    print(f"Pos precision    : {clf_report['1']['precision']:.4f}")
    print(f"Pos recall       : {clf_report['1']['recall']:.4f}")
    print(f"Pos f1           : {clf_report['1']['f1-score']:.4f}")
    print("Top 8 features by permutation importance:")
    for d in importances[:8]:
        print(f"  {d['feature']:<28} {d['importance']:.5f}")


def build_X(labeled: pd.DataFrame):
    X, cat_maps = features.build_matrix(labeled)
    assert list(X.columns) == features.FEATURE_ORDER, "feature order drift"
    return X, cat_maps


if __name__ == "__main__":
    main()
