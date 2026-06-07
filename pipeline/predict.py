"""
Inference for the ghost-hotel ML classifier.

Loads the joblib bundle produced by train_model.py and scores listings using ONLY
the live-scrapeable features defined in features.py. Provides:

  * load_bundle()   - load + validate the trained bundle
  * predict_one()   - score a single listing dict, with a local occlusion-based
                      explanation of which feature values pushed risk up/down
  * predict_batch() - vectorised whole-frame probability scoring

The bundle contract (see train_model.py) is a dict with keys:
  "model", "cat_maps", "feature_order", "feature_baselines",
  "importances", "metrics", "trained_on", "note".
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import joblib

from . import features  # shared column spec / build_matrix


DEFAULT_BUNDLE_PATH = "data/models/ghost_clf.joblib"

# Risk tiers — must match the dashboard thresholds.
_TIER_THRESHOLDS = [
    (0.90, "Priority"),
    (0.75, "Review"),
    (0.50, "Monitor"),
]


def _tier_for(prob: float) -> str:
    """Map a probability to its dashboard risk tier."""
    for threshold, name in _TIER_THRESHOLDS:
        if prob >= threshold:
            return name
    return "Low"


def load_bundle(path: str = DEFAULT_BUNDLE_PATH) -> dict:
    """
    Load the trained joblib bundle.

    Raises FileNotFoundError (with a fix-it hint) if the model hasn't been
    trained yet.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. "
            "Train the model first by running:\n"
            "    python3 pipeline/train_model.py"
        )
    return joblib.load(path)


def _build_X(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """build_matrix + reindex to the bundle's exact training feature order."""
    X, _ = features.build_matrix(df, cat_maps=bundle["cat_maps"])
    # Defensive: guarantee identical column order/set to what the model saw.
    # Cast to float so occlusion can write median baselines into the int-encoded
    # cat_* columns without dtype-incompatibility warnings (the model is
    # dtype-agnostic about the numeric matrix).
    return X.reindex(columns=bundle["feature_order"]).astype("float64")


def predict_one(features: dict, bundle: dict | None = None) -> dict:
    """
    Score a single listing.

    Parameters
    ----------
    features : dict
        Raw listing fields keyed by features.py column names, e.g.
        {"price": "$120.00", "room_type": "Entire home/apt", "accommodates": 4}.
        Missing keys are fine — the model handles NaN.
    bundle : dict, optional
        A loaded bundle. If None, load_bundle() is called with the default path.

    Returns
    -------
    dict with keys: "prob", "tier", "top_factors", "note".
        top_factors is the 5 most influential features by |contribution|, each
        {"feature", "value", "contribution"} where a positive contribution means
        the listing's actual value for that feature pushed risk UP relative to the
        training-median baseline.
    """
    # NOTE: shadows the module name within this function; we only need the
    # build_matrix module via the bundle path, accessed below by module alias.
    raw_features = features
    if bundle is None:
        bundle = load_bundle()

    feature_order = bundle["feature_order"]
    baselines = bundle["feature_baselines"]
    model = bundle["model"]

    # Build the 1-row model matrix.
    df = pd.DataFrame([raw_features])
    X = _build_X(df, bundle)

    base_prob = float(model.predict_proba(X)[0, 1])

    # Local explanation via occlusion: replace one feature at a time with its
    # training-median baseline and see how the probability moves. One predict per
    # feature — trivially cheap for a single listing.
    contributions = []
    actual_row = X.iloc[0]
    for feat in feature_order:
        X_occ = X.copy()
        X_occ.iloc[0, X_occ.columns.get_loc(feat)] = baselines[feat]
        prob_baseline = float(model.predict_proba(X_occ)[0, 1])
        contributions.append(
            {
                "feature": feat,
                "value": float(actual_row[feat]),
                "contribution": base_prob - prob_baseline,
            }
        )

    top_factors = sorted(
        contributions, key=lambda d: abs(d["contribution"]), reverse=True
    )[:5]

    return {
        "prob": base_prob,
        "tier": _tier_for(base_prob),
        "top_factors": top_factors,
        "note": bundle["note"],
    }


def predict_batch(df: pd.DataFrame, bundle: dict | None = None) -> np.ndarray:
    """
    Vectorised whole-frame scoring. Returns a 1-D np.ndarray of P(ghost) per row.
    No per-row occlusion (use predict_one for explanations).
    """
    if bundle is None:
        bundle = load_bundle()
    X = _build_X(df, bundle)
    return bundle["model"].predict_proba(X)[:, 1]


if __name__ == "__main__":
    # Hardcoded sample listing — fields keyed by features.py column names.
    sample = {
        "price": "$220.00",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 6,
        "bedrooms": 3,
        "beds": 4,
        "bathrooms": 2,
        "minimum_nights": 2,
        "maximum_nights": 365,
        "availability_30": 28,
        "availability_60": 58,
        "availability_90": 88,
        "availability_365": 360,
        "number_of_reviews": 1,
        "number_of_reviews_ltm": 0,
        "reviews_per_month": 0.02,
        "host_listings_count": 14,
        "host_total_listings_count": 22,
        "host_is_superhost": "f",
        "instant_bookable": "t",
    }

    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        print(e)
        print("\nNo trained model yet — train it first, then re-run this script.")
    else:
        result = predict_one(sample, bundle=bundle)
        print(json.dumps(result, indent=2, default=float))
