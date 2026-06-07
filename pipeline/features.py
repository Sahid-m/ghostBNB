"""
Shared feature spec for the ghost-hotel ML classifier.

CRITICAL DESIGN CONSTRAINT — "live-scrapeable features only".
The model is meant to score listings we have NO history for (created after our
25 Sep 2025 snapshot). So it may only use signals that can be read straight off
a live Airbnb listing page. We deliberately EXCLUDE every engineered F1–F8
signal, calendar-derived nights, operator/portfolio joins, and Inside-Airbnb's
`calculated_*` within-snapshot counts — none of those exist for a brand-new
listing, and including them would leak the heuristic into the model.

train_model.py, predict.py and airbnb_scraper.py all import from here so the
column set can never drift between training and inference.
"""
from __future__ import annotations

import re
import pandas as pd

# Read straight off the listing page. Numeric → HistGradientBoosting handles NaN.
NUMERIC = [
    "price",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "host_listings_count",
    "host_total_listings_count",
]

# Free-text categories shown on the page.
CATEGORICAL = ["room_type", "property_type"]

# Rendered as t/f in Inside Airbnb; true/false flags on the live page.
BOOL = ["host_is_superhost", "instant_bookable"]

# Final model input order. cat_<name> holds the integer-encoded categorical.
FEATURE_ORDER = NUMERIC + BOOL + [f"cat_{c}" for c in CATEGORICAL]


def parse_price(x) -> float:
    """'$1,234.00' / 1234 / None -> float or NaN."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    m = re.findall(r"[\d.]+", str(x).replace(",", ""))
    return float(m[0]) if m else float("nan")


def to_bool(x) -> float:
    """t/f, true/false, yes/no, 1/0, bool -> 1.0 / 0.0 / NaN."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    s = str(x).strip().lower()
    if s in ("t", "true", "yes", "y", "1"):
        return 1.0
    if s in ("f", "false", "no", "n", "0"):
        return 0.0
    return float("nan")


def build_matrix(df: pd.DataFrame, cat_maps: dict | None = None):
    """
    Turn a raw listing frame into the numeric model matrix.

    Returns (X, cat_maps). Pass cat_maps from the trained bundle at inference so
    categories encode identically; unseen categories map to -1.
    """
    out = pd.DataFrame(index=df.index)

    for c in NUMERIC:
        col = df[c] if c in df.columns else pd.Series(index=df.index, dtype="float64")
        out[c] = col.map(parse_price) if c == "price" else pd.to_numeric(col, errors="coerce")

    for c in BOOL:
        col = df[c] if c in df.columns else pd.Series(index=df.index, dtype="object")
        out[c] = col.map(to_bool)

    new_maps = {} if cat_maps is None else cat_maps
    for c in CATEGORICAL:
        col = df[c].astype("string") if c in df.columns else pd.Series(index=df.index, dtype="string")
        if cat_maps is None:
            cats = sorted(col.dropna().unique().tolist())
            new_maps[c] = {v: i for i, v in enumerate(cats)}
        out[f"cat_{c}"] = col.map(lambda v: new_maps[c].get(v, -1)).fillna(-1).astype("int64")

    return out[FEATURE_ORDER], new_maps
