"""
Ghost Hotel Detector — Core Scoring Pipeline
/home/nvidia/AIR_BNB/pipeline/score.py

cuDF/cupy replaced with pandas/numpy — swap back when RAPIDS is available.
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import re
from pathlib import Path

DATA = Path("/home/nvidia/AIR_BNB/data/clean")
OUT  = Path("/home/nvidia/AIR_BNB/data/scored")
OUT.mkdir(exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

print("Loading listings...")
listings = pd.read_parquet(DATA / "listings.parquet")

print("Loading calendar...")
calendar = pd.read_parquet(DATA / "calendar.parquet")

print("Loading reviews...")
reviews = pd.read_parquet(DATA / "reviews.parquet")

print(f"Loaded: {len(listings):,} listings | {len(calendar):,} calendar rows | {len(reviews):,} reviews")

# ── 2. FEATURE 1 — 90-NIGHT BREACH SCORE (weight: 0.35) ──────────────────────

listings['nights_booked'] = listings['estimated_occupancy_l365d'].fillna(0)
listings['nights_ratio']  = (listings['nights_booked'] / 90).clip(upper=3.0)

def sigmoid_score(x, midpoint=1.5, steepness=3):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

listings['f1_nights_score'] = sigmoid_score(listings['nights_ratio'].values)

# ── 3. FEATURE 2 — OPERATOR PORTFOLIO SIZE (weight: 0.25) ────────────────────

portfolio = listings.groupby('host_id').agg(
    portfolio_size=('id', 'count'),
    portfolio_total_revenue=('estimated_revenue_l365d', 'sum')
).reset_index()

listings = listings.merge(portfolio, on='host_id', how='left')
listings['f2_portfolio_score'] = ((listings['portfolio_size'] - 1) / 20).clip(0, 1)

# ── 4. FEATURE 3 — REVIEW VELOCITY (weight: 0.20) ────────────────────────────

reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
reviews_2024 = reviews[reviews['date'] >= '2024-01-01']

review_counts = reviews_2024.groupby('listing_id').agg(
    reviews_2024=('id', 'count')
).reset_index().rename(columns={'listing_id': 'id'})

listings = listings.merge(review_counts, on='id', how='left')
listings['reviews_2024'] = listings['reviews_2024'].fillna(0)

borough_medians = (
    listings.groupby('neighbourhood_cleansed')['reviews_2024']
    .median()
    .rename('borough_median_reviews')
    .reset_index()
)
listings = listings.merge(borough_medians, on='neighbourhood_cleansed', how='left')

listings['f3_review_velocity'] = (
    listings['reviews_2024'] / (listings['borough_median_reviews'].fillna(1) + 1)
).clip(0, 3) / 3

# ── 5. FEATURE 4 — CALENDAR BOOKING DENSITY (weight: 0.20) ───────────────────

calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')
recent_cal = calendar[calendar['date'] >= '2024-09-01'].copy()
recent_cal['is_booked'] = (recent_cal['available'] == 'f').astype(int)

booking_density = (
    recent_cal.groupby('listing_id')['is_booked']
    .mean()
    .rename('booking_density_90d')
    .reset_index()
    .rename(columns={'listing_id': 'id'})
)

listings = listings.merge(booking_density, on='id', how='left')
listings['f4_booking_density'] = listings['booking_density_90d'].fillna(0)

# ── 6. FEATURES 5 & 6 — STUBS (HMO licence, borough complaints) ──────────────
listings['f5_hmo_score']        = 0.5
listings['f6_complaint_score']  = 0.5

# ── 7. COMPOSITE CONFIDENCE SCORE ─────────────────────────────────────────────

W1, W2, W3, W4, W5, W6 = 0.35, 0.25, 0.20, 0.20, 0.00, 0.00

listings['confidence_score'] = (
    W1 * listings['f1_nights_score']    +
    W2 * listings['f2_portfolio_score'] +
    W3 * listings['f3_review_velocity'] +
    W4 * listings['f4_booking_density']
).clip(0, 0.97)

# ── 8. OPERATOR ENTITY RESOLUTION ────────────────────────────────────────────

print("Running entity resolution...")

hosts_pdf = (
    listings[['host_id', 'host_name', 'portfolio_size']]
    .drop_duplicates('host_id')
    .copy()
)

def normalise_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(
        r'\b(ltd|limited|properties|property|management|mgmt|letting|lettings|group|uk)\b',
        '', name
    )
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return name.strip()

hosts_pdf['name_normalised'] = hosts_pdf['host_name'].apply(normalise_name)

commercial_hosts = hosts_pdf[hosts_pdf['portfolio_size'] >= 3].copy()
names = commercial_hosts['name_normalised'].tolist()
ids   = commercial_hosts['host_id'].tolist()

operator_map = {}
for i, (hid, name) in enumerate(zip(ids, names)):
    if hid in operator_map:
        continue
    operator_map[hid] = hid
    if not name:
        continue
    matches = process.extract(
        name, names[i + 1:], scorer=fuzz.token_sort_ratio, limit=5
    )
    for match_name, score, j in matches:
        if score >= 85:
            match_id = ids[i + 1 + j]
            if match_id not in operator_map:
                operator_map[match_id] = hid

# Single-listing hosts map to themselves
for hid in hosts_pdf['host_id']:
    if hid not in operator_map:
        operator_map[hid] = hid

hosts_pdf['canonical_operator_id'] = hosts_pdf['host_id'].map(
    lambda x: operator_map.get(x, x)
)

operator_portfolio = hosts_pdf.groupby('canonical_operator_id').agg(
    operator_total_listings=('host_id', 'count'),
    operator_name=('host_name', 'first')
).reset_index()

hosts_pdf = hosts_pdf.merge(operator_portfolio, on='canonical_operator_id', how='left')

listings = listings.merge(
    hosts_pdf[['host_id', 'canonical_operator_id', 'operator_total_listings', 'operator_name']],
    on='host_id', how='left'
)

# ── 9. PRIORITY TIERS ─────────────────────────────────────────────────────────

listings['priority_tier'] = pd.cut(
    listings['confidence_score'],
    bins=[0, 0.50, 0.75, 0.90, 1.0],
    labels=['Low', 'Monitor', 'Review', 'Priority']
)

# ── 10. SAVE OUTPUTS ──────────────────────────────────────────────────────────

listings.to_parquet(OUT / "scored_listings.parquet", index=False)

priority = listings[listings['confidence_score'] >= 0.75].sort_values(
    'confidence_score', ascending=False
)
priority.to_parquet(OUT / "priority_cases.parquet", index=False)
priority.to_csv(OUT / "priority_cases.csv", index=False)

operator_summary = listings.groupby('canonical_operator_id').agg(
    operator_name=('operator_name', 'first'),
    total_listings=('id', 'count'),
    listings_above_90_nights=('nights_booked', lambda x: (x > 90).sum()),
    avg_confidence=('confidence_score', 'mean'),
    max_confidence=('confidence_score', 'max'),
    total_estimated_revenue=('estimated_revenue_l365d', 'sum'),
    boroughs=('neighbourhood_cleansed', lambda x: list(x.unique()))
).reset_index().sort_values('max_confidence', ascending=False)

operator_summary.to_parquet(OUT / "operator_summary.parquet", index=False)
operator_summary.to_csv(OUT / "operator_summary.csv", index=False)

# ── 11. PRINT SUMMARY ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SCORING COMPLETE")
print("=" * 60)
print(f"Total listings scored:     {len(listings):,}")
print(f"Priority cases (>0.90):    {(listings['confidence_score'] >= 0.90).sum():,}")
print(f"Review cases (0.75-0.90):  {((listings['confidence_score'] >= 0.75) & (listings['confidence_score'] < 0.90)).sum():,}")
print(f"Unique operators found:    {listings['canonical_operator_id'].nunique():,}")

print(f"\nTop 10 operators by portfolio size:")
print(
    operator_summary[['operator_name', 'total_listings', 'listings_above_90_nights', 'avg_confidence']]
    .head(10)
    .to_string(index=False)
)

print(f"\nTop 10 priority cases:")
cols = ['id', 'host_name', 'neighbourhood_cleansed', 'nights_booked', 'confidence_score', 'operator_total_listings']
print(priority[cols].head(10).to_string(index=False))
