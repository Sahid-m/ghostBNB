"""
Feature 8 — Review Pattern Anomaly Detector
/home/nvidia/AIR_BNB/pipeline/review_anomaly.py

No review text available — anomaly detection from patterns:
  - Burst score:     max reviews in any 7-day window (ghost hotels get blitzes)
  - Regularity:      low std-dev of inter-review gaps (too regular = suspicious)
  - Reviewer reuse:  same reviewer_id appearing across multiple high-score listings
  - Velocity ratio:  2024 review rate vs lifetime rate

Nemotron interprets the pattern numbers for the top 200 cases.
"""

import pandas as pd
import numpy as np
import requests
import json
import re
from pathlib import Path

SCORED  = Path("/home/nvidia/AIR_BNB/data/scored")
CLEAN   = Path("/home/nvidia/AIR_BNB/data/clean")
NIM_URL = "http://10.18.216.24:8090/v1/chat/completions"
MODEL   = "nemotron-nano-vl"


def call_nim_anomaly(cases: list[dict]) -> list[float]:
    """Ask Nemotron to assess review pattern anomaly for a batch of cases."""
    lines = []
    for i, c in enumerate(cases, 1):
        lines.append(
            f"Case {i} (host: {c['host_name']}, {c['borough']}): "
            f"burst_score={c['burst_score']:.2f}, "
            f"regularity_score={c['regularity_score']:.2f}, "
            f"reviewer_reuse_score={c['reviewer_reuse_score']:.2f}, "
            f"velocity_ratio={c['velocity_ratio']:.2f}, "
            f"total_reviews={c['total_reviews']}"
        )

    prompt = (
        "You are assessing Airbnb review patterns for ghost hotel detection.\n"
        "Score each case 0.0–1.0 for review anomaly (suspicious pattern = high score).\n\n"
        "Signal meanings:\n"
        "- burst_score: proportion of reviews arriving in short bursts (high = suspicious)\n"
        "- regularity_score: how unnaturally evenly-spaced reviews are (high = suspicious)\n"
        "- reviewer_reuse_score: how often the same reviewers appear across operator listings (high = suspicious)\n"
        "- velocity_ratio: recent review rate vs lifetime rate (>2 = recent acceleration)\n\n"
        + "\n".join(lines)
        + f"\n\nReturn ONLY a JSON array of {len(cases)} floats 0.0–1.0. No text."
    )

    for attempt in range(3):
        try:
            resp = requests.post(
                NIM_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,   # reasoning model needs tokens to think first
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\[[\d\s.,]+\]", content)
            if match:
                scores = json.loads(match.group())
                if len(scores) == len(cases):
                    return [float(np.clip(s, 0, 1)) for s in scores]
            break
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                import time; time.sleep(3 * (attempt + 1))
                continue
            print(f"    NIM anomaly batch failed: {e}")
            break
    return [0.5] * len(cases)


def run():
    print("Loading data...")
    listings = pd.read_parquet(SCORED / "scored_listings.parquet")
    reviews  = pd.read_parquet(CLEAN  / "reviews.parquet")

    # Drop any existing F8 columns to avoid merge conflicts on re-run
    drop_cols = ["burst_score", "regularity_score", "velocity_ratio",
                 "op_reviewer_reuse", "f8_review_anomaly"]
    listings.drop(columns=[c for c in drop_cols if c in listings.columns], inplace=True)

    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])

    # ── Burst score ───────────────────────────────────────────────────────────
    print("Computing burst scores...")

    def burst_score(dates):
        if len(dates) < 3:
            return 0.0
        dates = sorted(dates)
        max_in_7d = 0
        for i, d in enumerate(dates):
            window_end = d + pd.Timedelta(days=7)
            count = sum(1 for x in dates[i:] if x <= window_end)
            max_in_7d = max(max_in_7d, count)
        return float(np.clip(max_in_7d / max(len(dates), 1), 0, 1))

    burst = (
        reviews.groupby("listing_id")["date"]
        .apply(burst_score)
        .rename("burst_score")
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )

    # ── Regularity score ──────────────────────────────────────────────────────
    print("Computing regularity scores...")

    def regularity_score(dates):
        if len(dates) < 4:
            return 0.0
        dates = sorted(dates)
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        if not gaps or np.mean(gaps) == 0:
            return 0.0
        cv = np.std(gaps) / (np.mean(gaps) + 1e-9)  # coefficient of variation
        # Low CV = suspiciously regular; invert so high score = suspicious
        return float(np.clip(1 - cv / 3, 0, 1))

    regularity = (
        reviews.groupby("listing_id")["date"]
        .apply(regularity_score)
        .rename("regularity_score")
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )

    # ── Reviewer reuse across operator listings ───────────────────────────────
    print("Computing reviewer reuse scores...")

    # Get operator membership per listing
    listing_to_operator = listings.set_index("id")["canonical_operator_id"].to_dict()
    reviews["operator_id"] = reviews["listing_id"].map(listing_to_operator)

    # For each operator, find what fraction of reviews come from repeated reviewers
    operator_reviews = reviews.dropna(subset=["operator_id"])
    reviewer_op_counts = (
        operator_reviews.groupby(["operator_id", "reviewer_id"])
        .size()
        .reset_index(name="appearances")
    )
    reviewer_reuse = (
        reviewer_op_counts.groupby("operator_id")
        .apply(lambda x: (x["appearances"] > 1).sum() / max(len(x), 1))
        .rename("op_reviewer_reuse")
        .reset_index()
    )

    # Map operator reuse score back to listings via canonical_operator_id
    op_reuse_map = reviewer_reuse.set_index("operator_id")["op_reviewer_reuse"].to_dict()
    listings["op_reviewer_reuse"] = listings["canonical_operator_id"].map(op_reuse_map).fillna(0)

    # ── Velocity ratio ────────────────────────────────────────────────────────
    reviews_2024 = reviews[reviews["date"] >= "2024-01-01"]
    recent_counts = (
        reviews_2024.groupby("listing_id").size()
        .rename("reviews_recent")
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )
    all_counts = (
        reviews.groupby("listing_id").size()
        .rename("reviews_all")
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )
    velocity = recent_counts.merge(all_counts, on="id", how="outer").fillna(0)
    # Annualised: recent covers ~18 months of data
    velocity["velocity_ratio"] = (
        (velocity["reviews_recent"] / 1.5) /
        ((velocity["reviews_all"] / 10).clip(lower=1))  # 10-year max history
    ).clip(0, 5) / 5

    # ── Merge all signals into listings ──────────────────────────────────────
    listings = listings.merge(burst,      on="id", how="left")
    listings = listings.merge(regularity, on="id", how="left")
    listings = listings.merge(velocity[["id", "velocity_ratio"]], on="id", how="left")

    for col in ["burst_score", "regularity_score", "velocity_ratio"]:
        listings[col] = listings[col].fillna(0)
    listings["op_reviewer_reuse"] = listings["op_reviewer_reuse"].fillna(0)

    # ── Composite F8 (rule-based, before NIM) ─────────────────────────────────
    listings["f8_review_anomaly"] = (
        0.35 * listings["burst_score"] +
        0.25 * listings["regularity_score"] +
        0.25 * listings["op_reviewer_reuse"] +
        0.15 * listings["velocity_ratio"]
    ).clip(0, 1)

    print(f"  F8 mean: {listings['f8_review_anomaly'].mean():.3f}  "
          f"max: {listings['f8_review_anomaly'].max():.3f}")

    # ── NIM enhancement for top 200 cases ────────────────────────────────────
    top200 = listings.nlargest(200, "confidence_score").copy()
    print(f"\nNIM anomaly scoring top {len(top200)} cases (20 batches of 10)...")

    batch_size = 10
    nim_scores = {}
    rows = top200[[
        "id", "host_name", "neighbourhood_cleansed",
        "burst_score", "regularity_score", "op_reviewer_reuse",
        "velocity_ratio", "number_of_reviews"
    ]].rename(columns={
        "neighbourhood_cleansed": "borough",
        "op_reviewer_reuse": "reviewer_reuse_score",
        "number_of_reviews": "total_reviews"
    }).to_dict("records")

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        scores = call_nim_anomaly(batch)
        for item, score in zip(batch, scores):
            nim_scores[item["id"]] = score
        print(f"  Batch {start//batch_size + 1}/{len(rows)//batch_size + 1} done")

    listings["f8_nim_score"] = listings["id"].map(nim_scores)
    # Blend: NIM overrides for top 200, rule-based for rest
    listings["f8_review_anomaly"] = listings["f8_nim_score"].combine_first(
        listings["f8_review_anomaly"]
    )
    listings.drop(columns=["f8_nim_score"], inplace=True)

    # ── Re-score with F7 + F8 as additive boosters ───────────────────────────
    # Base score is unchanged (F1–F4). F7/F8 boost above 0.5 raise the score,
    # below 0.5 lower it. Neutral (0.5) = no effect. NIM-down fallback safe.
    if "f7_commercial_language" in listings.columns:
        print("\nRecomputing confidence score with F7 + F8 boosters...")
        base = (
            0.35 * listings["f1_nights_score"] +
            0.25 * listings["f2_portfolio_score"] +
            0.20 * listings["f3_review_velocity"] +
            0.20 * listings["f4_booking_density"]
        )
        ai_boost = (
            0.12 * (listings["f7_commercial_language"] - 0.5) +
            0.08 * (listings["f8_review_anomaly"]      - 0.5)
        )
        listings["confidence_score"] = (base + ai_boost).clip(0, 0.97)

        listings["priority_tier"] = pd.cut(
            listings["confidence_score"],
            bins=[0, 0.50, 0.75, 0.90, 1.0],
            labels=["Low", "Monitor", "Review", "Priority"]
        )
        print(f"  Updated tier counts:\n{listings['priority_tier'].value_counts().sort_index().to_string()}")
    else:
        print("\nNote: run description_classifier.py first to include F7 in rescore")

    listings.to_parquet(SCORED / "scored_listings.parquet", index=False)

    # Rebuild priority_cases
    priority = listings[listings["confidence_score"] >= 0.75].sort_values(
        "confidence_score", ascending=False
    )
    priority.to_parquet(SCORED / "priority_cases.parquet", index=False)
    priority.to_csv(SCORED / "priority_cases.csv", index=False)

    print(f"\nTop 10 by review anomaly score:")
    top_anomaly = listings.nlargest(10, "f8_review_anomaly")[
        ["host_name", "neighbourhood_cleansed", "f8_review_anomaly",
         "burst_score", "regularity_score", "confidence_score"]
    ]
    print(top_anomaly.to_string(index=False))

    return listings


if __name__ == "__main__":
    run()
