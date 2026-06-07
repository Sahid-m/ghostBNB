"""
Feature 7 — Commercial Language Classifier
/home/nvidia/AIR_BNB/pipeline/description_classifier.py

Two-pass approach:
  Pass 1 — keyword scoring for all 51k listings (instant, deterministic)
  Pass 2 — Nemotron batch scoring for top 300 priority cases (10 per call)

F7 score: 0.0 = clearly personal home-share  /  1.0 = clearly commercial operator
"""

import pandas as pd
import numpy as np
import requests
import json
import re
import time
from pathlib import Path

SCORED  = Path("/home/nvidia/AIR_BNB/data/scored")
NIM_URL = "http://10.18.216.24:8090/v1/chat/completions"
MODEL   = "nemotron-nano-vl"
BATCH   = 10   # listings per NIM call

# ── Keyword dictionaries ──────────────────────────────────────────────────────

COMMERCIAL = [
    r"\bour team\b", r"\bour properties\b", r"\bour portfolio\b",
    r"\bprofessional clean", r"\bproperty management\b", r"\bmanaged by\b",
    r"\bguest services\b", r"\bdedicated host team\b", r"\b24.?7 support\b",
    r"\bconcierge\b", r"\bcheck.?in team\b", r"\bkey.?box\b", r"\block.?box\b",
    r"\bsmart lock\b", r"\bself.?check.?in\b.*\bteam\b",
    r"\bportfolio of\b", r"\bmanagement company\b", r"\bletting agent\b",
    r"\bseveral properties\b", r"\bmultiple properties\b", r"\bvarious properties\b",
    r"\bshort.?term rental\b", r"\bholiday let\b", r"\bserviced apartment\b",
    r"\bhotel.?style\b", r"\boutstanding reviews across\b",
]

PERSONAL = [
    r"\bmy home\b", r"\bmy flat\b", r"\bmy apartment\b", r"\bmy house\b",
    r"\bour home\b", r"\bour flat\b", r"\bour apartment\b",
    r"\bi live\b", r"\bwe live\b", r"\bi work\b", r"\bi('m| am) a\b",
    r"\bmy bedroom\b", r"\bi('ll| will) be\b", r"\bmy dog\b", r"\bmy cat\b",
    r"\bmy family\b", r"\bi love\b", r"\bmy neighbourhood\b", r"\bmy street\b",
    r"\bborn and raised\b", r"\blocal\b.*\bborn\b",
]


def keyword_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.5  # neutral if no text
    t = text.lower()
    commercial_hits = sum(1 for p in COMMERCIAL if re.search(p, t))
    personal_hits   = sum(1 for p in PERSONAL   if re.search(p, t))
    # Normalise: more commercial hits → higher score
    raw = (commercial_hits * 2 - personal_hits) / 8
    return float(np.clip(0.5 + raw, 0.0, 1.0))


def call_nim_batch(batch: list[dict]) -> list[float]:
    """Score a batch of listings via Nemotron. Returns list of floats 0-1."""
    lines = []
    for i, item in enumerate(batch, 1):
        desc      = str(item.get("description", "") or "")[:400]
        host_bio  = str(item.get("host_about",   "") or "")[:200]
        host_name = str(item.get("host_name",    "") or "")
        lines.append(f"Listing {i} (host: {host_name}): {desc} | Bio: {host_bio}")

    prompt = (
        "Score each Airbnb listing 0.0–1.0 for commercial operation indicators.\n"
        "1.0 = clearly commercial ('our team', 'portfolio', 'professional management', 'serviced apartment')\n"
        "0.0 = clearly personal home-share ('my home', 'I live here', 'my flat')\n"
        "0.5 = ambiguous\n\n"
        + "\n".join(lines)
        + f"\n\nReturn ONLY a JSON array of {len(batch)} numbers, e.g. [0.2, 0.8, 0.5, ...]. "
          "No explanation, no text — just the array."
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
                if len(scores) == len(batch):
                    return [float(np.clip(s, 0, 1)) for s in scores]
            break  # got a response, just couldn't parse it
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"    NIM batch failed: {e}")
            break
    return [0.5] * len(batch)


def run():
    print("Loading scored listings...")
    listings = pd.read_parquet(SCORED / "scored_listings.parquet")

    # ── Pass 1: keyword scoring for all listings ──────────────────────────────
    print(f"Pass 1 — keyword scoring {len(listings):,} listings...")

    combined_text = (
        listings["description"].fillna("") + " " +
        listings["host_about"].fillna("")
    )
    listings["f7_commercial_language"] = combined_text.apply(keyword_score)

    dist = pd.cut(listings["f7_commercial_language"],
                  bins=[0, 0.3, 0.5, 0.7, 1.01],
                  labels=["Personal", "Lean personal", "Lean commercial", "Commercial"])
    print(f"  Distribution:\n{dist.value_counts().sort_index().to_string()}")

    # ── Pass 2: NIM batch scoring for top 300 priority cases ─────────────────
    top300 = listings.nlargest(300, "confidence_score").copy()
    print(f"\nPass 2 — NIM batch scoring top {len(top300)} priority listings "
          f"({len(top300) // BATCH} batches of {BATCH})...")

    nim_scores = {}
    rows = top300[["id", "host_name", "description", "host_about"]].to_dict("records")

    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        scores = call_nim_batch(batch)
        for item, score in zip(batch, scores):
            nim_scores[item["id"]] = score
        print(f"  Batch {start//BATCH + 1}/{len(rows)//BATCH + 1} done "
              f"({start + len(batch)}/{len(rows)})")

    # Blend: NIM score overrides keyword score where available
    listings["f7_nim_score"] = listings["id"].map(nim_scores)
    listings["f7_commercial_language"] = listings["f7_nim_score"].combine_first(
        listings["f7_commercial_language"]
    )
    listings.drop(columns=["f7_nim_score"], inplace=True)

    print(f"\nNIM-scored: {len(nim_scores)} listings")
    print(f"Keyword-only: {len(listings) - len(nim_scores)} listings")

    # ── Save ──────────────────────────────────────────────────────────────────
    listings.to_parquet(SCORED / "scored_listings.parquet", index=False)

    top_commercial = listings.nlargest(10, "f7_commercial_language")[
        ["host_name", "neighbourhood_cleansed", "f7_commercial_language",
         "confidence_score", "nights_booked"]
    ]
    print(f"\nTop 10 most commercial-language listings:")
    print(top_commercial.to_string(index=False))

    return listings


if __name__ == "__main__":
    run()
