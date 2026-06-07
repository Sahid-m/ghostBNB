"""
Ghost Hotel Detector — Enforcement Notice Generator
/home/nvidia/AIR_BNB/pipeline/generate_enforcement.py

Calls Nemotron Nano on the local NIM endpoint to draft enforcement packages.
Run AFTER score.py
"""

import requests
import json
import pandas as pd
from pathlib import Path

NIM_URL = "http://10.18.216.24:8090/v1/chat/completions"
MODEL   = "nemotron-nano-vl"
OUT     = Path("/home/nvidia/AIR_BNB/data/enforcement")
OUT.mkdir(exist_ok=True)


def call_nim(prompt: str) -> str:
    resp = requests.post(
        NIM_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,   # reasoning model uses tokens before writing output
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def build_prompt(case: dict) -> str:
    nights = case.get("nights_booked", 0)
    overage = max(0, nights - 90)
    return f"""You are a housing enforcement officer for {case.get("borough", "London")} Council.

EVIDENCE:
- Airbnb listing ID: {case.get("listing_id")}
- Host name: {case.get("host_name")}
- Location: {case.get("borough")}, London
- Estimated nights booked (last 365 days): {nights:.0f}
- Legal annual limit: 90 nights (Deregulation Act 2015, Section 44)
- Nights over limit: {overage:.0f}
- Operator portfolio size: {case.get("operator_total_listings", 1)} properties
- Algorithmic confidence score: {case.get("confidence_score", 0):.2f} / 1.00

Generate three things:

1. ENFORCEMENT NOTICE (formal language, cite Deregulation Act 2015 s.44 and Housing Act 2004 where relevant)
2. EVIDENCE SUMMARY (3 sentences for the case file)
3. RECOMMENDED ACTION (choose one: Formal Warning / Fixed Penalty Notice £1,000 / Prosecution)

Format with these exact headers:
## ENFORCEMENT NOTICE
## EVIDENCE SUMMARY
## RECOMMENDED ACTION
"""


def generate_batch(n: int = 10):
    priority = pd.read_parquet(
        "/home/nvidia/AIR_BNB/data/scored/priority_cases.parquet"
    )
    top = priority.head(n).copy()

    results = []
    for idx, (_, row) in enumerate(top.iterrows()):
        case = {
            "listing_id":             row.get("id"),
            "host_name":              row.get("host_name"),
            "borough":                row.get("neighbourhood_cleansed"),
            "nights_booked":          row.get("nights_booked", 0),
            "operator_total_listings": row.get("operator_total_listings", 1),
            "confidence_score":       row.get("confidence_score", 0),
        }

        print(f"\n[{idx+1}/{n}] {case['host_name']} — {case['borough']} "
              f"— score {case['confidence_score']:.2f} "
              f"— {case['nights_booked']:.0f} nights")

        try:
            output = call_nim(build_prompt(case))
        except Exception as e:
            output = f"ERROR: {e}"
            print(f"  NIM call failed: {e}")

        result = {**case, "enforcement_output": output}
        results.append(result)

        fname = OUT / f"case_{case['listing_id']}.json"
        json.dump(result, open(fname, "w"), indent=2, default=str)
        print(f"  Saved → {fname.name}")

    pd.DataFrame(results).to_csv(OUT / "enforcement_batch.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} enforcement packages generated → {OUT}")

    # Print first case in full
    if results:
        print(f"\n{'='*60}")
        print("FIRST CASE — FULL OUTPUT")
        print(f"{'='*60}")
        c = results[0]
        print(f"Listing: {c['listing_id']}  Host: {c['host_name']}  Borough: {c['borough']}")
        print(f"Nights booked: {c['nights_booked']:.0f}  Score: {c['confidence_score']:.2f}\n")
        print(c["enforcement_output"])

    return results


if __name__ == "__main__":
    generate_batch(n=10)
