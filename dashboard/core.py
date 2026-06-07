"""
Shared backbone for the Ghost Hotel Detector dashboard + agent.

Everything that both the Dashboard view and the Agent console need lives here:
config, cached data loaders, the vLLM (Nemotron) chat helpers, the live scraper
wrapper and the ML predictor handle. Keeping it in one module avoids circular
imports between app.py and agent.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

# Make the sibling `pipeline/` package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.airbnb_scraper import fetch_listing            # noqa: E402
from pipeline import predict as ghost_predict                # noqa: E402
from pipeline.features import NUMERIC, CATEGORICAL, BOOL     # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
ROOT        = Path("/home/nvidia/AIR_BNB")
SCORED      = ROOT / "data" / "scored"
ENFORCE_DIR = ROOT / "data" / "enforcement"
BENCH       = ROOT / "benchmarks" / "calendar_benchmark.json"

NIM_URL = "http://10.18.216.24:8090/v1/chat/completions"
MODEL   = "nemotron-nano-vl"

SNAPSHOT_DATE = "25 Sep 2025"

TIER_COLOURS = {
    "Priority": "#e63946",
    "Review":   "#f4a261",
    "Monitor":  "#457b9d",
    "Low":      "#a8dadc",
}

# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import pandas as pd
    listings  = pd.read_parquet(SCORED / "scored_listings.parquet")
    operators = pd.read_parquet(SCORED / "operator_summary_enriched.parquet")
    priority  = pd.read_parquet(SCORED / "priority_cases.parquet")
    return listings, operators, priority


@st.cache_data
def load_benchmark():
    if BENCH.exists():
        return json.loads(BENCH.read_text())
    return {"cpu_seconds": 7.20, "gpu_seconds": 1.88, "speedup": 3.8}


@st.cache_resource
def load_ghost_model():
    """ML classifier bundle, or None if not trained yet."""
    try:
        return ghost_predict.load_bundle()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_scrape(listing_id: int):
    """Best-effort live scrape, cached per listing so reruns don't re-fetch."""
    return fetch_listing(str(listing_id))


# ── vLLM (Nemotron) helpers ─────────────────────────────────────────────────
def nim_ready() -> bool:
    try:
        r = requests.get("http://10.18.216.24:8090/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_nim(prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """Blocking chat call with retry on 503 (model still loading)."""
    import time
    for attempt in range(4):
        try:
            resp = requests.post(
                NIM_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=180,
            )
            if resp.status_code == 503:
                if attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError("Model is still loading — try again in ~30 seconds")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError("NIM call failed after retries")


def stream_nim(prompt: str, max_tokens: int = 4096, temperature: float = 0.0):
    """Generator yielding (kind, chunk) where kind is 'reasoning' or 'content'."""
    resp = requests.post(
        NIM_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw or raw == b"data: [DONE]":
            continue
        raw = raw.decode()
        if not raw.startswith("data: "):
            continue
        try:
            delta = json.loads(raw[6:])["choices"][0]["delta"]
            if delta.get("reasoning_content"):
                yield "reasoning", delta["reasoning_content"]
            if delta.get("content"):
                yield "content", delta["content"]
        except Exception:
            continue
