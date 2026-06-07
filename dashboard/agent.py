"""
Agentic console "brain" for the Ghost Hotel Detector.

This module is the pure-logic layer behind an agentic Streamlit console. It
ports the working logic out of `dashboard/app.py` (NL->pandas search,
enforcement drafting, property lookup, operator intel) into pure functions
that RETURN structured data instead of calling st.* — so the UI can render or
stream them however it likes.

Run via Streamlit as a flat sibling of core.py / app.py.

Design rules:
- Never call core.load_data() / other @st.cache loaders at import time (they
  need a Streamlit runtime). They are called lazily inside the tool functions.
- Tools never raise — on failure they return {"kind": "error", "message": ...}.
- route() never raises — it always returns a valid {tool, args, rationale}.
"""
from __future__ import annotations

import json
import re

import pandas as pd

import core

# ── A) Module constants ──────────────────────────────────────────────────────

SCHEMA_DESC = """
DATASET: 51,996 scored London Airbnb listings.

COLUMNS (exact names and types):
- host_name (str)
- neighbourhood_cleansed (str): London BOROUGH name — exact values include:
    'Westminster', 'Camden', 'Tower Hamlets', 'Hackney', 'Islington',
    'Kensington and Chelsea', 'Southwark', 'Lambeth', 'Hammersmith and Fulham',
    'Wandsworth', 'Newham', 'Lewisham', 'Greenwich', 'Barnet', 'Brent',
    'Ealing', 'Haringey', 'Croydon', 'Bromley', 'Hillingdon'
- confidence_score (float 0–0.97): ghost hotel score
- priority_tier (Categorical): EXACT values are 'Priority', 'Review', 'Monitor', 'Low'
    — ALWAYS use .astype(str) == 'Priority'  (never 'high', 'critical', or other values)
- nights_booked (float): estimated nights booked last 365 days (legal cap = 90)
- portfolio_size (int): listings under same host_id
- operator_total_listings (int): listings under resolved operator entity
- operator_name (str): resolved operator name
- f1_nights_score (float 0–1): 90-night breach signal
- f2_portfolio_score (float 0–1): portfolio size signal
- f3_review_velocity (float 0–1): review acceleration
- f4_booking_density (float 0–1): calendar booking density
- f7_commercial_language (float 0–1): 0=personal home, 1=commercial operator
- f8_review_anomaly (float 0–1): review pattern anomaly
- estimated_revenue_l365d (float USD): annual revenue estimate
- property_type (Categorical): use .astype(str).str.contains(...)
- room_type (Categorical): use .astype(str) — values: 'Entire home/apt', 'Private room', 'Shared room'
- number_of_reviews (int)
- reviews_per_month (float)
- latitude, longitude (float)
"""

QUICK_ACTIONS = [
    "Worst offenders in Westminster",
    "Investigate airbnb.com/rooms/13913",
    "Operators earning over £50,000 a year",
    "Priority cases in Tower Hamlets",
    "Show the top 20 most suspicious operators",
    "Draft an enforcement notice for the worst Camden case",
]

TOOLS = [
    {
        "name": "search_listings",
        "label": "AI Search",
        "description": (
            "Natural-language search over the 51,996 scored listings. Translates "
            "the question into a pandas filter and returns matching listings with "
            "summary stats."
        ),
    },
    {
        "name": "lookup_property",
        "label": "Property Lookup",
        "description": (
            "Full intelligence on a single property by Airbnb URL or listing id. "
            "Uses the snapshot if present, otherwise live-scrapes and ML-scores it."
        ),
    },
    {
        "name": "operator_intel",
        "label": "Operator Intel",
        "description": (
            "Commercial-operator portfolios with Companies House matches, revenue "
            "estimates and confidence. Search by name or list the top operators."
        ),
    },
    {
        "name": "top_cases",
        "label": "Top Cases",
        "description": (
            "The highest-confidence listings, optionally filtered by borough and/or "
            "priority tier — i.e. the worst offenders / priority queue."
        ),
    },
    {
        "name": "draft_enforcement",
        "label": "Draft Enforcement",
        "description": (
            "Drafts a formal enforcement package (Deregulation Act 2015 s.44) for a "
            "specific listing, the worst case in a borough, or the worst case overall."
        ),
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


# ── B) Prompt builders (verbatim from app.py) ────────────────────────────────

def build_filter_prompt(question: str) -> str:
    return f"""You are an AI analyst embedded in the Ghost Hotel Detector — a London housing enforcement intelligence system built for council officers.

The system has scored 51,996 London Airbnb listings for illegal short-term rental activity. Properties operating over 90 nights/year without planning permission breach the Deregulation Act 2015 s.44. The dataset is the result of cross-referencing Inside Airbnb, Land Registry, Companies House, and OS UPRN data.

{SCHEMA_DESC}

GEOGRAPHY — neighbourhood_cleansed is the borough, not the street/area. Mappings:
- Shoreditch, Dalston, Stoke Newington → 'Hackney'
- Brixton, Streatham, Clapham → 'Lambeth'
- Notting Hill, Portobello, Holland Park → 'Kensington and Chelsea'
- Soho, Mayfair, Covent Garden, Victoria → 'Westminster'
- Canary Wharf, Whitechapel, Bethnal Green → 'Tower Hamlets'
- Peckham, Bermondsey, Borough → 'Southwark'
- Battersea, Clapham (south) → 'Wandsworth'
- If unsure of the borough, use .astype(str).str.contains('area', case=False)

User question: "{question}"

Think step by step, then write a single pandas expression that answers the question.

STRICT OUTPUT RULES:
- Output ONLY the pandas expression on the very last line — no text after it
- Always return a full DataFrame (df[...] or df.sort_values(...).head(N))
- Never return a Series or a column selection
- Use df['col'].astype(str) for Categorical columns (priority_tier, room_type, property_type)
- String equality: df['col'].astype(str) == 'value'
- String partial match: df['col'].astype(str).str.contains('x', case=False)
- Revenue column is USD — £50k ≈ 63000
- Combine conditions with & and | inside parentheses
- No backticks, no assignment, no markdown

Expression:"""


def build_insight_prompt(question: str, result_stats: str) -> str:
    return f"""You are a housing enforcement intelligence analyst.

A data analyst asked: "{question}"

The query returned these results:
{result_stats}

Write a sharp 3–5 sentence intelligence briefing covering:
1. What the results reveal about illegal short-term rental activity
2. The highest-risk operator or listing and why
3. Recommended enforcement focus

Be specific — use the actual numbers provided. No fluff."""


# ── Internal helpers ─────────────────────────────────────────────────────────

def _clean_expr(content_text: str) -> str:
    """Clean a model-generated pandas expression exactly like app.py's tab_search."""
    expr = content_text.strip()
    expr = re.sub(r"```(?:python)?\s*", "", expr).strip("`").strip()
    if "Expression:" in expr:
        expr = expr.split("Expression:")[-1].strip().strip("`")
    lines = [l.strip().strip("`") for l in expr.splitlines() if l.strip()]
    df_lines = [l for l in lines if l.startswith("df")]
    expr = df_lines[-1] if df_lines else (lines[-1] if lines else expr)
    return expr


def _table_stats(result: pd.DataFrame) -> dict:
    """Summary stats block shared by search_listings and top_cases."""
    try:
        priority = int((result["priority_tier"].astype(str) == "Priority").sum())
    except Exception:
        priority = 0
    try:
        avg_score = float(result["confidence_score"].mean())
    except Exception:
        avg_score = 0.0
    try:
        avg_nights = float(result["nights_booked"].mean())
    except Exception:
        avg_nights = 0.0
    try:
        top_boroughs = (
            result.groupby("neighbourhood_cleansed").size().nlargest(5).to_dict()
        )
    except Exception:
        top_boroughs = {}
    return {
        "matched": int(len(result)),
        "priority": priority,
        "avg_score": avg_score,
        "avg_nights": avg_nights,
        "top_boroughs": top_boroughs,
    }


def _parse_listing_id(text: str):
    """Pull a numeric listing id from an Airbnb URL or bare string."""
    if text is None:
        return None
    text = str(text).strip()
    m = re.search(r"/rooms/(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\d{6,}", text)
    if m:
        return int(m.group(0))
    if text.isdigit():
        return int(text)
    return None


# Boroughs referenced for the fallback router / borough resolution.
_KNOWN_BOROUGHS = [
    "Westminster", "Camden", "Tower Hamlets", "Hackney", "Islington",
    "Kensington and Chelsea", "Southwark", "Lambeth", "Hammersmith and Fulham",
    "Wandsworth", "Newham", "Lewisham", "Greenwich", "Barnet", "Brent",
    "Ealing", "Haringey", "Croydon", "Bromley", "Hillingdon",
]
_KNOWN_TIERS = ["Priority", "Review", "Monitor", "Low"]


def _detect_borough(message: str):
    low = message.lower()
    for b in _KNOWN_BOROUGHS:
        if b.lower() in low:
            return b
    return None


def _detect_tier(message: str):
    low = message.lower()
    for t in _KNOWN_TIERS:
        if t.lower() in low:
            return t
    return None


# ── C) Router ────────────────────────────────────────────────────────────────

def _fallback_route(message: str) -> dict:
    """Deterministic regex router used when the LLM router fails."""
    low = message.lower()

    # 1. Property lookup — URL or a standalone run of >=6 digits.
    if "/rooms/" in low or re.search(r"\d{6,}", message):
        token = None
        m = re.search(r"/rooms/\d+", message)
        if m:
            token = m.group(0)
        else:
            m = re.search(r"\d{6,}", message)
            token = m.group(0) if m else message
        return {
            "tool": "lookup_property",
            "args": {"id_or_url": token},
            "rationale": "Message references a specific Airbnb listing id/URL.",
        }

    # 2. Enforcement drafting.
    if re.search(r"enforcement|notice|draft|prosecut", low):
        args = {}
        lid = _parse_listing_id(message)
        if lid is not None:
            args["listing_id"] = lid
        borough = _detect_borough(message)
        if borough:
            args["borough"] = borough
        return {
            "tool": "draft_enforcement",
            "args": args,
            "rationale": "Message asks to draft / prosecute an enforcement case.",
        }

    # 3. Operator intel.
    if re.search(r"operator|company|companies house|portfolio|landlord", low):
        name = re.sub(
            r"\b(operator|operators|company|companies house|companies|portfolio|"
            r"landlord|landlords|show|me|the|top|find|search|for|about|named|"
            r"called|with|of|a|an)\b",
            " ", low,
        )
        name = re.sub(r"[^a-z0-9 &']", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return {
            "tool": "operator_intel",
            "args": {"name": name},
            "rationale": "Message is about a commercial operator / portfolio.",
        }

    # 4. Top / worst / priority cases — only if a borough or tier is detectable.
    if re.search(r"top|worst|most suspicious|priority cases|review cases", low):
        borough = _detect_borough(message)
        tier = _detect_tier(message)
        if borough or tier:
            args = {}
            if borough:
                args["borough"] = borough
            if tier:
                args["tier"] = tier
            m = re.search(r"\b(\d{1,3})\b", message)
            if m:
                args["n"] = int(m.group(1))
            return {
                "tool": "top_cases",
                "args": args,
                "rationale": "Message asks for top/worst/priority cases in a borough or tier.",
            }

    # 5. Default — NL search.
    return {
        "tool": "search_listings",
        "args": {"query": message},
        "rationale": "No specific tool matched — falling back to natural-language search.",
    }


def _build_router_prompt(message: str) -> str:
    return f"""You are the router for an agentic housing-enforcement console (the Ghost Hotel Detector). Choose exactly ONE tool to handle the officer's message and extract its arguments.

TOOLS:
1. search_listings — natural-language search/filter over the scored listings dataset.
   args: {{"query": "<the user's question, verbatim or lightly cleaned>"}}
2. lookup_property — full intelligence on ONE property identified by an Airbnb URL or listing id.
   args: {{"id_or_url": "<the url or numeric id>"}}
3. operator_intel — commercial operator portfolios / Companies House matches.
   args: {{"name": "<operator name to search, or empty string for top operators>"}}
4. top_cases — highest-confidence (worst) listings, optionally filtered by borough and/or tier.
   args: {{"borough": "<London borough or null>", "tier": "<Priority|Review|Monitor|Low or null>", "n": <int, default 20>}}
5. draft_enforcement — draft a formal enforcement notice (Deregulation Act 2015 s.44).
   args: {{"listing_id": <numeric id or null>, "borough": "<borough or null>"}}

OFFICER MESSAGE: "{message}"

Reply with ONLY a single JSON object, no prose, no markdown fences:
{{"tool": "<one of the 5 tool names>", "args": {{...}}, "rationale": "<one short sentence>"}}"""


def _extract_json(text: str):
    """Best-effort: strip fences, grab the first {...} blob, json.loads it."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = m.group(0) if m else cleaned
    try:
        return json.loads(candidate)
    except Exception:
        return None


def route(message: str) -> dict:
    """Route a free-text message to a tool. Never raises."""
    message = "" if message is None else str(message)

    # Primary: LLM router.
    try:
        raw = core.call_nim(_build_router_prompt(message), max_tokens=512, temperature=0.0)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            tool = parsed.get("tool")
            args = parsed.get("args")
            rationale = parsed.get("rationale", "")
            if tool in _TOOL_NAMES:
                if not isinstance(args, dict):
                    args = {}
                args = _coerce_args(tool, args, message)
                return {
                    "tool": tool,
                    "args": args,
                    "rationale": str(rationale) if rationale else "LLM routing.",
                }
    except Exception:
        pass

    # Fallback: deterministic regex router.
    return _fallback_route(message)


def _coerce_args(tool: str, args: dict, message: str) -> dict:
    """Validate / normalise args produced by the LLM router."""
    out = {}
    if tool == "search_listings":
        out["query"] = str(args.get("query") or message)
    elif tool == "lookup_property":
        val = args.get("id_or_url")
        if val in (None, "", "null"):
            val = message
        out["id_or_url"] = str(val)
    elif tool == "operator_intel":
        name = args.get("name")
        out["name"] = "" if name in (None, "null") else str(name)
    elif tool == "top_cases":
        b = args.get("borough")
        t = args.get("tier")
        out["borough"] = None if b in (None, "", "null") else str(b)
        out["tier"] = None if t in (None, "", "null") else str(t)
        n = args.get("n", 20)
        try:
            out["n"] = int(n)
        except Exception:
            out["n"] = 20
    elif tool == "draft_enforcement":
        lid = args.get("listing_id")
        if lid in (None, "", "null"):
            out["listing_id"] = None
        else:
            try:
                out["listing_id"] = int(lid)
            except Exception:
                out["listing_id"] = _parse_listing_id(lid)
        b = args.get("borough")
        out["borough"] = None if b in (None, "", "null") else str(b)
    return out


# ── D) Tools ─────────────────────────────────────────────────────────────────

def search_listings(query: str) -> dict:
    """LLM NL->pandas search over the listings snapshot."""
    try:
        listings = core.load_data()[0]
    except Exception as e:
        return {"kind": "error", "message": f"Could not load data: {e}"}

    # Stage 1 — generate the pandas expression.
    try:
        raw = core.call_nim(build_filter_prompt(query))
    except Exception as e:
        return {"kind": "error", "message": f"Nemotron call failed: {e}"}
    expr = _clean_expr(raw)

    # Stage 2 — execute, guarded exactly like app.py.
    try:
        evaled = eval(expr, {"df": listings, "pd": pd})  # noqa: S307
        if isinstance(evaled, pd.DataFrame):
            result = evaled
        elif isinstance(evaled, pd.Series) and evaled.dtype == bool:
            result = listings[evaled]
        elif isinstance(evaled, pd.Series):
            result = listings.loc[evaled.index]
        else:
            result = listings[evaled]
        if len(result.columns) < 10:
            result = listings.loc[result.index]
        result = result.sort_values("confidence_score", ascending=False)
    except Exception as e:
        return {
            "kind": "error",
            "message": f"Filter expression failed: {e} | expression: {expr}",
        }

    stats = _table_stats(result)
    return {
        "kind": "table",
        "title": f"Search: {query}",
        "expr": expr,
        "df": result,
        "stats": stats,
    }


def lookup_property(id_or_url: str) -> dict:
    """Full intelligence on a single property by id or URL."""
    listing_id = _parse_listing_id(id_or_url)
    if listing_id is None:
        return {
            "kind": "error",
            "message": (
                "Could not parse a listing ID. Expected an Airbnb URL like "
                "https://www.airbnb.com/rooms/12345 or a numeric id."
            ),
        }

    try:
        listings = core.load_data()[0]
    except Exception as e:
        return {"kind": "error", "message": f"Could not load data: {e}"}

    match = listings[listings["id"] == listing_id]

    if not match.empty:
        r = match.iloc[0]
        useful = [
            "id", "host_name", "neighbourhood_cleansed", "confidence_score",
            "priority_tier", "nights_booked", "portfolio_size",
            "operator_total_listings", "operator_name", "property_type",
            "room_type", "price", "minimum_nights", "number_of_reviews",
            "reviews_per_month", "host_is_superhost", "license", "latitude",
            "longitude", "description", "host_about",
            "f1_nights_score", "f2_portfolio_score", "f3_review_velocity",
            "f4_booking_density", "f5_pricing_score", "f6_availability_score",
            "f7_commercial_language", "f8_review_anomaly",
        ]
        row = {}
        for c in useful:
            if c in match.columns:
                val = r.get(c)
                try:
                    if pd.isna(val):
                        val = None
                except (TypeError, ValueError):
                    pass
                row[c] = val
        return {
            "kind": "property",
            "listing_id": listing_id,
            "in_snapshot": True,
            "row": row,
            "scrape": None,
            "ml": None,
        }

    # Not in snapshot — live scrape + ML.
    scrape = core.cached_scrape(listing_id)
    ml = None
    try:
        bundle = core.load_ghost_model()
        if bundle is not None:
            feats = (scrape or {}).get("features", {}) or {}
            ml = core.ghost_predict.predict_one(feats, bundle=bundle)
    except Exception:
        ml = None

    return {
        "kind": "property",
        "listing_id": listing_id,
        "in_snapshot": False,
        "row": None,
        "scrape": scrape,
        "ml": ml,
    }


def operator_intel(name: str = "") -> dict:
    """Commercial-operator intelligence — by name, or top operators."""
    try:
        operators = core.load_data()[1]
    except Exception as e:
        return {"kind": "error", "message": f"Could not load data: {e}"}

    name = "" if name is None else str(name).strip()
    try:
        subset = operators
        if name:
            mask = (
                subset["operator_name"].astype(str)
                .str.contains(name, case=False, na=False, regex=False)
            )
            subset = subset[mask]
        subset = subset.sort_values("max_confidence", ascending=False).head(50)
    except Exception as e:
        return {"kind": "error", "message": f"Operator filter failed: {e}"}

    key_cols = [
        "operator_name", "total_listings", "listings_above_90_nights",
        "avg_confidence", "max_confidence", "total_estimated_revenue",
        "company_name", "company_status", "match_score",
    ]
    cols = [c for c in key_cols if c in subset.columns]
    return {
        "kind": "operator",
        "query": name,
        "df": subset[cols],
    }


def top_cases(borough: str = None, tier: str = None, n: int = 20) -> dict:
    """Highest-confidence listings, optionally filtered by borough and/or tier."""
    try:
        listings = core.load_data()[0]
    except Exception as e:
        return {"kind": "error", "message": f"Could not load data: {e}"}

    try:
        n = int(n)
    except Exception:
        n = 20

    result = listings
    title_bits = []

    if borough:
        try:
            uniques = listings["neighbourhood_cleansed"].dropna().astype(str).unique()
            resolved = None
            for u in uniques:
                if u.lower() == str(borough).strip().lower():
                    resolved = u
                    break
            if resolved is None:
                for u in uniques:
                    if str(borough).strip().lower() in u.lower():
                        resolved = u
                        break
            if resolved is not None:
                result = result[result["neighbourhood_cleansed"].astype(str) == resolved]
                title_bits.append(resolved)
            else:
                title_bits.append(str(borough))
                result = result[
                    result["neighbourhood_cleansed"].astype(str)
                    .str.contains(str(borough), case=False, na=False)
                ]
        except Exception as e:
            return {"kind": "error", "message": f"Borough filter failed: {e}"}

    if tier:
        try:
            result = result[result["priority_tier"].astype(str) == str(tier)]
            title_bits.append(f"{tier} tier")
        except Exception as e:
            return {"kind": "error", "message": f"Tier filter failed: {e}"}

    try:
        result = result.sort_values("confidence_score", ascending=False).head(n)
    except Exception as e:
        return {"kind": "error", "message": f"Sort failed: {e}"}

    scope = " · ".join(title_bits) if title_bits else "all London"
    return {
        "kind": "table",
        "title": f"Top {n} cases — {scope}",
        "df": result,
        "stats": _table_stats(result),
    }


def _make_enforcement_prompt(row) -> str:
    """make_prompt template, verbatim from app.py's tab_enforce."""
    nights = row.get("nights_booked", 0)
    overage = max(0, nights - 90)
    return f"""You are a housing enforcement officer for {row.get("neighbourhood_cleansed", "London")} Council.

EVIDENCE:
- Airbnb listing ID: {row.get("id")}
- Host name: {row.get("host_name")}
- Location: {row.get("neighbourhood_cleansed")}, London
- Estimated nights booked (last 365 days): {nights:.0f}
- Legal annual limit: 90 nights (Deregulation Act 2015, Section 44)
- Nights over limit: {overage:.0f}
- Operator portfolio size: {row.get("operator_total_listings", 1)} properties
- Algorithmic confidence score: {row.get("confidence_score", 0):.2f} / 1.00

Generate three things:

1. ENFORCEMENT NOTICE (formal language, cite Deregulation Act 2015 s.44 and Housing Act 2004)
2. EVIDENCE SUMMARY (3 sentences for the case file)
3. RECOMMENDED ACTION (choose one: Formal Warning / Fixed Penalty Notice £1,000 / Prosecution)

Format with these exact headers:
## ENFORCEMENT NOTICE
## EVIDENCE SUMMARY
## RECOMMENDED ACTION
"""


def draft_enforcement(listing_id=None, borough=None) -> dict:
    """Draft an enforcement package and persist it to ENFORCE_DIR."""
    try:
        listings = core.load_data()[0]
    except Exception as e:
        return {"kind": "error", "message": f"Could not load data: {e}"}

    # Resolve the target row.
    row = None
    try:
        if listing_id is not None:
            lid = _parse_listing_id(listing_id) if not isinstance(listing_id, int) else listing_id
            if lid is not None:
                match = listings[listings["id"] == lid]
                if not match.empty:
                    row = match.iloc[0]
        if row is None and borough:
            uniques = listings["neighbourhood_cleansed"].dropna().astype(str).unique()
            resolved = None
            for u in uniques:
                if u.lower() == str(borough).strip().lower():
                    resolved = u
                    break
            if resolved is None:
                for u in uniques:
                    if str(borough).strip().lower() in u.lower():
                        resolved = u
                        break
            sub = listings
            if resolved is not None:
                sub = listings[listings["neighbourhood_cleansed"].astype(str) == resolved]
            if not sub.empty:
                row = sub.sort_values("confidence_score", ascending=False).iloc[0]
        if row is None:
            pri = listings[listings["priority_tier"].astype(str) == "Priority"]
            base = pri if not pri.empty else listings
            if base.empty:
                return {"kind": "error", "message": "No listings available to draft from."}
            row = base.sort_values("confidence_score", ascending=False).iloc[0]
    except Exception as e:
        return {"kind": "error", "message": f"Could not resolve a case: {e}"}

    # Draft via the LLM.
    try:
        text = core.call_nim(_make_enforcement_prompt(row), max_tokens=4096, temperature=0.2)
    except Exception as e:
        return {"kind": "error", "message": f"Nemotron call failed: {e}"}

    # Persist exactly like app.py.
    saved = None
    try:
        out_dir = core.ENFORCE_DIR
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"case_{row.get('id')}_dashboard.json"
        with open(fname, "w") as fh:
            json.dump(
                {
                    "listing_id": str(row.get("id")),
                    "host_name": str(row.get("host_name")),
                    "borough": str(row.get("neighbourhood_cleansed")),
                    "nights_booked": float(row.get("nights_booked", 0) or 0),
                    "score": float(row.get("confidence_score", 0) or 0),
                    "output": text,
                },
                fh, indent=2,
            )
        saved = str(fname)
    except Exception:
        saved = None

    return {
        "kind": "enforcement",
        "listing_id": row.get("id"),
        "host": str(row.get("host_name")),
        "borough": str(row.get("neighbourhood_cleansed")),
        "text": text,
        "saved": saved,
    }


# ── E) Dispatcher ────────────────────────────────────────────────────────────

_DISPATCH = {
    "search_listings": search_listings,
    "lookup_property": lookup_property,
    "operator_intel": operator_intel,
    "top_cases": top_cases,
    "draft_enforcement": draft_enforcement,
}

_ALLOWED_ARGS = {
    "search_listings": {"query"},
    "lookup_property": {"id_or_url"},
    "operator_intel": {"name"},
    "top_cases": {"borough", "tier", "n"},
    "draft_enforcement": {"listing_id", "borough"},
}


def run_tool(tool: str, args: dict) -> dict:
    """Dispatch to a tool by name, defensively filtering args. Never raises."""
    fn = _DISPATCH.get(tool)
    if fn is None:
        return {"kind": "error", "message": f"Unknown tool: {tool!r}"}
    if not isinstance(args, dict):
        args = {}
    allowed = _ALLOWED_ARGS.get(tool, set())
    safe = {k: v for k, v in args.items() if k in allowed}
    try:
        return fn(**safe)
    except Exception as e:
        return {"kind": "error", "message": f"{tool} failed: {e}"}
