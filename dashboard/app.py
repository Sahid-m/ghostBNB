"""
Ghost Hotel Detector — Officer Dashboard
/home/nvidia/AIR_BNB/dashboard/app.py

Run: streamlit run dashboard/app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
import sys
from pathlib import Path

# Make the sibling `pipeline/` package importable (scraper + ML predictor)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.airbnb_scraper import fetch_listing
from pipeline import predict as ghost_predict
from pipeline.features import NUMERIC, CATEGORICAL, BOOL

st.set_page_config(
    page_title="Ghost Hotel Detector",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

SCORED    = Path("/home/nvidia/AIR_BNB/data/scored")
BENCH     = Path("/home/nvidia/AIR_BNB/benchmarks/calendar_benchmark.json")
NIM_URL   = "http://10.18.216.24:8090/v1/chat/completions"
MODEL     = "nemotron-nano-vl"

TIER_COLOURS = {
    "Priority": "#e63946",
    "Review":   "#f4a261",
    "Monitor":  "#457b9d",
    "Low":      "#a8dadc",
}

# ── NIM helper — shared by all tabs ──────────────────────────────────────────

def nim_ready() -> bool:
    """True if the model has finished loading and can accept requests."""
    try:
        r = requests.get("http://10.18.216.24:8090/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def call_nim(prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """
    Call Nemotron with automatic retry on 503 (model still loading).
    Raises RuntimeError if unavailable after retries.
    """
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
                timeout=120,
            )
            if resp.status_code == 503:
                wait = 5 * (attempt + 1)
                if attempt < 3:
                    time.sleep(wait)
                    continue
                raise RuntimeError("Model is still loading — try again in ~30 seconds")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"NIM error: {e}") from e
    raise RuntimeError("NIM unavailable after retries")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
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
    """Load the ML classifier bundle once. Returns None if not trained yet."""
    try:
        return ghost_predict.load_bundle()
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def cached_scrape(listing_id: int):
    """Best-effort live scrape, cached per listing so reruns don't re-fetch."""
    return fetch_listing(str(listing_id))

listings, operators, priority = load_data()
bench = load_benchmark()
ghost_bundle = load_ghost_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/London_coat_of_arms.svg/200px-London_coat_of_arms.svg.png", width=60)
    st.title("Ghost Hotel Detector")
    st.caption("London STR Enforcement Intelligence")

    st.divider()

    boroughs = sorted(listings["neighbourhood_cleansed"].dropna().unique())
    selected_boroughs = st.multiselect(
        "Borough filter",
        options=boroughs,
        default=[],
        placeholder="All boroughs",
    )

    tiers = st.multiselect(
        "Priority tier",
        options=["Priority", "Review", "Monitor", "Low"],
        default=["Priority", "Review"],
    )

    min_score = st.slider("Min confidence score", 0.0, 0.97, 0.75, 0.01)
    min_portfolio = st.slider("Min operator portfolio size", 1, 50, 1)

    st.divider()
    if nim_ready():
        st.success("Nemotron: ready", icon="🟢")
    else:
        st.warning("Nemotron: loading…", icon="🟡")
        st.caption("AI features will retry automatically. Reload in ~30s.")

    st.divider()
    st.markdown("**GPU Benchmark**")
    col1, col2 = st.columns(2)
    col1.metric("CPU", f"{bench['cpu_seconds']:.1f}s")
    col2.metric("GPU", f"{bench['gpu_seconds']:.1f}s", f"{bench['speedup']:.1f}x faster")
    st.caption("35.4M calendar rows | cuDF on GB10")

# ── Filter data ───────────────────────────────────────────────────────────────

view = listings.copy()
if selected_boroughs:
    view = view[view["neighbourhood_cleansed"].isin(selected_boroughs)]
if tiers:
    view = view[view["priority_tier"].isin(tiers)]
view = view[view["confidence_score"] >= min_score]
view = view[view["portfolio_size"] >= min_portfolio]

# ── Header metrics ────────────────────────────────────────────────────────────

st.markdown("## 🏚️ Ghost Hotel Detector — London")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Listings scored",       f"{len(listings):,}")
m2.metric("Priority cases",        f"{(listings['priority_tier']=='Priority').sum():,}", "score > 0.90")
m3.metric("Review cases",          f"{(listings['priority_tier']=='Review').sum():,}",   "score 0.75–0.90")
m4.metric("Commercial operators",  f"{operators['canonical_operator_id'].nunique():,}")
m5.metric("CH matches",            f"{operators['company_number'].notna().sum():,}", "Companies House")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_map, tab_cases, tab_ops, tab_enforce, tab_search, tab_lookup = st.tabs([
    "🗺️  Map", "📋  Priority Cases", "🏢  Operators", "⚖️  Enforcement", "🔍  AI Search", "🔗  Property Lookup"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ════════════════════════════════════════════════════════════════════════════

with tab_map:
    st.markdown(f"**{len(view):,} listings** match current filters")

    map_data = view.dropna(subset=["latitude", "longitude"]).copy()
    # Sample for rendering performance if very large
    if len(map_data) > 3000:
        map_data = map_data.nlargest(3000, "confidence_score")

    m = folium.Map(location=[51.505, -0.09], zoom_start=10, tiles="CartoDB positron")

    # Borough heat — count priority listings per borough
    borough_counts = (
        view[view["priority_tier"].isin(["Priority", "Review"])]
        .groupby("neighbourhood_cleansed")
        .size()
        .reset_index(name="count")
    )

    # Plot individual dots
    for _, row in map_data.iterrows():
        tier = str(row.get("priority_tier", "Low"))
        colour = TIER_COLOURS.get(tier, "#888")
        radius  = 4 if tier == "Priority" else 3

        popup_html = (
            f"<b>{row.get('host_name','')}</b><br>"
            f"{row.get('neighbourhood_cleansed','')}<br>"
            f"Score: <b>{row['confidence_score']:.2f}</b> | {tier}<br>"
            f"Nights: {row.get('nights_booked',0):.0f} | "
            f"Portfolio: {row.get('portfolio_size',1)}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row.get('host_name','')} — {row['confidence_score']:.2f}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;background:white;
                padding:10px 14px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
                font-size:12px;z-index:9999;">
      <b>Confidence tier</b><br>
      <span style="color:#e63946;">●</span> Priority (&gt;0.90)<br>
      <span style="color:#f4a261;">●</span> Review (0.75–0.90)<br>
      <span style="color:#457b9d;">●</span> Monitor (0.50–0.75)<br>
      <span style="color:#a8dadc;">●</span> Low (&lt;0.50)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=560, returned_objects=[])

    # Borough bar chart
    st.markdown("#### Priority + Review cases by borough")
    if not borough_counts.empty:
        fig = px.bar(
            borough_counts.sort_values("count", ascending=True).tail(20),
            x="count", y="neighbourhood_cleansed",
            orientation="h",
            color="count",
            color_continuous_scale="Reds",
            labels={"count": "Cases", "neighbourhood_cleansed": ""},
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRIORITY CASES
# ════════════════════════════════════════════════════════════════════════════

with tab_cases:
    display_cols = [
        "id", "host_name", "neighbourhood_cleansed",
        "nights_booked", "confidence_score", "priority_tier",
        "portfolio_size", "operator_total_listings",
        "f1_nights_score", "f2_portfolio_score",
        "f3_review_velocity", "f4_booking_density",
    ]
    show = view[[c for c in display_cols if c in view.columns]].sort_values(
        "confidence_score", ascending=False
    )

    st.markdown(f"**{len(show):,} listings** | sorted by confidence score")

    col_cfg = {
        "confidence_score":    st.column_config.ProgressColumn("Score",    min_value=0, max_value=1, format="%.2f"),
        "nights_booked":       st.column_config.NumberColumn("Nights",     format="%d"),
        "portfolio_size":      st.column_config.NumberColumn("Portfolio",  format="%d"),
        "f1_nights_score":     st.column_config.ProgressColumn("F1 nights",  min_value=0, max_value=1, format="%.2f"),
        "f2_portfolio_score":  st.column_config.ProgressColumn("F2 portfolio", min_value=0, max_value=1, format="%.2f"),
        "f3_review_velocity":  st.column_config.ProgressColumn("F3 reviews",  min_value=0, max_value=1, format="%.2f"),
        "f4_booking_density":  st.column_config.ProgressColumn("F4 calendar", min_value=0, max_value=1, format="%.2f"),
        "neighbourhood_cleansed": st.column_config.TextColumn("Borough"),
        "host_name":           st.column_config.TextColumn("Host"),
        "priority_tier":       st.column_config.TextColumn("Tier"),
    }

    st.dataframe(show.head(500), column_config=col_cfg, use_container_width=True, height=520)

    csv = show.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "priority_cases_filtered.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPERATORS
# ════════════════════════════════════════════════════════════════════════════

with tab_ops:
    ops_view = operators.copy()
    if selected_boroughs:
        ops_view = ops_view[
            ops_view["boroughs"].apply(
                lambda bl: any(b in selected_boroughs for b in (bl if isinstance(bl, list) else []))
            )
        ]
    ops_view = ops_view[ops_view["max_confidence"] >= min_score].sort_values(
        "max_confidence", ascending=False
    )

    st.markdown(f"**{len(ops_view):,} operators** | {ops_view['company_number'].notna().sum():,} matched to Companies House")

    op_cols = [
        "operator_name", "total_listings", "listings_above_90_nights",
        "avg_confidence", "max_confidence", "total_estimated_revenue",
        "company_name", "company_status", "match_score",
    ]
    op_cfg = {
        "max_confidence":           st.column_config.ProgressColumn("Max score",  min_value=0, max_value=1, format="%.2f"),
        "avg_confidence":           st.column_config.ProgressColumn("Avg score",  min_value=0, max_value=1, format="%.2f"),
        "total_listings":           st.column_config.NumberColumn("Listings",  format="%d"),
        "listings_above_90_nights": st.column_config.NumberColumn(">90 nights", format="%d"),
        "total_estimated_revenue":  st.column_config.NumberColumn("Est. revenue £", format="£%,.0f"),
        "match_score":              st.column_config.ProgressColumn("CH match %", min_value=0, max_value=100, format="%.0f"),
        "operator_name":            st.column_config.TextColumn("Operator"),
        "company_name":             st.column_config.TextColumn("Company (CH)"),
        "company_status":           st.column_config.TextColumn("Status"),
    }

    show_ops = ops_view[[c for c in op_cols if c in ops_view.columns]].head(500)
    st.dataframe(show_ops, column_config=op_cfg, use_container_width=True, height=480)

    # Revenue vs portfolio scatter
    st.markdown("#### Revenue vs portfolio size (top 200 operators)")
    scatter_data = ops_view.dropna(subset=["total_estimated_revenue"]).head(200)
    if not scatter_data.empty:
        fig2 = px.scatter(
            scatter_data,
            x="total_listings",
            y="total_estimated_revenue",
            size="max_confidence",
            color="max_confidence",
            color_continuous_scale="Reds",
            hover_name="operator_name",
            hover_data={"company_name": True, "listings_above_90_nights": True},
            labels={
                "total_listings": "Total listings",
                "total_estimated_revenue": "Estimated annual revenue (£)",
            },
        )
        fig2.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0),
                           coloraxis_colorbar_title="Max score")
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ENFORCEMENT GENERATOR
# ════════════════════════════════════════════════════════════════════════════

with tab_enforce:
    st.markdown("### AI Enforcement Notice Generator")
    st.caption("Powered by Nemotron Nano (30B) on NVIDIA GB10 · Deregulation Act 2015 s.44")

    # Case selector
    priority_cases = listings[listings["priority_tier"] == "Priority"].sort_values(
        "confidence_score", ascending=False
    ).head(100)

    case_options = {
        f"{row['host_name']} — {row['neighbourhood_cleansed']} — score {row['confidence_score']:.2f} — {row['nights_booked']:.0f} nights": idx
        for idx, row in priority_cases.iterrows()
    }

    selected_label = st.selectbox("Select a priority case", options=list(case_options.keys()))
    selected_idx   = case_options[selected_label]
    case_row       = priority_cases.loc[selected_idx]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Nights booked",    f"{case_row.get('nights_booked', 0):.0f}")
        st.metric("Confidence score", f"{case_row.get('confidence_score', 0):.2f}")
    with col_b:
        st.metric("Portfolio size",   f"{case_row.get('portfolio_size', 1):.0f}")
        st.metric("Borough",          str(case_row.get("neighbourhood_cleansed", "")))

    def make_prompt(row):
        nights  = row.get("nights_booked", 0)
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

    if st.button("⚖️  Generate Enforcement Notice", type="primary", use_container_width=True):
        with st.spinner("Nemotron drafting enforcement package..."):
            try:
                output = call_nim(make_prompt(case_row), max_tokens=4096, temperature=0.2)
                st.success("Notice generated")
                st.markdown("---")
                st.markdown(output)

                # Save to disk
                out_dir = Path("/home/nvidia/AIR_BNB/data/enforcement")
                out_dir.mkdir(exist_ok=True)
                fname = out_dir / f"case_{case_row.get('id')}_dashboard.json"
                json.dump(
                    {
                        "listing_id":    str(case_row.get("id")),
                        "host_name":     str(case_row.get("host_name")),
                        "borough":       str(case_row.get("neighbourhood_cleansed")),
                        "nights_booked": float(case_row.get("nights_booked", 0)),
                        "score":         float(case_row.get("confidence_score", 0)),
                        "output":        output,
                    },
                    open(fname, "w"), indent=2
                )
                st.caption(f"Saved → {fname.name}")

            except Exception as e:
                st.error(f"NIM call failed: {e}")

    st.divider()

    # Show pre-generated cases
    enforce_dir = Path("/home/nvidia/AIR_BNB/data/enforcement")
    pre_generated = sorted(enforce_dir.glob("case_*.json"))
    if pre_generated:
        st.markdown(f"**{len(pre_generated)} pre-generated cases** (from batch run)")
        selected_file = st.selectbox(
            "View a pre-generated case",
            options=[f.name for f in pre_generated],
        )
        if selected_file:
            case_data = json.loads((enforce_dir / selected_file).read_text())
            st.markdown(f"**{case_data.get('host_name')} — {case_data.get('borough')}** "
                        f"| {case_data.get('nights_booked', 0):.0f} nights | score {case_data.get('score', case_data.get('confidence_score', 0)):.2f}")
            st.markdown("---")
            st.markdown(case_data.get("enforcement_output", case_data.get("output", "")))

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI NATURAL LANGUAGE SEARCH
# ════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.markdown("### AI-Powered Case Search")
    st.caption("Ask anything about the dataset — Nemotron reasons through it live")

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

    # ── Streaming helpers ─────────────────────────────────────────────────────

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

    # ── Example chips ─────────────────────────────────────────────────────────

    EXAMPLES = [
        "Westminster operators with more than 5 properties and over 200 nights",
        "Priority cases in Tower Hamlets or Islington",
        "Hosts with 10+ listings and commercial language above 0.7",
        "Entire homes in Kensington and Chelsea scoring above 0.85",
        "Operators estimated to earn over £50,000 a year",
        "High review anomaly — burst score with big portfolios",
        "Who are the top 20 most suspicious operators across all boroughs?",
        "Show me listings that breach 90 nights AND have commercial language",
    ]

    st.markdown("**Quick searches:**")
    ex_cols = st.columns(4)
    clicked = None
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
            clicked = ex

    st.markdown("---")
    query = st.text_input(
        "Ask anything:",
        value=clicked or "",
        placeholder="e.g. who are the worst offenders in Westminster?",
    )

    col_cfg = {
        "confidence_score":        st.column_config.ProgressColumn("Score",       min_value=0, max_value=1, format="%.2f"),
        "f7_commercial_language":  st.column_config.ProgressColumn("F7 language", min_value=0, max_value=1, format="%.2f"),
        "f8_review_anomaly":       st.column_config.ProgressColumn("F8 anomaly",  min_value=0, max_value=1, format="%.2f"),
        "nights_booked":           st.column_config.NumberColumn("Nights",        format="%d"),
        "portfolio_size":          st.column_config.NumberColumn("Portfolio",     format="%d"),
        "estimated_revenue_l365d": st.column_config.NumberColumn("Revenue £",    format="£%,.0f"),
        "neighbourhood_cleansed":  st.column_config.TextColumn("Borough"),
        "host_name":               st.column_config.TextColumn("Host"),
        "priority_tier":           st.column_config.TextColumn("Tier"),
    }

    DISPLAY_COLS = [
        "host_name", "neighbourhood_cleansed", "confidence_score", "priority_tier",
        "nights_booked", "portfolio_size", "operator_total_listings",
        "f7_commercial_language", "f8_review_anomaly", "estimated_revenue_l365d",
    ]

    if st.button("🔍  Search", type="primary", use_container_width=True) and query:

        # ── Stage 1: stream the reasoning + extract expression ────────────────
        st.markdown("#### Nemotron reasoning")
        reasoning_box  = st.empty()
        expression_box = st.empty()

        reasoning_text  = ""
        content_text    = ""

        try:
            for kind, chunk in stream_nim(build_filter_prompt(query)):
                if kind == "reasoning":
                    reasoning_text += chunk
                    reasoning_box.markdown(
                        f"<div style='background:#1e1e2e;color:#cdd6f4;padding:10px 14px;"
                        f"border-radius:6px;font-size:12px;font-family:monospace;"
                        f"max-height:220px;overflow-y:auto;white-space:pre-wrap'>"
                        f"{reasoning_text}</div>",
                        unsafe_allow_html=True,
                    )
                elif kind == "content":
                    content_text += chunk

            # Clean expression — strip fences, backticks, leading labels
            expr = content_text.strip()
            expr = re.sub(r"```(?:python)?\s*", "", expr).strip("`").strip()
            if "Expression:" in expr:
                expr = expr.split("Expression:")[-1].strip().strip("`")
            # Find the line that looks like a pandas expression (starts with df)
            lines = [l.strip().strip("`") for l in expr.splitlines() if l.strip()]
            df_lines = [l for l in lines if l.startswith("df")]
            expr = df_lines[-1] if df_lines else (lines[-1] if lines else expr)

            expression_box.code(expr, language="python")

        except Exception as e:
            st.error(f"Nemotron stream error: {e}")
            st.stop()

        # ── Stage 2: execute ──────────────────────────────────────────────────
        try:
            evaled = eval(expr, {"df": listings, "pd": pd})  # noqa: S307
            if isinstance(evaled, pd.DataFrame):
                result = evaled
            elif isinstance(evaled, pd.Series) and evaled.dtype == bool:
                result = listings[evaled]
            elif isinstance(evaled, pd.Series):
                # Model returned a column — use its index to filter rows
                result = listings.loc[evaled.index]
            else:
                result = listings[evaled]
            # Reset to full columns if model accidentally dropped some
            if len(result.columns) < 10:
                result = listings.loc[result.index]
            result = result.sort_values("confidence_score", ascending=False)
        except Exception as e:
            st.error(f"Filter expression failed: {e}")
            st.caption(f"Expression attempted: `{expr}`")
            st.stop()

        if result.empty:
            st.warning("No listings matched. Try broadening the query.")
            st.stop()

        # ── Stage 3: metrics ──────────────────────────────────────────────────
        st.markdown(f"#### Results — {len(result):,} listings")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Matched",       f"{len(result):,}")
        r2.metric("Priority",      f"{(result['priority_tier']=='Priority').sum():,}")
        r3.metric("Avg score",     f"{result['confidence_score'].mean():.2f}")
        r4.metric("Avg nights",    f"{result['nights_booked'].mean():.0f}")

        show_cols = [c for c in DISPLAY_COLS if c in result.columns]
        st.dataframe(result[show_cols].head(500),
                     column_config=col_cfg, use_container_width=True, height=420)

        csv = result[show_cols].to_csv(index=False).encode()
        st.download_button("⬇  Download CSV", csv, "search_results.csv", "text/csv")

        # ── Stage 4: stream intelligence briefing ────────────────────────────
        st.markdown("#### Intelligence briefing")

        top5 = result[show_cols].head(5).to_string(index=False)
        borough_breakdown = result.groupby("neighbourhood_cleansed").size().nlargest(5).to_string()
        stats = (
            f"Total matched: {len(result)}\n"
            f"Priority tier: {(result['priority_tier']=='Priority').sum()}\n"
            f"Avg confidence: {result['confidence_score'].mean():.2f}\n"
            f"Avg nights booked: {result['nights_booked'].mean():.0f}\n"
            f"Top boroughs:\n{borough_breakdown}\n"
            f"Top 5 rows:\n{top5}"
        )

        briefing_box   = st.empty()
        briefing_text  = ""
        brief_reasoning = ""
        brief_reason_box = st.empty()

        for kind, chunk in stream_nim(build_insight_prompt(query, stats), temperature=0.3):
            if kind == "reasoning":
                brief_reasoning += chunk
                brief_reason_box.markdown(
                    f"<details><summary style='font-size:12px;color:#888'>reasoning</summary>"
                    f"<pre style='font-size:11px;color:#aaa;white-space:pre-wrap'>{brief_reasoning}</pre></details>",
                    unsafe_allow_html=True,
                )
            elif kind == "content":
                briefing_text += chunk
                briefing_box.markdown(briefing_text)

    # ── Score reference ───────────────────────────────────────────────────────
    with st.expander("Feature score reference"):
        st.markdown("""
| Score | Signal |
|---|---|
| **F1** nights breach | Estimated nights vs 90-night cap |
| **F2** portfolio | Listings under same host |
| **F3** review velocity | 2024 reviews vs borough median |
| **F4** calendar density | % booked days Sept 2024+ |
| **F7** commercial language | Nemotron: description + bio |
| **F8** review anomaly | Burst timing, reviewer reuse |
""")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — PROPERTY LOOKUP
# ════════════════════════════════════════════════════════════════════════════

with tab_lookup:
    st.markdown("### Property Lookup")
    st.caption("Paste any Airbnb listing URL to get full intelligence on that property")

    url_input = st.text_input(
        "Airbnb listing URL",
        placeholder="https://www.airbnb.com/rooms/13913",
    )

    def extract_listing_id(url: str):
        """Pull numeric listing ID from any Airbnb URL format."""
        import re
        url = url.strip()
        # Match /rooms/<id> with optional query params
        m = re.search(r"/rooms/(\d+)", url)
        if m:
            return int(m.group(1))
        # Plain numeric ID pasted directly
        if url.isdigit():
            return int(url)
        return None

    def nim_property_summary(row: pd.Series, operator_row=None) -> str:
        nights   = float(row.get("nights_booked", 0) or 0)
        overage  = max(0, nights - 90)
        op_size  = int(row.get("operator_total_listings", 1) or 1)
        score    = float(row.get("confidence_score", 0) or 0)
        f7       = float(row.get("f7_commercial_language", 0.5) or 0.5)
        f8       = float(row.get("f8_review_anomaly", 0) or 0)
        borough  = str(row.get("neighbourhood_cleansed", "London") or "London")
        host     = str(row.get("host_name", "Unknown") or "Unknown")
        prop_type = str(row.get("property_type", "") or "")
        description = str(row.get("description", "") or "")[:500]
        host_bio    = str(row.get("host_about", "") or "")[:300]
        ch_match = ""
        if operator_row is not None and pd.notna(operator_row.get("company_name")):
            ch_match = (
                f"Companies House match: {operator_row['company_name']} "
                f"({operator_row.get('company_status','')}, "
                f"score {operator_row.get('match_score',0):.0f}%)"
            )

        prompt = f"""You are an expert housing enforcement analyst. Analyse this Airbnb property and write an intelligence briefing.

PROPERTY DATA:
- Host: {host}
- Borough: {borough}
- Property type: {prop_type}
- Estimated nights booked (last 12 months): {nights:.0f}
- Nights over 90-night legal limit: {overage:.0f}
- Operator portfolio (total listings): {op_size}
- Ghost hotel confidence score: {score:.2f} / 0.97
- Commercial language score (F7): {f7:.2f} (0=personal, 1=commercial)
- Review anomaly score (F8): {f8:.2f}
- {ch_match}

LISTING DESCRIPTION (excerpt): {description}
HOST BIO (excerpt): {host_bio}

Write a 4-section intelligence briefing:

## PROPERTY PROFILE
Brief summary of what this property is and how it is operated.

## RISK INDICATORS
Bullet list of specific red flags found in the data and listing text.

## LEGAL EXPOSURE
Cite Deregulation Act 2015 s.44. State estimated nights over limit. Assess likelihood of prosecution success.

## RECOMMENDED ACTION
One of: No Action / Intelligence File / Formal Warning / Fixed Penalty Notice £1,000 / Prosecution
With one sentence of justification.
"""
        try:
            return call_nim(prompt, max_tokens=4096, temperature=0.3)
        except RuntimeError as e:
            return str(e)

    if url_input:
        listing_id = extract_listing_id(url_input)

        if listing_id is None:
            st.error("Could not parse a listing ID from that URL. Expected format: https://www.airbnb.com/rooms/12345")
        else:
            st.caption(f"Looking up listing ID: **{listing_id}**")

            match = listings[listings["id"] == listing_id]

            if match.empty:
                st.warning(
                    f"⚠️ Listing **{listing_id}** is **not in our 25 Sep 2025 snapshot**. "
                    "It was likely created after our data cut-off (or removed before the scrape / "
                    "below our pre-filter threshold). We have no history for it — so we pull it "
                    "**live from Airbnb** and score it with the ML classifier instead."
                )
                st.markdown(f"[View on Airbnb ↗](https://www.airbnb.com/rooms/{listing_id})")

                # ── 1. Live scrape (best-effort, never fatal) ─────────────
                with st.spinner("Pulling live data from Airbnb…"):
                    scraped = cached_scrape(listing_id)

                scraped_feats = scraped.get("features", {}) if scraped else {}
                if scraped and scraped.get("ok") and scraped_feats:
                    st.success(f"Live data pulled — {len(scraped_feats)} fields read from the listing page.")
                else:
                    reason = (scraped or {}).get("reason", "unknown")
                    st.info(
                        f"Couldn't read live listing data automatically ({reason}). "
                        "Airbnb blocks server-side scraping — enter the key fields manually below and we'll still score it."
                    )

                # ── 2. Feature entry form (pre-filled from scrape) ────────
                ML_FIELDS = NUMERIC + BOOL + CATEGORICAL
                st.markdown("#### Listing features for ML scoring")
                st.caption("Pre-filled from the live page where possible — correct or complete any field, then score.")

                entered = {}
                with st.form("ml_lookup_form"):
                    cols = st.columns(3)
                    for i, f in enumerate(ML_FIELDS):
                        prefill = scraped_feats.get(f, "")
                        prefill = "" if prefill is None else str(prefill)
                        entered[f] = cols[i % 3].text_input(f, value=prefill, key=f"ml_{f}")
                    submitted = st.form_submit_button("🤖  Predict ghost-hotel risk",
                                                      type="primary", use_container_width=True)

                # ── 3. ML prediction ─────────────────────────────────────
                if submitted:
                    if ghost_bundle is None:
                        st.error("ML model not trained yet. Run `python3 pipeline/train_model.py` first.")
                    else:
                        feats = {k: v for k, v in entered.items() if str(v).strip() != ""}
                        result = ghost_predict.predict_one(feats, bundle=ghost_bundle)
                        prob = result["prob"]
                        tier = result["tier"]
                        tcol = TIER_COLOURS.get(tier, "#888")
                        st.markdown(
                            f"<h3 style='color:{tcol}'>🤖 ML risk: {tier} — {prob*100:.1f}% ghost-hotel likelihood</h3>",
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(prob, 0.0), 1.0))

                        st.markdown("**Top factors driving this score** (occlusion analysis):")
                        for fac in result["top_factors"]:
                            arrow = "🔺" if fac["contribution"] > 0 else "🔻"
                            direction = "raises" if fac["contribution"] > 0 else "lowers"
                            st.markdown(
                                f"- {arrow} **{fac['feature']}** = {fac['value']} "
                                f"— {direction} risk by {abs(fac['contribution'])*100:.1f} pts"
                            )

                        st.caption(
                            "ℹ️ " + result.get("note", "")
                        )
            else:
                row = match.iloc[0]
                tier      = str(row.get("priority_tier", "Low"))
                score     = float(row.get("confidence_score", 0) or 0)
                nights    = float(row.get("nights_booked", 0) or 0)
                tier_col  = TIER_COLOURS.get(tier, "#888")

                # ── Header ────────────────────────────────────────────────
                st.markdown(
                    f"<h3 style='color:{tier_col}'>⚠ {tier} — score {score:.2f}</h3>",
                    unsafe_allow_html=True,
                )

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Confidence",   f"{score:.2f}")
                col2.metric("Nights booked", f"{nights:.0f}",
                            f"{nights-90:.0f} over limit" if nights > 90 else "Within limit")
                col3.metric("Portfolio",    f"{int(row.get('portfolio_size',1) or 1)}")
                col4.metric("Borough",      str(row.get("neighbourhood_cleansed","") or ""))
                col5.metric("Host",         str(row.get("host_name","") or ""))

                st.markdown(f"[View on Airbnb ↗](https://www.airbnb.com/rooms/{listing_id})")

                st.divider()

                # ── Two-column layout: details + map ──────────────────────
                left, right = st.columns([1, 1])

                with left:
                    st.markdown("#### Scoring breakdown")
                    score_data = {
                        "F1 — Nights breach":       float(row.get("f1_nights_score", 0) or 0),
                        "F2 — Portfolio size":       float(row.get("f2_portfolio_score", 0) or 0),
                        "F3 — Review velocity":      float(row.get("f3_review_velocity", 0) or 0),
                        "F4 — Calendar density":     float(row.get("f4_booking_density", 0) or 0),
                        "F7 — Commercial language":  float(row.get("f7_commercial_language", 0.5) or 0.5),
                        "F8 — Review anomaly":       float(row.get("f8_review_anomaly", 0) or 0),
                    }
                    fig = px.bar(
                        x=list(score_data.values()),
                        y=list(score_data.keys()),
                        orientation="h",
                        range_x=[0, 1],
                        color=list(score_data.values()),
                        color_continuous_scale="Reds",
                    )
                    fig.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                                      showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Listing details")
                    detail_fields = [
                        ("Property type",    row.get("property_type", "")),
                        ("Room type",        row.get("room_type", "")),
                        ("Accommodates",     row.get("accommodates", "")),
                        ("Price/night",      row.get("price", "")),
                        ("Min nights",       row.get("minimum_nights", "")),
                        ("Host since",       row.get("host_since", "")),
                        ("Host listings",    row.get("host_total_listings_count", "")),
                        ("Reviews (total)",  row.get("number_of_reviews", "")),
                        ("Reviews/month",    row.get("reviews_per_month", "")),
                        ("Superhost",        row.get("host_is_superhost", "")),
                        ("Instant book",     row.get("instant_bookable", "")),
                        ("Licence field",    row.get("license", "None") or "None"),
                    ]
                    for label, val in detail_fields:
                        if val not in ("", None, float("nan")):
                            st.markdown(f"**{label}:** {val}")

                    # Operator + Companies House
                    op_name = row.get("operator_name", "")
                    op_size = int(row.get("operator_total_listings", 1) or 1)
                    if op_size > 1:
                        st.markdown(f"**Operator (entity-resolved):** {op_name} — {op_size} listings")

                    operators_df = pd.read_parquet(SCORED / "operator_summary_enriched.parquet")
                    op_id = row.get("canonical_operator_id")
                    op_row = None
                    if op_id is not None:
                        op_match = operators_df[operators_df["canonical_operator_id"] == op_id]
                        if not op_match.empty:
                            op_row = op_match.iloc[0]
                            if pd.notna(op_row.get("company_name")):
                                st.markdown(
                                    f"**Companies House:** {op_row['company_name']} · "
                                    f"{op_row.get('company_status','')} · "
                                    f"match {op_row.get('match_score',0):.0f}%"
                                )

                with right:
                    lat = row.get("latitude")
                    lng = row.get("longitude")
                    if pd.notna(lat) and pd.notna(lng):
                        st.markdown("#### Location")
                        m = folium.Map(location=[lat, lng], zoom_start=15,
                                       tiles="CartoDB positron")
                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=10,
                            color=tier_col,
                            fill=True,
                            fill_color=tier_col,
                            fill_opacity=0.9,
                            popup=f"{row.get('host_name','')} — score {score:.2f}",
                        ).add_to(m)
                        st_folium(m, width="100%", height=300, returned_objects=[])

                    # Description excerpt
                    desc = str(row.get("description", "") or "")
                    if desc:
                        st.markdown("#### Listing description")
                        st.markdown(
                            f"<div style='background:#f8f9fa;padding:10px;border-radius:6px;"
                            f"font-size:13px;max-height:200px;overflow-y:auto'>{desc[:800]}</div>",
                            unsafe_allow_html=True,
                        )

                    host_bio = str(row.get("host_about", "") or "")
                    if host_bio:
                        st.markdown("#### Host bio")
                        st.markdown(
                            f"<div style='background:#f8f9fa;padding:10px;border-radius:6px;"
                            f"font-size:13px;max-height:150px;overflow-y:auto'>{host_bio[:500]}</div>",
                            unsafe_allow_html=True,
                        )

                st.divider()

                # ── AI Intelligence Briefing ──────────────────────────────
                st.markdown("#### AI Intelligence Briefing")
                st.caption("Nemotron Nano analyses the property data and listing text")

                if st.button("Generate Intelligence Briefing", type="primary",
                             use_container_width=True, key="lookup_nim"):
                    with st.spinner("Nemotron analysing property..."):
                        briefing = nim_property_summary(row, op_row)
                        st.markdown(briefing)
