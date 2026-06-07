"""
Ghost Hotel Detector — Agentic Officer Console
/home/nvidia/AIR_BNB/dashboard/app.py

Run: streamlit run dashboard/app.py --server.port 8501

Two surfaces:
  • Dashboard  — the at-a-glance intelligence overview (KPIs, map, boroughs, operators)
  • Agent      — a conversational console. Ask in plain English; the agent routes
                 your request to the right tool (search, property lookup + ML,
                 operator intel, top cases, enforcement drafting) and shows its work.
"""
import json

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

import core
import ui
import agent

st.set_page_config(
    page_title="Ghost Hotel Detector",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()

# ── Data ────────────────────────────────────────────────────────────────────
listings, operators, priority = core.load_data()
bench = core.load_benchmark()
ghost_bundle = core.load_ghost_model()
TIER_COLOURS = core.TIER_COLOURS

# Shared dataframe column configs / display columns
COL_CFG = {
    "confidence_score":        st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
    "f7_commercial_language":  st.column_config.ProgressColumn("F7 lang", min_value=0, max_value=1, format="%.2f"),
    "f8_review_anomaly":       st.column_config.ProgressColumn("F8 anom", min_value=0, max_value=1, format="%.2f"),
    "nights_booked":           st.column_config.NumberColumn("Nights", format="%d"),
    "portfolio_size":          st.column_config.NumberColumn("Portfolio", format="%d"),
    "estimated_revenue_l365d": st.column_config.NumberColumn("Revenue £", format="£%,.0f"),
    "neighbourhood_cleansed":  st.column_config.TextColumn("Borough"),
    "host_name":               st.column_config.TextColumn("Host"),
    "priority_tier":           st.column_config.TextColumn("Tier"),
}
DISPLAY_COLS = [
    "host_name", "neighbourhood_cleansed", "confidence_score", "priority_tier",
    "nights_booked", "portfolio_size", "operator_total_listings",
    "f7_commercial_language", "f8_review_anomaly", "estimated_revenue_l365d",
]
OP_CFG = {
    "max_confidence":           st.column_config.ProgressColumn("Max score", min_value=0, max_value=1, format="%.2f"),
    "avg_confidence":           st.column_config.ProgressColumn("Avg score", min_value=0, max_value=1, format="%.2f"),
    "total_listings":           st.column_config.NumberColumn("Listings", format="%d"),
    "listings_above_90_nights": st.column_config.NumberColumn(">90 nights", format="%d"),
    "total_estimated_revenue":  st.column_config.NumberColumn("Est. revenue £", format="£%,.0f"),
    "match_score":              st.column_config.ProgressColumn("CH match %", min_value=0, max_value=100, format="%.0f"),
    "operator_name":            st.column_config.TextColumn("Operator"),
    "company_name":             st.column_config.TextColumn("Company (CH)"),
    "company_status":           st.column_config.TextColumn("Status"),
}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏚️ Ghost Hotel Detector")
    st.caption("London STR Enforcement Intelligence")
    st.divider()

    if core.nim_ready():
        st.success("Nemotron VL: online", icon="🟢")
    else:
        st.warning("Nemotron VL: loading…", icon="🟡")
    st.caption(f"Model · `{core.MODEL}`")

    st.divider()
    st.markdown("**Dashboard map filter**")
    boroughs = sorted(listings["neighbourhood_cleansed"].dropna().unique())
    selected_boroughs = st.multiselect("Borough", options=boroughs, default=[], placeholder="All boroughs")
    min_score = st.slider("Min confidence", 0.0, 0.97, 0.75, 0.01)

    st.divider()
    st.markdown("**GPU benchmark**")
    b1, b2 = st.columns(2)
    b1.metric("CPU", f"{bench['cpu_seconds']:.1f}s")
    b2.metric("GPU", f"{bench['gpu_seconds']:.1f}s", f"{bench['speedup']:.1f}x")
    st.caption(f"Snapshot · {core.SNAPSHOT_DATE}")

# ── Header ──────────────────────────────────────────────────────────────────
ui.hero(
    "Ghost Hotel Detector",
    "AI-driven enforcement intelligence for illegal London short-term rentals · Deregulation Act 2015 s.44",
)
ui.kpi_cards([
    {"label": "Listings scored",      "value": f"{len(listings):,}"},
    {"label": "Priority cases",       "value": f"{(listings['priority_tier']=='Priority').sum():,}", "sub": "score > 0.90", "color": "#e63946"},
    {"label": "Review cases",         "value": f"{(listings['priority_tier']=='Review').sum():,}",   "sub": "0.75–0.90",  "color": "#f4a261"},
    {"label": "Commercial operators", "value": f"{operators['canonical_operator_id'].nunique():,}"},
    {"label": "Companies House",      "value": f"{operators['company_number'].notna().sum():,}", "sub": "matched"},
])
st.write("")

tab_dash, tab_agent = st.tabs(["📊  Dashboard", "🤖  Agent Console"])

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    view = listings.copy()
    if selected_boroughs:
        view = view[view["neighbourhood_cleansed"].isin(selected_boroughs)]
    view = view[view["confidence_score"] >= min_score]

    left, right = st.columns([3, 2])

    with left:
        ui.section("Risk map", f"{len(view):,} listings match the current filter")
        with ui.card():
            map_data = view.dropna(subset=["latitude", "longitude"]).copy()
            if len(map_data) > 3000:
                map_data = map_data.nlargest(3000, "confidence_score")
            m = folium.Map(location=[51.505, -0.09], zoom_start=10, tiles="CartoDB positron")
            for _, row in map_data.iterrows():
                tier = str(row.get("priority_tier", "Low"))
                colour = TIER_COLOURS.get(tier, "#888")
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=4 if tier == "Priority" else 3,
                    color=colour, fill=True, fill_color=colour, fill_opacity=0.75,
                    tooltip=f"{row.get('host_name','')} — {row['confidence_score']:.2f}",
                ).add_to(m)
            st_folium(m, width="100%", height=460, returned_objects=[])

    with right:
        ui.section("Priority + Review by borough")
        with ui.card():
            bc = (view[view["priority_tier"].isin(["Priority", "Review"])]
                  .groupby("neighbourhood_cleansed").size().reset_index(name="count"))
            if not bc.empty:
                fig = px.bar(bc.sort_values("count").tail(15), x="count", y="neighbourhood_cleansed",
                             orientation="h", color="count", color_continuous_scale="Reds",
                             labels={"count": "Cases", "neighbourhood_cleansed": ""})
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No priority/review cases in the current filter.")

    ui.section("Top operators", "Highest-confidence commercial operators, with Companies House matches")
    with ui.card():
        op_cols = ["operator_name", "total_listings", "listings_above_90_nights",
                   "avg_confidence", "max_confidence", "total_estimated_revenue",
                   "company_name", "company_status", "match_score"]
        ops_show = (operators.sort_values("max_confidence", ascending=False)
                    [[c for c in op_cols if c in operators.columns]].head(100))
        st.dataframe(ops_show, column_config=OP_CFG, use_container_width=True, height=360)

# ════════════════════════════════════════════════════════════════════════════
# AGENT CONSOLE
# ════════════════════════════════════════════════════════════════════════════

def _stats_str(df: pd.DataFrame) -> str:
    bb = df.groupby("neighbourhood_cleansed").size().nlargest(5).to_string()
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    top5 = df[cols].head(5).to_string(index=False)
    return (f"Total matched: {len(df)}\n"
            f"Priority tier: {(df['priority_tier']=='Priority').sum()}\n"
            f"Avg confidence: {df['confidence_score'].mean():.2f}\n"
            f"Avg nights booked: {df['nights_booked'].mean():.0f}\n"
            f"Top boroughs:\n{bb}\nTop 5 rows:\n{top5}")


def render_table(res, live, query):
    df = res["df"]
    if df is None or df.empty:
        st.warning("No listings matched. Try broadening the request.")
        return ""
    s = res.get("stats", {})
    ui.kpi_cards([
        {"label": "Matched",   "value": f"{s.get('matched', len(df)):,}"},
        {"label": "Priority",  "value": f"{s.get('priority', 0):,}", "color": "#e63946"},
        {"label": "Avg score", "value": f"{s.get('avg_score', df['confidence_score'].mean()):.2f}"},
        {"label": "Avg nights","value": f"{s.get('avg_nights', df['nights_booked'].mean()):.0f}"},
    ])
    if res.get("expr"):
        with st.expander("pandas expression the agent ran"):
            st.code(res["expr"], language="python")
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    st.dataframe(df[cols].head(500), column_config=COL_CFG, use_container_width=True, height=380)
    st.download_button("⬇ Download CSV", df[cols].to_csv(index=False).encode(),
                       "results.csv", "text/csv", key=f"dl_{abs(hash(query))%99999}")

    # Intelligence briefing
    st.markdown("**Intelligence briefing**")
    if live:
        box, text = st.empty(), ""
        try:
            for kind, chunk in core.stream_nim(agent.build_insight_prompt(query, _stats_str(df)), temperature=0.3):
                if kind == "content":
                    text += chunk
                    box.markdown(text)
        except Exception as e:
            text = f"_Briefing unavailable: {e}_"
            box.markdown(text)
        return text
    else:
        st.markdown(res.get("briefing", "") or "_(briefing not stored)_")
        return res.get("briefing", "")


def render_property(res):
    lid = res["listing_id"]
    st.markdown(f"[View on Airbnb ↗](https://www.airbnb.com/rooms/{lid})")
    if res.get("in_snapshot") and res.get("row"):
        row = res["row"]
        tier = str(row.get("priority_tier", "Low"))
        st.markdown(f"{ui.tier_badge(tier)} &nbsp; **score {float(row.get('confidence_score',0)):.2f}**",
                    unsafe_allow_html=True)
        ui.kpi_cards([
            {"label": "Confidence", "value": f"{float(row.get('confidence_score',0)):.2f}"},
            {"label": "Nights",     "value": f"{float(row.get('nights_booked',0)):.0f}", "sub": "cap 90"},
            {"label": "Portfolio",  "value": f"{int(row.get('portfolio_size',1) or 1)}"},
            {"label": "Borough",    "value": str(row.get("neighbourhood_cleansed", ""))},
        ])
        det = {k: row.get(k) for k in ["property_type", "room_type", "price", "minimum_nights",
                                        "number_of_reviews", "reviews_per_month", "host_is_superhost",
                                        "operator_name", "license"] if row.get(k) not in (None, "")}
        st.json(det, expanded=False)
        if row.get("description"):
            with st.expander("Listing description"):
                st.write(str(row["description"])[:1000])
    else:
        st.warning(f"⚠️ Listing **{lid}** is **not in our {core.SNAPSHOT_DATE} snapshot** — likely created "
                   "after our data cut-off. Pulled live + scored with the ML classifier instead.")
        scrape = res.get("scrape") or {}
        feats = scrape.get("features", {})
        if scrape.get("ok") and feats:
            st.success(f"Live data pulled — {len(feats)} fields read from the listing page.")
            st.json(feats, expanded=False)
        else:
            st.info(f"Couldn't auto-read live data ({scrape.get('reason','unknown')}). "
                    "Airbnb blocks server-side scraping from this host.")
        ml = res.get("ml")
        if ml:
            tcol = TIER_COLOURS.get(ml["tier"], "#888")
            st.markdown(f"<h3 style='color:{tcol}'>🤖 ML risk: {ml['tier']} — {ml['prob']*100:.1f}% ghost-hotel likelihood</h3>",
                        unsafe_allow_html=True)
            st.progress(min(max(ml["prob"], 0.0), 1.0))
            st.markdown("**Top factors** (occlusion analysis):")
            for f in ml["top_factors"]:
                arrow = "🔺" if f["contribution"] > 0 else "🔻"
                verb = "raises" if f["contribution"] > 0 else "lowers"
                st.markdown(f"- {arrow} **{f['feature']}** = {f['value']} — {verb} risk by {abs(f['contribution'])*100:.1f} pts")
            st.caption("ℹ️ " + ml.get("note", ""))
        else:
            st.caption("ML model not available — run `python3 pipeline/train_model.py`.")


def render_operator(res):
    df = res.get("df")
    if df is None or df.empty:
        st.warning("No matching operators.")
        return
    st.dataframe(df, column_config=OP_CFG, use_container_width=True, height=360)


def render_enforcement(res):
    st.success(f"Enforcement package drafted for **{res.get('host','?')}** — {res.get('borough','')}")
    st.markdown(res.get("text", ""))
    if res.get("saved"):
        st.caption(f"Saved → {res['saved']}")


def render_body(res, live=False, query=""):
    """Render a tool result by kind. Returns a 'briefing' string to persist (tables only)."""
    kind = res.get("kind")
    if kind == "table":
        return render_table(res, live, query)
    if kind == "property":
        render_property(res); return ""
    if kind == "operator":
        render_operator(res); return ""
    if kind == "enforcement":
        render_enforcement(res); return ""
    st.error(res.get("message", "Something went wrong."))
    return ""


with tab_agent:
    ui.section("Agent console", "Ask in plain English — the agent picks the right tool and shows its work.")

    if "agent_chat" not in st.session_state:
        st.session_state.agent_chat = []

    clicked = ui.chips(agent.QUICK_ACTIONS, key="qa", cols=3)

    # Replay history (static, no streaming / no status cards)
    for turn in st.session_state.agent_chat:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            d = turn["decision"]
            st.caption(f"🛠️ {d['tool']} · {d.get('rationale','')}")
            render_body(turn["result"], live=False, query=turn["q"])

    # New message
    submitted_q = clicked
    with st.form("agent_form", clear_on_submit=True):
        c1, c2 = st.columns([6, 1])
        typed = c1.text_input("msg", placeholder="e.g. who are the worst offenders in Westminster?",
                              label_visibility="collapsed")
        sent = c2.form_submit_button("Send", type="primary", use_container_width=True)
    if sent and typed.strip():
        submitted_q = typed.strip()

    if submitted_q:
        with st.chat_message("user"):
            st.markdown(submitted_q)
        with st.chat_message("assistant"):
            with st.status("Thinking…", expanded=True) as status:
                st.write("🧭 Routing your request…")
                decision = agent.route(submitted_q)
                st.write(f"→ tool **{decision['tool']}** · {decision.get('rationale','')}")
                st.write("⚙️ Running tool…")
                result = agent.run_tool(decision["tool"], decision["args"])
                status.update(label=f"Done · {decision['tool']}", state="complete", expanded=False)
            briefing = render_body(result, live=True, query=submitted_q)

        # Persist (store briefing text so replay is static)
        if isinstance(result, dict) and result.get("kind") == "table":
            result = {**result, "briefing": briefing}
        st.session_state.agent_chat.append({"q": submitted_q, "decision": decision, "result": result})
