"""ui.py — Presentation layer for the "Ghost Hotel Detector".

A pure presentation / theming module for the Streamlit dashboard used by a
London council short-let enforcement team. The goal is a clean, modern, LIGHT
theme: rounded cards, comfortable typography and subtle motion — professional
and trustworthy rather than flashy.

This module contains NO business logic. It only emits styled HTML/Streamlit
widgets. It is imported as a flat sibling module (``import ui``).

Public API:
    inject_theme()                      -> None
    hero(title, subtitle)               -> None
    kpi_cards(items)                    -> None
    section(title, sub="")              -> None
    chips(labels, key, cols=4)          -> str | None
    badge(text, color="#4f46e5")        -> str
    tier_badge(tier)                    -> str
    card(title="", subtitle="")         -> context manager (bordered container)
"""

from __future__ import annotations

import contextlib

import streamlit as st

# --------------------------------------------------------------------------- #
# Palette constants (kept in sync with the CSS variables in inject_theme).
# --------------------------------------------------------------------------- #
ACCENT = "#4f46e5"   # indigo — primary accent
DANGER = "#e63946"   # red    — danger / priority

# Triage tier -> colour. Used by tier_badge().
TIER = {
    "Priority": "#e63946",
    "Review": "#f4a261",
    "Monitor": "#457b9d",
    "Low": "#a8dadc",
}


# --------------------------------------------------------------------------- #
# 1. Global theme
# --------------------------------------------------------------------------- #
def inject_theme() -> None:
    """Emit a single global <style> block establishing the light theme.

    Call this once, early, after ``st.set_page_config(...)``.
    """
    st.markdown(
        """
        <style>
        /* ---- Font ---------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --accent: #4f46e5;   /* indigo  */
            --danger: #e63946;   /* red     */
            --ink:    #1f2430;   /* body text */
            --canvas: #f6f7f9;   /* app background */
            --line:   #eceef2;   /* hairline borders */
        }

        /* Base font + light canvas */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--ink);
        }
        .stApp { background: var(--canvas); }

        /* ---- Hide Streamlit chrome ----------------------------------- */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        [data-testid="stToolbar"] { display: none; }            /* deploy/toolbar */
        .stDeployButton { display: none; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---- Layout -------------------------------------------------- */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        /* ---- Primary buttons ----------------------------------------- */
        button[kind="primary"] {
            background: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.30);
            color: #ffffff;
        }

        /* ---- Secondary buttons styled as pill "chips" ---------------- */
        button[kind="secondary"] {
            background: #ffffff;
            color: var(--ink);
            border: 1px solid #e3e6ec;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        button[kind="secondary"]:hover {
            border-color: var(--accent);
            background: rgba(79, 70, 229, 0.06);
            color: var(--accent);
        }

        /* ---- Bordered containers as cards ---------------------------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
            padding: 18px 20px;
        }

        /* ---- Chat bubbles -------------------------------------------- */
        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 10px 14px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        /* ---- Metrics ------------------------------------------------- */
        [data-testid="stMetricValue"] {
            font-weight: 700;
            font-size: 1.9rem;
        }

        /* ---- Motion -------------------------------------------------- */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .gh-fade { animation: fadeInUp 0.35s ease both; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 2. Hero header
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str) -> None:
    """Render a full-width gradient header card with title + subtitle."""
    st.markdown(
        f"""
        <div class="gh-fade" style="
            background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            border-radius: 16px;
            padding: 22px 26px;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.25);
            margin-bottom: 1.1rem;">
            <div style="font-size: 1.6rem; font-weight: 700; line-height: 1.2;">{title}</div>
            <div style="font-size: 0.95rem; opacity: 0.9; margin-top: 4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 3. KPI cards
# --------------------------------------------------------------------------- #
def kpi_cards(items: list[dict]) -> None:
    """Render a responsive row of KPI cards.

    Each item is a dict:
        {"label": str, "value": str, "sub": str?, "color": str? (hex)}
    """
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color = item.get("color", ACCENT)
        sub = item.get("sub", "")
        sub_html = (
            f'<div style="font-size:0.78rem;color:#8a90a0;margin-top:2px;">{sub}</div>'
            if sub
            else ""
        )
        with col:
            st.markdown(
                f"""
                <div class="gh-fade" style="
                    background:#ffffff;
                    border:1px solid #eceef2;
                    border-radius:14px;
                    box-shadow:0 1px 3px rgba(0,0,0,.06);
                    padding:16px 18px;">
                    <div style="
                        font-size:0.72rem;
                        letter-spacing:0.04em;
                        text-transform:uppercase;
                        color:#8a90a0;
                        font-weight:600;">{item.get("label", "")}</div>
                    <div style="
                        font-size:1.7rem;
                        font-weight:700;
                        color:{color};
                        line-height:1.2;
                        margin-top:4px;">{item.get("value", "")}</div>
                    {sub_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# 4. Section header
# --------------------------------------------------------------------------- #
def section(title: str, sub: str = "") -> None:
    """Render a styled section header with a small indigo accent bar."""
    sub_html = (
        f'<div style="font-size:0.85rem;color:#8a90a0;margin-top:2px;">{sub}</div>'
        if sub
        else ""
    )
    st.markdown(
        f"""
        <div style="margin:0.6rem 0 0.4rem 0;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="
                    display:inline-block;
                    width:6px;height:20px;
                    background:var(--accent, #4f46e5);
                    border-radius:3px;"></span>
                <span style="font-size:1.15rem;font-weight:650;color:#1f2430;">{title}</span>
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 5. Chips (clickable filter pills)
# --------------------------------------------------------------------------- #
def chips(labels: list[str], key: str, cols: int = 4) -> str | None:
    """Render labels as secondary buttons (pill chips) in a grid.

    Returns the label clicked this run, or None. Keys are ``f"{key}_{i}"``.
    """
    clicked: str | None = None
    columns = st.columns(cols)
    for i, label in enumerate(labels):
        col = columns[i % cols]
        with col:
            if st.button(
                label,
                key=f"{key}_{i}",
                use_container_width=True,
                type="secondary",
            ):
                clicked = label
    return clicked


# --------------------------------------------------------------------------- #
# 6. Badge (returns HTML, does not render)
# --------------------------------------------------------------------------- #
def badge(text: str, color: str = "#4f46e5") -> str:
    """Return an HTML pill <span>. Caller renders with unsafe_allow_html."""
    return (
        f'<span style="background:{color}22;color:{color};'
        f"padding:2px 10px;border-radius:999px;"
        f'font-size:0.72rem;font-weight:600;">{text}</span>'
    )


# --------------------------------------------------------------------------- #
# 7. Tier badge
# --------------------------------------------------------------------------- #
def tier_badge(tier: str) -> str:
    """Return a coloured badge for a triage tier (falls back to accent)."""
    return badge(tier, TIER.get(tier, ACCENT))


# --------------------------------------------------------------------------- #
# 8. Card context manager
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def card(title: str = "", subtitle: str = ""):
    """Bordered container card with an optional title/subtitle.

    Usage:
        with ui.card("Results"):
            st.dataframe(...)
    """
    c = st.container(border=True)
    with c:
        if title:
            sub_html = (
                f'<div style="font-size:0.85rem;color:#8a90a0;margin-top:2px;">{subtitle}</div>'
                if subtitle
                else ""
            )
            st.markdown(
                f"""
                <div style="margin-bottom:0.5rem;">
                    <div style="font-size:1.05rem;font-weight:650;color:#1f2430;">{title}</div>
                    {sub_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        yield c
