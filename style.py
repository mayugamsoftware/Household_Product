"""Shared visual styling for the dashboard (custom CSS injection)."""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at top left, #0f172a 0%, #0b1120 45%, #05070d 100%);
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero header */
.hero-header {
    padding: 1.75rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
    box-shadow: 0 12px 32px -12px rgba(124, 58, 237, 0.55);
    margin-bottom: 1.5rem;
}
.hero-header h1 {
    color: white;
    font-weight: 800;
    font-size: 2rem;
    margin: 0 0 0.25rem 0;
}
.hero-header p {
    color: rgba(255,255,255,0.9);
    font-size: 0.98rem;
    margin: 0;
}

/* Metric-style cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(6px);
    height: 100%;
}
.metric-card .label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 700;
}
.metric-card .unit {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    margin-left: 0.25rem;
}

/* Product summary card on the overview page */
.product-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.product-card:hover {
    transform: translateY(-2px);
    border-color: rgba(124, 58, 237, 0.5);
}
.product-card h3 {
    color: #f1f5f9;
    margin: 0 0 0.6rem 0;
    font-size: 1.15rem;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    letter-spacing: 0.02em;
}
.subtle {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Section divider titles */
.section-title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 1.05rem;
    margin: 1.4rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.12);
}
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}<span class="unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def condition_badge(condition: str, color: str):
    st.markdown(
        f'<span class="badge" style="background:{color};">{condition}</span>',
        unsafe_allow_html=True,
    )
