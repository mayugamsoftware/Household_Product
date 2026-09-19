import streamlit as st

from utils import (
    PRODUCT_CONFIG,
    calculate_product_age,
    condition_color,
    fetch_common_values,
    fetch_product_current,
    load_models,
    load_history,
    predict_product,
)
from style import inject_css, hero, metric_card

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

hero(
    "📡 Predictive Maintenance Dashboard",
    "Live IoT sensor monitoring & ML-powered condition / RUL prediction — powered by Blynk",
)

models, model_issues = load_models()

if model_issues:
    st.error(
        "⚠️ Model files could not be loaded. Place `scaler.pkl`, `label_encoder.pkl`, "
        "`condition.pkl` and `life.pkl` in the app folder.\n\n"
        f"Details: {model_issues}"
    )

st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.info("Use the pages in the sidebar above to open each product's detail view.")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption(
    "This dashboard pulls live voltage, temperature and vibration data from Blynk, "
    "combines it with each product's load current, and runs it through trained ML "
    "models to estimate current condition and Remaining Useful Life (RUL)."
)

refresh = st.button("🔄 Fetch Live Data & Run All Predictions", type="primary", use_container_width=False)

st.markdown('<div class="section-title">🌡️ Common Sensor Readings</div>', unsafe_allow_html=True)

common_placeholder = st.container()

if "common_values" not in st.session_state:
    st.session_state.common_values = None
if "product_results" not in st.session_state:
    st.session_state.product_results = {}

if refresh:
    with st.spinner("Fetching sensor data from Blynk..."):
        common = fetch_common_values()
        st.session_state.common_values = common

    for name in PRODUCT_CONFIG:
        current, err = fetch_product_current(name)
        if current is None or common["voltage"] is None:
            st.session_state.product_results[name] = None
            continue
        if models is None:
            st.session_state.product_results[name] = None
            continue
        result, pred_err = predict_product(
            models,
            name,
            common["voltage"],
            current,
            common["temperature"],
            common["vibration"],
        )
        st.session_state.product_results[name] = result

common = st.session_state.common_values

with common_placeholder:
    c1, c2, c3 = st.columns(3)
    if common:
        with c1:
            metric_card("Voltage (V0)", f"{common['voltage']:.2f}" if common["voltage"] is not None else "—", "V")
        with c2:
            metric_card("Temperature (V4)", f"{common['temperature']:.2f}" if common["temperature"] is not None else "—", "°C")
        with c3:
            metric_card("Vibration (V5)", f"{common['vibration']:.4f}" if common["vibration"] is not None else "—", "")

        for key, err in common["errors"].items():
            if err:
                st.warning(f"**{key}**: {err}")
    else:
        with c1:
            metric_card("Voltage (V0)", "—", "V")
        with c2:
            metric_card("Temperature (V4)", "—", "°C")
        with c3:
            metric_card("Vibration (V5)", "—", "")
        st.caption("Click **Fetch Live Data & Run All Predictions** to pull the latest readings.")

st.markdown('<div class="section-title">📦 Product Overview</div>', unsafe_allow_html=True)

cols = st.columns(3)
for idx, (name, info) in enumerate(PRODUCT_CONFIG.items()):
    result = st.session_state.product_results.get(name)
    age = calculate_product_age(info["manufacturing_date"], info["installation_date"])
    history_count = len(load_history(name))

    with cols[idx]:
        if result:
            color = condition_color(result["condition"])
            condition_html = f'<span class="badge" style="background:{color};">{result["condition"]}</span>'
            rul_html = f"{result['rul']:.1f} months"
            current_html = f"{result['current']:.2f} A"
        else:
            condition_html = '<span class="badge" style="background:#475569;">No data yet</span>'
            rul_html = "—"
            current_html = "—"

        st.markdown(
            f"""
            <div class="product-card">
                <h3>{info['icon']} {name}</h3>
                {condition_html}
                <div style="margin-top:0.8rem; line-height:1.9;">
                    <div class="subtle">Load Current: <b style="color:#e2e8f0;">{current_html}</b></div>
                    <div class="subtle">Estimated RUL: <b style="color:#e2e8f0;">{rul_html}</b></div>
                    <div class="subtle">Installed: {info['installation_date']} ({age['installation_age_years']:.2f} yrs)</div>
                    <div class="subtle">Manufactured: {info['manufacturing_date']} ({age['manufacturing_age_years']:.2f} yrs)</div>
                    <div class="subtle">History logged: {history_count} entries</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    "Open a product page from the sidebar to fetch that product's data individually, "
    "run a prediction, view its history log, and download it as CSV."
)
