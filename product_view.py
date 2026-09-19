"""Reusable renderer for a single product's detail page."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    PRODUCT_CONFIG,
    append_history,
    calculate_product_age,
    clear_history,
    condition_color,
    fetch_common_values,
    fetch_product_current,
    load_history,
    load_models,
    predict_product,
)
from style import inject_css, hero, metric_card, condition_badge


def _gauge(value, title, max_value, color, suffix=""):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 30, "color": "#f8fafc"}},
            title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": "#475569"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.12)",
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )
    return fig


def render_product_page(product_name: str):
    info = PRODUCT_CONFIG[product_name]

    st.set_page_config(
        page_title=f"{product_name} · Dashboard",
        page_icon=info["icon"],
        layout="wide",
    )
    inject_css()

    hero(
        f"{info['icon']} {product_name}",
        f"Manufactured {info['manufacturing_date']}  ·  Installed {info['installation_date']}",
    )

    models, model_issues = load_models()
    if model_issues:
        st.error(
            "⚠️ Model files could not be loaded. Place `scaler.pkl`, `label_encoder.pkl`, "
            f"`condition.pkl` and `life.pkl` in the app folder.\n\nDetails: {model_issues}"
        )

    age = calculate_product_age(info["manufacturing_date"], info["installation_date"])

    age_col1, age_col2, age_col3, age_col4 = st.columns(4)
    with age_col1:
        metric_card("Manufacturing Age", f"{age['manufacturing_age_years']:.2f}", "yrs")
    with age_col2:
        metric_card("Installation Age", f"{age['installation_age_years']:.2f}", "yrs")
    with age_col3:
        metric_card("Manufacturing Date", info["manufacturing_date"], "")
    with age_col4:
        metric_card("Installation Date", info["installation_date"], "")

    st.markdown('<div class="section-title">🔴 Live Fetch & Predict</div>', unsafe_allow_html=True)

    fetch_clicked = st.button(
        f"🔄 Fetch Live Data & Predict — {product_name}", type="primary", key=f"fetch_{product_name}"
    )

    state_key = f"result_{product_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    if fetch_clicked:
        with st.spinner("Fetching sensor values from Blynk..."):
            common = fetch_common_values()
            current, current_err = fetch_product_current(product_name)

        errors = [e for e in common["errors"].values() if e]
        if current_err:
            errors.append(f"Load current: {current_err}")

        if common["voltage"] is None or common["temperature"] is None or common["vibration"] is None or current is None:
            st.error("❌ Could not fetch all required sensor values.")
            for e in errors:
                st.caption(f"- {e}")
        elif models is None:
            st.error("❌ Cannot run prediction — ML models are not loaded.")
        else:
            result, pred_err = predict_product(
                models, product_name, common["voltage"], current, common["temperature"], common["vibration"]
            )
            if pred_err:
                st.error(f"❌ Prediction failed: {pred_err}")
            else:
                st.session_state[state_key] = result
                append_history(result)
                st.success("✅ Prediction complete and logged to history.")

    result = st.session_state[state_key]

    if result:
        st.markdown('<div class="section-title">📥 Model Input (Live Sensor Values)</div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            metric_card("Voltage", f"{result['voltage']:.2f}", "V")
        with i2:
            metric_card("Load Current", f"{result['current']:.2f}", "A")
        with i3:
            metric_card("Temperature", f"{result['temperature']:.2f}", "°C")
        with i4:
            metric_card("Vibration", f"{result['vibration']:.4f}", "")

        st.markdown('<div class="section-title">🔮 Prediction Result</div>', unsafe_allow_html=True)

        color = condition_color(result["condition"])
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("**Quality / Condition**")
            condition_badge(result["condition"], color)
            st.plotly_chart(
                _gauge(result["rul"], "Remaining Useful Life", max(result["rul"] * 1.4, 24), color, " mo"),
                use_container_width=True,
            )
        with res_col2:
            st.plotly_chart(
                _gauge(result["voltage"], "Voltage", max(result["voltage"] * 1.5, 260), "#4f46e5", " V"),
                use_container_width=True,
            )
            st.plotly_chart(
                _gauge(result["temperature"], "Temperature", max(result["temperature"] * 1.5, 100), "#db2777", " °C"),
                use_container_width=True,
            )
    else:
        st.info("No prediction yet — click **Fetch Live Data & Predict** above to get started.")

    # ========================================================
    # HISTORY
    # ========================================================
    st.markdown('<div class="section-title">🗂️ Prediction History</div>', unsafe_allow_html=True)

    history_df = load_history(product_name)

    if history_df.empty:
        st.caption("No history recorded yet for this product. Run a prediction to start logging.")
    else:
        display_df = history_df.copy()
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True, height=320)

        h1, h2, h3 = st.columns([1, 1, 2])
        with h1:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download History CSV",
                data=csv_bytes,
                file_name=f"{product_name.replace(' ', '_').lower()}_history.csv",
                mime="text/csv",
                key=f"download_{product_name}",
                use_container_width=True,
            )
        with h2:
            if st.button("🗑️ Clear History", key=f"clear_{product_name}", use_container_width=True):
                clear_history(product_name)
                st.rerun()
        with h3:
            st.caption(f"{len(history_df)} entries logged · stored at `data/`")

        if len(history_df) > 1:
            st.markdown('<div class="section-title">📈 Trend</div>', unsafe_allow_html=True)
            trend_df = history_df.copy()
            trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
            t1, t2 = st.columns(2)
            with t1:
                st.line_chart(trend_df.set_index("timestamp")[["rul_months"]])
            with t2:
                st.line_chart(trend_df.set_index("timestamp")[["voltage", "current", "temperature"]])
