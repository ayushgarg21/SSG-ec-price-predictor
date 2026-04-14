"""Streamlit frontend for EC Price Prediction."""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="EC Price Predictor", page_icon="🏠", layout="centered")

st.title("EC Price Predictor")
st.caption("Predict Executive Condominium resale prices at MOP (5yr) and privatisation (10yr)")

# --- Sidebar: API health ---
try:
    health = httpx.get(f"{API_BASE}/health", timeout=5).json()
    status_color = "🟢" if health["model_loaded"] else "🔴"
    st.sidebar.markdown(f"**API Status:** {status_color} {health['status']}")
    st.sidebar.markdown(f"**Model loaded:** {health['model_loaded']}")
    if health.get("active_run_id"):
        st.sidebar.markdown(f"**Run ID:** `{health['active_run_id']}`")
except Exception:
    st.sidebar.error("API unreachable. Start with `make serve`.")

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "Predicts EC resale prices using an **appreciation model** trained on "
    "10,140 URA resale transactions. Returns point estimates with 80% prediction intervals."
)

# --- Input form ---
DISTRICTS: dict[str, int] = {
    "17 - Changi, Loyang": 17,
    "18 - Pasir Ris, Tampines": 18,
    "19 - Punggol, Sengkang": 19,
    "20 - Ang Mo Kio, Bishan": 20,
    "22 - Boon Lay, Jurong": 22,
    "23 - Bukit Batok, Bukit Panjang": 23,
    "24 - Choa Chu Kang": 24,
    "25 - Woodlands, Admiralty": 25,
    "27 - Sembawang, Yishun": 27,
    "28 - Seletar, Yio Chu Kang": 28,
}

EC_PROJECTS = [
    "", "PIERMONT GRAND", "PARC GREENWICH", "OLA", "RIVERCOVE RESIDENCES",
    "COPEN GRAND", "NORTH GAIA", "TENET", "PARC CENTRAL RESIDENCES",
    "PROVENCE RESIDENCE", "PARC CANBERRA", "SOL ACRES", "BELLEWATERS",
    "THE TOPIARY", "RIVELLE TAMPINES", "COASTAL CABANA", "OTTO PLACE",
    "NOVO PLACE", "LUMINA GRAND", "ALTURA", "AURELLE OF TAMPINES",
]

tab_milestone, tab_custom = st.tabs(["Milestone Comparison", "Custom Prediction"])

with tab_milestone:
    with st.form("milestone_form"):
        col1, col2 = st.columns(2)
        with col1:
            district_label = st.selectbox("Postal District", list(DISTRICTS.keys()), index=2)
            area = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=300.0, value=95.0)
            project = st.selectbox("EC Project (optional)", EC_PROJECTS, index=0)
        with col2:
            floor = st.number_input("Floor Level", min_value=1, max_value=50, value=8)
            lease_year = st.number_input("Lease Commencement Year", min_value=1995, max_value=2030, value=2020)
            segment = st.selectbox("Market Segment", ["OCR", "RCR", "CCR"], index=0)
        submitted = st.form_submit_button("Predict at MOP & Privatisation", use_container_width=True)

    if submitted:
        payload = {
            "district": DISTRICTS[district_label],
            "area_sqm": area,
            "floor": floor,
            "lease_commence_year": lease_year,
            "market_segment": segment,
        }
        if project:
            payload["project"] = project

        with st.spinner("Predicting..."):
            try:
                resp = httpx.post(f"{API_BASE}/predict/milestones", json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Price at MOP (5yr)", f"S${data['mop_5yr_price']:,.0f}")
                    pi = data.get("mop_5yr_interval", {})
                    if pi:
                        st.caption(f"80% PI: S${pi['lower_bound']:,.0f} — S${pi['upper_bound']:,.0f}")

                with col_b:
                    st.metric("Price at Privatisation (10yr)", f"S${data['privatised_10yr_price']:,.0f}")
                    pi = data.get("privatised_10yr_interval", {})
                    if pi:
                        st.caption(f"80% PI: S${pi['lower_bound']:,.0f} — S${pi['upper_bound']:,.0f}")

                appreciation = data["price_appreciation"]
                pct = data.get("appreciation_pct", 0)
                st.metric(
                    "5yr → 10yr Appreciation",
                    f"S${appreciation:,.0f}",
                    delta=f"{pct:+.1f}%",
                )

            except httpx.HTTPStatusError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_custom:
    with st.form("custom_form"):
        col1, col2 = st.columns(2)
        with col1:
            c_district_label = st.selectbox("Postal District", list(DISTRICTS.keys()), index=2, key="c_dist")
            c_area = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=300.0, value=95.0, key="c_area")
            c_years = st.number_input("Years from Launch", min_value=0, max_value=99, value=5, key="c_years")
            c_project = st.selectbox("EC Project (optional)", EC_PROJECTS, index=0, key="c_proj")
        with col2:
            c_floor = st.number_input("Floor Level", min_value=1, max_value=50, value=8, key="c_floor")
            c_lease_year = st.number_input("Lease Commencement Year", min_value=1995, max_value=2030, value=2020, key="c_lease")
            c_sale_type = st.selectbox("Sale Type", ["Resale", "Sub Sale", "New Sale"], key="c_sale")
            c_segment = st.selectbox("Market Segment", ["OCR", "RCR", "CCR"], index=0, key="c_seg")
        c_submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if c_submitted:
        payload = {
            "district": DISTRICTS[c_district_label],
            "area_sqm": c_area,
            "floor": c_floor,
            "lease_commence_year": c_lease_year,
            "years_from_launch": c_years,
            "sale_type": c_sale_type,
            "market_segment": c_segment,
        }
        if c_project:
            payload["project"] = c_project

        with st.spinner("Predicting..."):
            try:
                resp = httpx.post(f"{API_BASE}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                st.metric("Predicted Price", f"S${data['predicted_price']:,.0f}")

                pi = data.get("prediction_interval", {})
                if pi:
                    st.caption(f"80% Prediction Interval: S${pi['lower_bound']:,.0f} — S${pi['upper_bound']:,.0f}")

                # SHAP explanation
                explanation = data.get("explanation")
                if explanation and explanation.get("feature_contributions"):
                    st.subheader("Feature Contributions (SHAP)")
                    st.caption(f"Base appreciation ratio: {explanation['base_value']:.3f}")

                    contribs = explanation["feature_contributions"]
                    for feature, value in contribs.items():
                        if abs(value) > 0.001:
                            direction = "+" if value > 0 else ""
                            st.markdown(
                                f"**{feature}**: `{direction}{value:.4f}` "
                                f"{'🔺' if value > 0 else '🔻'}"
                            )

            except httpx.HTTPStatusError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
