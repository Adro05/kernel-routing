"""
Streamlit internal operations console.

Run with:
    streamlit run app/ui.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.predictor import RoutingPredictor
from src.ops_analytics import build_ops_summary

st.set_page_config(page_title="Kestrel Routing Console", layout="wide", page_icon="🧭")


@st.cache_resource
def get_predictor():
    return RoutingPredictor.instance()


@st.cache_data
def get_ops_summary():
    summary = build_ops_summary()
    # DataFrames aren't hashable-friendly across cache boundaries by default,
    # but st.cache_data handles pandas objects fine via pickling.
    return summary


@st.cache_data
def get_reports():
    reports = {}
    for name in ["model_comparison.csv", "per_team_metrics.csv", "error_analysis.csv", "confidence_bands.json"]:
        p = Path("reports") / name
        if not p.exists():
            continue
        if name.endswith(".csv"):
            reports[name] = pd.read_csv(p)
        else:
            reports[name] = json.loads(p.read_text())
    return reports


PAGES = ["Route Request", "Operations", "Model Performance", "About / Model Card"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Kestrel Home — Service Routing Intelligence")

BAND_COLOR = {"high": "🟢", "medium": "🟡", "low": "🔴"}


if page == "Route Request":
    st.title("Route a Service Request")
    st.caption("Predicts the operational team for a new inbound request, with confidence, "
               "explanation and similar historical evidence.")

    col1, col2 = st.columns([2, 1])
    with col1:
        request_text = st.text_area(
            "Request text", height=120,
            placeholder="e.g. My water purifier is leaking from the bottom",
        )
    with col2:
        product_family = st.selectbox(
            "Product family",
            ["Water Purifier", "Air Fryer", "Mixer Grinder", "Induction Cooktop",
             "Room Heater", "Ceiling Fan", "Robot Vacuum"],
        )
        warranty_status = st.selectbox("Warranty status", ["in_warranty", "shield", "out_of_warranty"])
        channel = st.selectbox("Channel", ["chat", "whatsapp", "ivr", "email"])

    if st.button("Route Request", type="primary"):
        if not request_text.strip():
            st.error("Please enter request text.")
        else:
            predictor = get_predictor()
            result = predictor.predict({
                "request_text": request_text,
                "product_family": product_family,
                "warranty_status": warranty_status,
                "channel": channel,
            })

            band = result["confidence_band"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted team", result["predicted_team"])
            c2.metric("Confidence", f"{result['confidence']:.0%}")
            c3.metric("Confidence band", f"{BAND_COLOR.get(band,'')} {band}")

            if result["review_required"]:
                st.warning("⚠️ Human review recommended — confidence is below the 'high' band.")
            else:
                st.success("No review flag — this prediction falls in the high-confidence band.")

            if result.get("runner_up"):
                ru = result["runner_up"]
                st.caption(f"Runner-up team: **{ru['team']}** ({ru['confidence']:.0%})")

            st.subheader("Why this team?")
            for r in result["reasons"]:
                st.markdown(f"- {r}")

            st.subheader("Similar historical requests")
            if result["similar_cases"]:
                st.dataframe(pd.DataFrame(result["similar_cases"]), hide_index=True, use_container_width=True)
            else:
                st.info("No similar historical cases found.")


elif page == "Operations":
    st.title("Operations Analytics")
    st.caption("Derived from resolution_log.csv — separate from the routing model; "
               "never used as a prediction feature.")

    summary = get_ops_summary()
    friction = summary["friction"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests analyzed", f"{friction['total_requests']:,}")
    c2.metric("Legacy-bot misroute rate", f"{friction['misroute_rate']:.1%}")
    c3.metric("Total transfers", f"{friction['total_transfers']:,}")
    c4.metric("Est. friction cost (15 mo.)", f"₹{friction['estimated_total_friction_cost_inr']:,}")

    st.markdown(
        f"At ₹305/transfer and ₹260/extra contact (ops-policy.pdf §4), the legacy bot's own "
        f"misrouting cost an estimated **₹{friction['estimated_total_friction_cost_inr']:,}** over the "
        f"15 months of history in this dataset — well above its ₹3.2 lakh/year licence fee."
    )

    st.subheader("Workload by team (requests actually closed)")
    st.bar_chart(summary["workload_by_team"].set_index("final_team"))

    st.subheader("Where the legacy bot's initial routing needed a transfer")
    st.caption("Transfer rate = share of requests initially sent to this team that were later moved elsewhere.")
    st.dataframe(summary["transfers_by_first_team"], hide_index=True, use_container_width=True)

    st.subheader("Monthly request volume by team")
    mv = summary["monthly_volume"].pivot(index="month", columns="final_team", values="count").fillna(0)
    st.line_chart(mv)

    st.subheader("Resolution time (hours)")
    rt = summary["resolution_times"]["resolution_hours"].dropna()
    st.write(f"Median: **{rt.median():.1f}h**, 90th percentile: **{rt.quantile(0.9):.1f}h**")
    st.caption("Legacy-Zoho-sourced resolution timestamps were stored in UTC and corrected by "
               "+5:30 for this analysis (ops-policy.pdf §9); this correction is not applied "
               "anywhere in the routing model.")


elif page == "Model Performance":
    st.title("Model Performance")
    reports = get_reports()

    bands = reports.get("confidence_bands.json", {})
    c1, c2 = st.columns(2)
    c1.metric("Temporal holdout accuracy", f"{bands.get('overall_accuracy', 0):.1%}")
    c2.metric("Temporal holdout macro F1", f"{bands.get('overall_macro_f1', 0):.3f}")

    st.subheader("Experiment comparison")
    if "model_comparison.csv" in reports:
        st.dataframe(reports["model_comparison.csv"], hide_index=True, use_container_width=True)

    st.subheader("Per-team metrics (temporal holdout)")
    if "per_team_metrics.csv" in reports:
        st.dataframe(reports["per_team_metrics.csv"], hide_index=True, use_container_width=True)

    st.subheader("Confusion matrix")
    cm_path = Path("reports/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path))

    st.subheader("Error categories")
    if "error_analysis.csv" in reports:
        err = reports["error_analysis.csv"]
        st.bar_chart(err["error_category"].value_counts())
        with st.expander("Browse individual errors"):
            st.dataframe(err, hide_index=True, use_container_width=True)


elif page == "About / Model Card":
    st.title("About this system")
    model_card_path = Path("MODEL_CARD.md")
    if model_card_path.exists():
        st.markdown(model_card_path.read_text())
    else:
        st.info("MODEL_CARD.md not found.")
