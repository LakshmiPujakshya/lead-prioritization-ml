"""
app.py
------
Streamlit app for the Primriq Lead Prioritization system.

Run locally with:
    streamlit run app.py

Features:
  - Upload a new_incoming_leads.csv style file
  - Loads the trained model (models/best_model.pkl)
  - Cleans data, engineers features, generates predictions
  - Displays leads ranked by priority
  - Lets the user download predictions.csv
"""

import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model
from preprocessing import run_pipeline

st.set_page_config(page_title="Primriq Lead Prioritization", layout="wide")

st.title("📞 Primriq Lead Prioritization")
st.write(
    "Upload a CSV of new incoming leads (same format as `new_incoming_leads.csv`) "
    "to get a model-ranked call list for your counselling team."
)

MODEL_PATH = "models/best_model.pkl"
FEATURE_PATH = "models/feature_columns.json"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, FEATURE_PATH)


uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])

use_sample = st.checkbox("Use the bundled sample (data/new_incoming_leads.csv) instead", value=False)

if uploaded_file is not None or use_sample:
    if use_sample:
        raw = pd.read_csv("data/new_incoming_leads.csv")
        st.info("Using bundled sample file: data/new_incoming_leads.csv")
    else:
        raw = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(raw.head(10), use_container_width=True)

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found at models/best_model.pkl. "
            "Please run `python src/train.py` first."
        )
    else:
        with st.spinner("Cleaning data, engineering features and scoring leads..."):
            model, feat_info = get_model()
            num_feats, cat_feats = feat_info["numeric"], feat_info["categorical"]

            df, cmap, report = run_pipeline(raw, is_historical=False)
            for col in num_feats + cat_feats:
                if col not in df.columns:
                    df[col] = pd.NA

            X = df[num_feats + cat_feats]
            proba = model.predict_proba(X)[:, 1]

            id_col = cmap.get("lead_id", "Lead_ID")
            result = pd.DataFrame({
                "Lead_ID": df[id_col] if id_col in df.columns else range(len(df)),
                "predicted_probability": proba,
            })
            result["priority_score"] = (result["predicted_probability"] * 100).round(1)

            def categorize(p):
                if p >= 0.65:
                    return "High"
                elif p >= 0.40:
                    return "Medium"
                return "Low"

            result["priority_category"] = result["predicted_probability"].apply(categorize)
            result = result.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
            result["recommended_callback_order"] = result.index + 1
            result["predicted_probability"] = result["predicted_probability"].round(4)
            result = result[["Lead_ID", "predicted_probability", "priority_score",
                              "priority_category", "recommended_callback_order"]]

        st.success(f"Scored {len(result)} leads using model: {feat_info.get('best_model', 'N/A')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("High priority", int((result["priority_category"] == "High").sum()))
        col2.metric("Medium priority", int((result["priority_category"] == "Medium").sum()))
        col3.metric("Low priority", int((result["priority_category"] == "Low").sum()))

        st.subheader("Ranked call list")
        category_filter = st.multiselect(
            "Filter by priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
        )
        filtered = result[result["priority_category"].isin(category_filter)]
        st.dataframe(filtered, use_container_width=True, height=500)

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions.csv",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file (or tick the sample-data checkbox) to get started.")

st.divider()
st.caption(
    "Note: model predictions are decision support, not a replacement for "
    "counsellor judgement. See README.md and brief_report.md for limitations."
)
