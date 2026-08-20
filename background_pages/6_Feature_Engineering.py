from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import engineer_data, numeric_columns, require_data, sidebar_filters, download_csv

st.set_page_config(page_title="Feature Engineering & Processing", page_icon="🛠️", layout="wide")
st.title("🛠️ Feature Engineering, Transformation & Data Processing")
st.caption("Prepare different analysis situations: cleaning, robust transformation, ML-ready scaling, or sequential forecasting features.")

raw = require_data()
work = sidebar_filters(raw, "eng")

st.subheader("Data quality diagnostic")
quality = pd.DataFrame({
    "Column": work.columns,
    "Type": [str(work[c].dtype) for c in work.columns],
    "Missing": [int(work[c].isna().sum()) for c in work.columns],
    "Missing %": [float(work[c].isna().mean()) for c in work.columns],
    "Unique": [int(work[c].nunique(dropna=True)) for c in work.columns],
}).sort_values(["Missing %", "Unique"], ascending=[False, False])
a, b, c = st.columns(3)
a.metric("Duplicate rows", int(work.duplicated().sum()))
b.metric("Columns", len(work.columns))
c.metric("Missing cells", int(work.isna().sum().sum()))
st.dataframe(quality.style.format({"Missing %":"{:.1%}"}), use_container_width=True, hide_index=True)

nums = numeric_columns(work)
if nums:
    st.subheader("Numeric transformation preview")
    feature = st.selectbox("Feature", nums)
    q1, q3 = work[feature].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    preview = work[[feature]].copy()
    preview["IQR-capped"] = preview[feature].clip(lower, upper)
    mean, std = preview[feature].mean(), preview[feature].std()
    preview["Z-score"] = (preview[feature] - mean) / std if std and not np.isnan(std) else 0
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(px.histogram(preview, x=feature, nbins=40, marginal="box", title="Original Distribution"), use_container_width=True)
    with p2:
        st.plotly_chart(px.histogram(preview, x="IQR-capped", nbins=40, marginal="box", title="Robust / Capped Preview"), use_container_width=True)

st.subheader("Feature engineering recipes")
recipe = st.selectbox("Processing recipe", ["Clean & robust", "ML-ready", "Sequential forecasting"], index=2)
if recipe == "Clean & robust":
    st.info("Fills numeric missing values with medians and clips extreme numeric values to the 1st–99th percentile range.")
elif recipe == "ML-ready":
    st.info("Clean & robust processing plus standardized Z-score copies of numeric Finder features.")
else:
    st.info("Adds standardized numeric copies plus lag-1, rolling-3 mean, first-difference features, and previous sequential regime/decision fields by Symbol + Timeframe.")

processed = engineer_data(work, recipe)
st.session_state["engineered_df"] = processed

m1, m2, m3, m4 = st.columns(4)
m1.metric("Input columns", len(work.columns))
m2.metric("Output columns", len(processed.columns))
m3.metric("New features", len(processed.columns) - len(work.columns))
m4.metric("Output rows", f"{len(processed):,}")

new_cols = [c for c in processed.columns if c not in work.columns]
if new_cols:
    st.subheader("Created features")
    st.dataframe(pd.DataFrame({"New feature": new_cols}), use_container_width=True, hide_index=True)

st.subheader("Processed data preview")
st.dataframe(processed.head(200), use_container_width=True, hide_index=True)

st.subheader("Missing values after processing")
miss = processed.isna().sum().sort_values(ascending=False).reset_index()
miss.columns = ["Column", "Missing"]
miss = miss[miss["Missing"] > 0]
if miss.empty:
    st.success("No missing values remain in the processed dataset.")
else:
    st.dataframe(miss, use_container_width=True, hide_index=True)
    st.caption("Sequential lag/difference features naturally have missing values at the first observation of each Symbol + Timeframe sequence.")

download_csv("Download engineered / transformed dataset", processed, "middle_regime_engineered_data.csv", key="engineered_dl")

st.subheader("Situation presets")
st.markdown("""
- **Exploratory statistics:** use the cleaned original data on Analysis & Insights.
- **Pattern recognition:** use original numeric scale because the clustering page standardizes internally.
- **Random Forest / prediction:** the model pipelines impute, scale numeric values, and one-hot encode categories automatically.
- **Sequential forecasting research:** use this page's lag, rolling, delta, and previous-state features for external experiments or a future outcome-labelled model.
""")
