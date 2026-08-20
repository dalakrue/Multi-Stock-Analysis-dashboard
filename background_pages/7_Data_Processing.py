from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import download_csv, numeric_columns, require_data, sidebar_filters

st.set_page_config(page_title="Data Processing", page_icon="⚙️", layout="wide")
st.title("⚙️ Data Processing & Transformation")
st.caption("Build a custom processed copy of the Finder data without changing the original session dataset.")

work = sidebar_filters(require_data(uploader_key="shared_proc_upload"), "process")
processed = work.copy()
nums = numeric_columns(processed)

st.subheader("Processing controls")
c1, c2, c3 = st.columns(3)
with c1:
    remove_dupes = st.checkbox("Remove exact duplicate rows", value=True)
    fill_numeric = st.checkbox("Fill numeric missing values with median", value=True)
with c2:
    fill_category = st.checkbox("Fill categorical missing values with mode", value=True)
    clip_outliers = st.checkbox("Clip numeric outliers", value=False)
with c3:
    clip_low = st.number_input("Lower quantile", min_value=0.0, max_value=0.49, value=0.01, step=0.01, disabled=not clip_outliers)
    clip_high = st.number_input("Upper quantile", min_value=0.51, max_value=1.0, value=0.99, step=0.01, disabled=not clip_outliers)

if remove_dupes:
    processed = processed.drop_duplicates().copy()

if fill_numeric:
    for c in nums:
        processed[c] = processed[c].replace([np.inf, -np.inf], np.nan)
        if processed[c].notna().any():
            processed[c] = processed[c].fillna(processed[c].median())

if fill_category:
    cat_cols = [c for c in processed.columns if c not in nums and c not in ("Datetime", "Date")]
    for c in cat_cols:
        if processed[c].isna().any():
            mode = processed[c].mode(dropna=True)
            if not mode.empty:
                processed[c] = processed[c].fillna(mode.iloc[0])

if clip_outliers:
    for c in nums:
        if processed[c].notna().sum() > 4:
            lo, hi = processed[c].quantile([clip_low, clip_high])
            processed[c] = processed[c].clip(lo, hi)

st.subheader("Numeric transformations")
transform_cols = st.multiselect("Numeric fields to transform", nums, default=[c for c in ["Avg Regime Candle Rank", "Regime Maturity Score", "Candle After Regime Start"] if c in nums])
transform = st.selectbox("Transformation", ["No extra transformation", "Z-score copies", "Min-Max copies", "Robust-IQR copies", "Signed log1p copies"])

for c in transform_cols:
    s = pd.to_numeric(processed[c], errors="coerce")
    if transform == "Z-score copies":
        std = s.std()
        processed[f"Z_{c}"] = (s - s.mean()) / std if std and not np.isnan(std) else 0.0
    elif transform == "Min-Max copies":
        lo, hi = s.min(), s.max()
        processed[f"MinMax_{c}"] = (s - lo) / (hi - lo) if hi > lo else 0.0
    elif transform == "Robust-IQR copies":
        med = s.median(); q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        processed[f"Robust_{c}"] = (s - med) / iqr if iqr and not np.isnan(iqr) else 0.0
    elif transform == "Signed log1p copies":
        processed[f"Log1p_{c}"] = np.sign(s) * np.log1p(np.abs(s))

st.session_state["processed_df"] = processed

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows before", f"{len(work):,}")
m2.metric("Rows after", f"{len(processed):,}")
m3.metric("Columns before", len(work.columns))
m4.metric("Columns after", len(processed.columns))

st.subheader("Before / after quality")
compare = pd.DataFrame({
    "Metric": ["Duplicate rows", "Missing cells", "Numeric columns", "Total columns"],
    "Before": [int(work.duplicated().sum()), int(work.isna().sum().sum()), len(numeric_columns(work)), len(work.columns)],
    "After": [int(processed.duplicated().sum()), int(processed.isna().sum().sum()), len(numeric_columns(processed)), len(processed.columns)],
})
st.dataframe(compare, use_container_width=True, hide_index=True)

if transform_cols:
    feature = st.selectbox("Visual comparison feature", transform_cols)
    transformed_candidates = [c for c in processed.columns if c.endswith(feature) and c != feature]
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(work, x=feature, nbins=40, marginal="box", title=f"Before: {feature}"), use_container_width=True)
    with b:
        if transformed_candidates:
            tc = transformed_candidates[-1]
            st.plotly_chart(px.histogram(processed, x=tc, nbins=40, marginal="box", title=f"After: {tc}"), use_container_width=True)
        else:
            st.plotly_chart(px.histogram(processed, x=feature, nbins=40, marginal="box", title=f"After cleaning: {feature}"), use_container_width=True)

st.subheader("Processed data preview")
st.dataframe(processed.head(200), use_container_width=True, hide_index=True)
download_csv("Download custom processed dataset", processed, "middle_regime_processed_data.csv", key="processed_dl")
