from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import download_csv, numeric_columns, require_data, sidebar_filters

st.set_page_config(page_title="Analysis & Insights", page_icon="📊", layout="wide")
st.title("📊 Analysis & Insights")
st.caption("Symbol-first and decision-first analysis. Every calculation is based on the selected filtered rows; there are no placeholder zero metrics for unavailable fields.")

base = require_data(uploader_key="analysis_shared_upload")
work = sidebar_filters(base, "analysis")

# Symbol-focused selection is separate from the global filters.
symbols = sorted(work["Symbol"].dropna().astype(str).unique()) if "Symbol" in work.columns else []
selected_symbol = st.selectbox("Focus Symbol", symbols, index=0 if symbols else None, key="analysis_focus_symbol") if symbols else None
focus = work[work["Symbol"].astype(str) == selected_symbol].copy() if selected_symbol else work.copy()

st.subheader("Core Metrics")
metrics = [
    ("Rows", len(focus)),
    ("Symbols", focus["Symbol"].nunique() if "Symbol" in focus else None),
    ("Decisions", focus["Strategy Decision"].nunique() if "Strategy Decision" in focus else None),
    ("Rank Classes", focus["Decision Rank"].nunique() if "Decision Rank" in focus else None),
    ("Middle States", focus["Middle Standard Regime State"].nunique() if "Middle Standard Regime State" in focus else None),
]
cols = st.columns(len(metrics))
for c, (label, value) in zip(cols, metrics):
    c.metric(label, "—" if value is None else f"{value:,}")

if "Regime Maturity Score" in focus.columns:
    c = st.container()
    c.metric("Focused Median Maturity", f"{focus['Regime Maturity Score'].median():.2f}" if focus["Regime Maturity Score"].notna().any() else "—")

# Decision distribution for the selected symbol.
left, right = st.columns(2)
with left:
    st.subheader(f"{selected_symbol or 'Universe'} — Decision Distribution")
    if "Strategy Decision" in focus.columns:
        d = focus["Strategy Decision"].fillna("MISSING").value_counts().rename_axis("Strategy Decision").reset_index(name="Rows")
        st.plotly_chart(px.bar(d, x="Strategy Decision", y="Rows", text_auto=True), use_container_width=True)
    else:
        st.info("Strategy Decision is not available in this dataset.")
with right:
    st.subheader(f"{selected_symbol or 'Universe'} — Decision Rank")
    if "Decision Rank Score" in focus.columns:
        r = focus.dropna(subset=["Decision Rank Score"]).groupby("Decision Rank", as_index=False).agg(Rows=("Decision Rank Score", "size"), AvgScore=("Decision Rank Score", "mean"))
        st.plotly_chart(px.bar(r, x="Decision Rank", y="AvgScore", text_auto=".2f"), use_container_width=True)
        st.dataframe(r.sort_values("AvgScore", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Decision rank score is unavailable.")

st.subheader(f"{selected_symbol or 'Universe'} — Regime Profile")
regime_cols = [c for c in ["Lower Standard Regime State", "Middle Standard Regime State", "Higher Standard Regime State"] if c in focus.columns]
if regime_cols:
    regime_counts = []
    for c in regime_cols:
        tmp = focus[c].fillna("MISSING").value_counts().rename_axis("State").reset_index(name="Rows")
        tmp["Regime Level"] = c.replace(" Standard Regime State", "")
        regime_counts.append(tmp)
    reg = pd.concat(regime_counts, ignore_index=True)
    st.plotly_chart(px.bar(reg, x="State", y="Rows", color="Regime Level", barmode="group"), use_container_width=True)
else:
    st.info("Regime state columns are unavailable.")

# Symbol × decision comparison: genuine per-symbol aggregates.
if "Symbol" in work.columns and "Strategy Decision" in work.columns:
    st.subheader("Symbol × Decision Comparison")
    comparison = work.groupby(["Symbol", "Strategy Decision"], dropna=False).size().reset_index(name="Rows")
    comparison["Share within Symbol"] = comparison["Rows"] / comparison.groupby("Symbol")["Rows"].transform("sum")
    st.plotly_chart(px.bar(comparison, x="Symbol", y="Share within Symbol", color="Strategy Decision", barmode="stack", title="Decision Mix Within Each Symbol"), use_container_width=True)
    st.dataframe(comparison.sort_values(["Symbol", "Share within Symbol"], ascending=[True, False]), use_container_width=True, hide_index=True)
    download_csv("Download Symbol × Decision Summary", comparison, "symbol_decision_summary.csv", key="analysis_symbol_decision_dl")

# Numeric symbol profile.
nums = numeric_columns(focus)
if nums:
    st.subheader(f"{selected_symbol or 'Universe'} — Numeric Profile")
    selected_num = st.selectbox("Numeric feature", nums, key="analysis_numeric_feature")
    if selected_num:
        stat = focus[selected_num].describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_frame().T
        st.dataframe(stat, use_container_width=True, hide_index=True)
        color = "Strategy Decision" if "Strategy Decision" in focus.columns else None
        st.plotly_chart(px.histogram(focus, x=selected_num, color=color, nbins=40, marginal="box"), use_container_width=True)

# Time behaviour for the selected symbol.
if "Datetime" in focus.columns and "Strategy Decision" in focus.columns and not focus.empty:
    st.subheader(f"{selected_symbol or 'Universe'} — Decision Through Time")
    trend = focus.dropna(subset=["Datetime"]).copy()
    trend["Date"] = trend["Datetime"].dt.floor("D")
    daily = trend.groupby(["Date", "Strategy Decision"], dropna=False).size().reset_index(name="Rows")
    st.plotly_chart(px.line(daily, x="Date", y="Rows", color="Strategy Decision", markers=True), use_container_width=True)

# Latest per symbol, not repeated all-symbol snapshot.
if "Datetime" in work.columns:
    st.subheader("Latest Condition Per Symbol")
    latest = work.sort_values("Datetime").groupby("Symbol", dropna=False, as_index=False).tail(1)
    latest_cols = [c for c in ["Datetime", "Symbol", "Timeframe", "Strategy Decision", "Decision Rank", "Middle Standard Regime State", "Regime Maturity Score", "Candle After Regime Start"] if c in latest.columns]
    if latest_cols:
        latest = latest[latest_cols].sort_values("Datetime", ascending=False)
        st.dataframe(latest, use_container_width=True, hide_index=True)
        download_csv("Download Latest Per Symbol", latest, "latest_per_symbol.csv", key="analysis_latest_symbol_dl")
