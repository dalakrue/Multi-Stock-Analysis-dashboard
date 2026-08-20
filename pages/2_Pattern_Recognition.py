from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import anomaly_table, cluster_table, nearest_history, require_data, sidebar_filters, transition_matrix

st.set_page_config(page_title="Pattern Recognition", page_icon="🧩", layout="wide")
st.title("🧩 Pattern Recognition")
st.caption("Patterns are calculated from the selected symbol or universe—not from one global snapshot copied across symbols.")

base = require_data(uploader_key="pattern_shared_upload")
work = sidebar_filters(base, "pattern")

symbols = sorted(work["Symbol"].dropna().astype(str).unique()) if "Symbol" in work.columns else []
comparison_mode = st.radio("Pattern scope", ["Selected symbol", "All filtered symbols"], horizontal=True, key="pattern_scope")
selected_symbol = st.selectbox("Pattern Symbol", symbols, key="pattern_symbol") if symbols else None
focus = work[work["Symbol"].astype(str) == selected_symbol].copy() if comparison_mode == "Selected symbol" and selected_symbol else work.copy()

if comparison_mode == "Selected symbol":
    st.info(f"All pattern calculations below use only **{selected_symbol}** rows: {len(focus):,} observations.")
else:
    st.info(f"Universe pattern calculations use all filtered symbols: {work['Symbol'].nunique() if 'Symbol' in work else 1} symbols, {len(work):,} observations.")

clusters_tab, anomaly_tab, transition_tab, similar_tab = st.tabs(["Clusters & PCA", "Anomalies", "Transitions", "Similar History"])

with clusters_tab:
    st.subheader("Feature Clusters")
    max_k = min(10, max(2, len(focus) // 5))
    k = st.slider("Number of clusters", 2, max_k, min(4, max_k), key="pattern_k")
    result = cluster_table(focus, k)
    if result is None:
        st.warning("Not enough usable numeric data for clustering. Select a symbol with more history or use All filtered symbols.")
    else:
        clustered, centers = result
        hover = [c for c in ["Datetime", "Symbol", "Strategy Decision", "Decision Rank", "Middle Standard Regime State", "Timeframe"] if c in clustered.columns]
        st.plotly_chart(px.scatter(clustered, x="PCA 1", y="PCA 2", color="Pattern Cluster", hover_data=hover, title="PCA Pattern Map"), use_container_width=True)
        profile = clustered.groupby("Pattern Cluster").agg(Rows=("Pattern Cluster", "size")).reset_index()
        if "Symbol" in clustered.columns:
            profile["Symbols"] = clustered.groupby("Pattern Cluster")["Symbol"].nunique().values
        if "Strategy Decision" in clustered.columns:
            profile["Decisions"] = clustered.groupby("Pattern Cluster")["Strategy Decision"].nunique().values
        st.dataframe(profile, use_container_width=True, hide_index=True)
        st.dataframe(centers, use_container_width=True, hide_index=True)
        if "Symbol" in clustered.columns:
            sym_cluster = clustered.groupby(["Pattern Cluster", "Symbol"], dropna=False).size().reset_index(name="Rows")
            st.plotly_chart(px.bar(sym_cluster, x="Symbol", y="Rows", color="Pattern Cluster", barmode="group", title="Pattern Cluster by Symbol"), use_container_width=True)
        if "Strategy Decision" in clustered.columns:
            dec_cluster = clustered.groupby(["Pattern Cluster", "Strategy Decision"], dropna=False).size().reset_index(name="Rows")
            st.plotly_chart(px.bar(dec_cluster, x="Strategy Decision", y="Rows", color="Pattern Cluster", barmode="group", title="Pattern Cluster by Decision"), use_container_width=True)

with anomaly_tab:
    st.subheader("Unusual Symbol / Decision Conditions")
    contamination = st.slider("Expected anomaly share", 0.01, 0.20, 0.05, 0.01, key="pattern_contamination")
    anomalies = anomaly_table(focus, contamination=contamination)
    if anomalies is None:
        st.warning("Not enough numeric history for anomaly detection.")
    else:
        show_cols = [c for c in ["Anomaly", "Anomaly Score", "Datetime", "Symbol", "Strategy Decision", "Decision Rank", "Middle Standard Regime State", "Regime Maturity Score"] if c in anomalies.columns]
        st.dataframe(anomalies[show_cols].head(150), use_container_width=True, hide_index=True)
        if "Symbol" in anomalies.columns:
            sym = anomalies[anomalies["Anomaly"] == "ANOMALY"]["Symbol"].value_counts().rename_axis("Symbol").reset_index(name="Anomalies")
            if not sym.empty:
                st.plotly_chart(px.bar(sym, x="Symbol", y="Anomalies", title="Anomalies by Symbol"), use_container_width=True)

with transition_tab:
    st.subheader("State / Decision Transitions")
    transition_col = st.selectbox("Field", [c for c in ["Strategy Decision", "Decision Rank", "Middle Standard Regime State", "RAS Decision"] if c in focus.columns], key="pattern_transition_col")
    counts, probs = transition_matrix(focus, transition_col)
    if counts.empty:
        st.info("No transitions are available for this selection.")
    else:
        st.markdown("**Transition count**")
        st.dataframe(counts, use_container_width=True)
        st.markdown("**Transition probability**")
        st.dataframe(probs.style.format("{:.1%}"), use_container_width=True)
        st.plotly_chart(px.imshow(probs, text_auto=".0%", aspect="auto", title=f"Next {transition_col} probability"), use_container_width=True)

with similar_tab:
    st.subheader("Similar Historical Conditions")
    n = st.slider("Historical matches", 5, 100, 25, 5, key="pattern_n")
    similar = nearest_history(focus, n=n)
    if similar is None:
        st.warning("Not enough numeric observations for similarity search.")
    else:
        show = [c for c in ["Similarity Distance", "Datetime", "Symbol", "Timeframe", "Strategy Decision", "Decision Rank", "Middle Standard Regime State", "Regime Maturity Score"] if c in similar.columns]
        st.dataframe(similar[show].head(n), use_container_width=True, hide_index=True)
        if "Strategy Decision" in similar.columns:
            dec = similar["Strategy Decision"].value_counts().rename_axis("Strategy Decision").reset_index(name="Matches")
            st.plotly_chart(px.bar(dec, x="Strategy Decision", y="Matches", title="Decisions in Similar History"), use_container_width=True)
        if "Symbol" in similar.columns:
            sym = similar["Symbol"].value_counts().rename_axis("Symbol").reset_index(name="Matches")
            st.plotly_chart(px.bar(sym, x="Symbol", y="Matches", title="Symbols in Similar History"), use_container_width=True)
