from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import association_rules, clean_finder, download_csv, load_csv, require_data

st.set_page_config(page_title="Loss Analysis", page_icon="🧯", layout="wide")
st.title("🧯 Loss Analysis & Loss Pattern Mining")
st.caption("Upload a separate loss/trade-history file. The app compares loss situations with the complete Finder history, maps loss rows to nearby market conditions, mines recurring loss combinations, and produces evidence-based avoid/monitor insights.")

history = require_data(uploader_key="loss_history_shared_upload")

with st.sidebar:
    st.subheader("Loss file")
    loss_upload = st.file_uploader("Upload Loss CSV/XLSX/XLS", type=["csv", "xlsx", "xls"], key="loss_file_upload")

if loss_upload is not None:
    try:
        raw = load_csv(loss_upload) if loss_upload.name.lower().endswith(".csv") else pd.read_excel(loss_upload)
        loss_df = raw.copy()
        loss_df.columns = [str(c).strip() for c in loss_df.columns]
        st.session_state["loss_df"] = loss_df
        st.session_state["loss_filename"] = loss_upload.name
    except Exception as exc:
        st.error(f"Could not load loss file: {exc}")

loss_df = st.session_state.get("loss_df")
if loss_df is None or loss_df.empty:
    st.info("Upload your loss CSV/XLSX in the sidebar to start loss analysis. The full Finder history is already loaded and will be used as the comparison baseline.")
    st.stop()

# Normalize common aliases without destroying original columns.
def find_col(columns, aliases):
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    for c in columns:
        low = str(c).strip().lower()
        if any(alias.lower() in low for alias in aliases):
            return c
    return None

symbol_col = find_col(loss_df.columns, ["symbol", "ticker", "instrument", "pair"])
datetime_col = find_col(loss_df.columns, ["datetime", "date time", "timestamp", "time", "open time", "entry time"])
result_col = find_col(loss_df.columns, ["pnl", "profit", "loss", "net profit", "result", "outcome", "status", "win/loss"])
reason_col = find_col(loss_df.columns, ["loss reason", "reason", "why loss", "comment", "note", "notes", "cause"])
decision_col = find_col(loss_df.columns, ["strategy decision", "decision", "signal"])
rank_col = find_col(loss_df.columns, ["decision rank", "rank"])

# User-visible mapping controls for non-standard files.
with st.expander("Column mapping", expanded=False):
    opts = [None] + list(loss_df.columns)
    symbol_col = st.selectbox("Loss Symbol column", opts, index=opts.index(symbol_col) if symbol_col in opts else 0, key="loss_map_symbol")
    datetime_col = st.selectbox("Loss Datetime column", opts, index=opts.index(datetime_col) if datetime_col in opts else 0, key="loss_map_datetime")
    result_col = st.selectbox("Loss result/P&L column", opts, index=opts.index(result_col) if result_col in opts else 0, key="loss_map_result")
    reason_col = st.selectbox("Loss reason column (optional)", opts, index=opts.index(reason_col) if reason_col in opts else 0, key="loss_map_reason")

analysis = loss_df.copy()
if symbol_col:
    analysis["Symbol"] = analysis[symbol_col].astype("string").str.strip()
if datetime_col:
    analysis["Loss Datetime"] = pd.to_datetime(analysis[datetime_col], errors="coerce", utc=True)

# Decide whether this table is loss-only or mixed trade history.
loss_flag = pd.Series(True, index=analysis.index)
if result_col:
    s = analysis[result_col]
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() >= max(3, int(len(s) * 0.5)):
        analysis["Loss Value"] = numeric
        loss_flag = numeric < 0
    else:
        txt = s.astype("string").str.upper()
        win_tokens = r"WIN|PROFIT|TP|SUCCESS"
        loss_tokens = r"LOSS|LOSE|SL|STOP|NEGATIVE"
        inferred = txt.str.contains(loss_tokens, regex=True, na=False) & ~txt.str.contains(win_tokens, regex=True, na=False)
        if inferred.any():
            loss_flag = inferred
        else:
            loss_flag = pd.Series(True, index=analysis.index)
else:
    analysis["Loss Value"] = np.nan

losses = analysis.loc[loss_flag].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Loss rows", f"{len(losses):,}")
c2.metric("Loss symbols", f"{losses['Symbol'].nunique():,}" if "Symbol" in losses else "—")
c3.metric("All history rows", f"{len(history):,}")
if "Loss Value" in losses.columns and losses["Loss Value"].notna().any():
    c4.metric("Total loss value", f"{losses['Loss Value'].sum():,.2f}")
else:
    c4.metric("Loss reasons", f"{losses[reason_col].nunique(dropna=True):,}" if reason_col and reason_col in losses else "—")

# ------------------------------------------------------------------
# Map each loss to the latest Finder context before the loss time.
# ------------------------------------------------------------------
context = losses.copy()
if "Symbol" in context.columns and "Loss Datetime" in context.columns and "Symbol" in history.columns and "Datetime" in history.columns:
    hist = history.copy().sort_values(["Symbol", "Datetime"])
    hist["Symbol"] = hist["Symbol"].astype("string")
    hist = hist.dropna(subset=["Symbol", "Datetime"])
    context = context.dropna(subset=["Symbol", "Loss Datetime"]).sort_values(["Symbol", "Loss Datetime"])
    tolerance_hours = st.sidebar.number_input("History context window (hours)", min_value=1, max_value=168, value=24, step=1, key="loss_context_hours")
    context = pd.merge_asof(
        context,
        hist,
        left_on="Loss Datetime",
        right_on="Datetime",
        by="Symbol",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
        suffixes=("", " History"),
    )

st.subheader("Auto Insights")
overall_loss_rate = len(losses) / len(history) if len(history) else np.nan
insights = []
if "Symbol" in losses.columns and "Symbol" in history.columns:
    hist_counts = history["Symbol"].astype(str).value_counts().rename("History Rows")
    loss_counts = losses["Symbol"].astype(str).value_counts().rename("Losses")
    symbol_risk = pd.concat([loss_counts, hist_counts], axis=1).fillna(0)
    symbol_risk["Loss Rate vs History"] = symbol_risk["Losses"] / symbol_risk["History Rows"].replace(0, np.nan)
    symbol_risk["Relative Risk"] = symbol_risk["Loss Rate vs History"] / overall_loss_rate if overall_loss_rate and not np.isnan(overall_loss_rate) else np.nan
    symbol_risk = symbol_risk.reset_index(names="Symbol").sort_values(["Relative Risk", "Losses"], ascending=False)
    if not symbol_risk.empty:
        top = symbol_risk.iloc[0]
        if top["Losses"] >= 2:
            insights.append(f"**{top['Symbol']}** has the highest observed loss-rate multiplier in the uploaded loss sample ({top['Relative Risk']:.1f}× the history baseline). Treat this as a review candidate, not proof of causation.")

if reason_col and reason_col in losses.columns:
    reasons = losses[reason_col].astype("string").fillna("Unspecified").value_counts()
    if not reasons.empty:
        insights.append(f"The most common recorded loss reason is **{reasons.index[0]}** ({reasons.iloc[0]:,} losses).")

if "Strategy Decision" in context.columns:
    d = context["Strategy Decision"].astype("string").value_counts()
    if not d.empty:
        insights.append(f"The most frequent decision at mapped loss contexts is **{d.index[0]}** ({d.iloc[0]:,} loss contexts).")

if "Middle Standard Regime State" in context.columns:
    r = context["Middle Standard Regime State"].astype("string").value_counts()
    if not r.empty:
        insights.append(f"The most frequent middle-regime state in mapped loss contexts is **{r.index[0]}** ({r.iloc[0]:,} losses).")

if insights:
    for text in insights[:6]:
        st.info(text)
else:
    st.info("Add Symbol and/or Datetime columns to the loss file for richer automatic insights and context matching.")

# ------------------------------------------------------------------
# Symbol risk table
# ------------------------------------------------------------------
if "Symbol" in losses.columns and "Symbol" in history.columns:
    st.subheader("Loss Rate by Symbol")
    hist_counts = history["Symbol"].astype(str).value_counts().rename("History Rows")
    loss_counts = losses["Symbol"].astype(str).value_counts().rename("Losses")
    symbol_risk = pd.concat([loss_counts, hist_counts], axis=1).fillna(0).reset_index(names="Symbol")
    symbol_risk["Loss Rate"] = symbol_risk["Losses"] / symbol_risk["History Rows"].replace(0, np.nan)
    symbol_risk["Baseline Loss Rate"] = overall_loss_rate
    symbol_risk["Relative Risk"] = symbol_risk["Loss Rate"] / overall_loss_rate if overall_loss_rate and not np.isnan(overall_loss_rate) else np.nan
    symbol_risk = symbol_risk.sort_values(["Relative Risk", "Losses"], ascending=False)
    st.dataframe(symbol_risk.style.format({"Loss Rate": "{:.2%}", "Baseline Loss Rate": "{:.2%}", "Relative Risk": "{:.2f}×"}), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(symbol_risk.head(30), x="Symbol", y="Relative Risk", title="Relative Loss Risk by Symbol (1.0× = baseline)"), use_container_width=True)
    download_csv("Download Symbol Loss Risk", symbol_risk, "loss_risk_by_symbol.csv", key="loss_symbol_risk_dl")

# ------------------------------------------------------------------
# Loss decision/regime patterns versus history.
# ------------------------------------------------------------------
comparison_fields = [c for c in ["Strategy Decision", "Decision Rank", "Lower Regime Bias", "Middle Regime Bias", "Higher Standard Regime Bias", "RAS Decision", "Middle Standard Regime State", "Lower Standard Regime State", "Higher Standard Regime State", "Timeframe"] if c in history.columns and c in context.columns]

st.subheader("Loss Conditions vs Full History")
if comparison_fields:
    rows = []
    for field in comparison_fields:
        h = history[field].astype("string").fillna("MISSING").value_counts(normalize=True)
        l = context[field].astype("string").fillna("MISSING").value_counts(normalize=True)
        vals = sorted(set(h.index) | set(l.index))
        for value in vals:
            rows.append({"Field": field, "Value": value, "History Share": h.get(value, 0.0), "Loss Share": l.get(value, 0.0), "Lift": (l.get(value, 0.0) / h.get(value, np.nan))})
    comparison = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).sort_values("Lift", ascending=False)
    st.dataframe(comparison.style.format({"History Share": "{:.1%}", "Loss Share": "{:.1%}", "Lift": "{:.2f}×"}), use_container_width=True, hide_index=True)
    top_comp = comparison.dropna(subset=["Lift"]).head(30)
    if not top_comp.empty:
        top_comp["Condition"] = top_comp["Field"] + " = " + top_comp["Value"].astype(str)
        st.plotly_chart(px.bar(top_comp.iloc[::-1], x="Lift", y="Condition", orientation="h", title="Conditions Overrepresented in Losses"), use_container_width=True)
    download_csv("Download Loss vs History Conditions", comparison, "loss_vs_history_conditions.csv", key="loss_vs_history_dl")
else:
    st.info("No common Finder condition columns were detected in the mapped loss context yet.")

# ------------------------------------------------------------------
# Data mining of loss contexts.
# ------------------------------------------------------------------
if comparison_fields:
    mining_cols = comparison_fields[:]
    st.subheader("Loss Data Mining — Recurring Combinations")
    min_support = st.slider("Minimum loss-combination support", 0.01, 0.50, 0.05, 0.01, key="loss_mining_support")
    rules = association_rules(context, mining_cols[:8], min_support=min_support, top_n=100)
    if rules.empty:
        st.info("No recurring combinations met the selected support threshold.")
    else:
        st.dataframe(rules, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(rules.head(25).iloc[::-1], x="Lift", y="A", orientation="h", hover_data=["B", "Support", "Confidence A→B"], title="Strongest Loss-Context Associations"), use_container_width=True)
        download_csv("Download Loss Mining Rules", rules, "loss_pattern_mining_rules.csv", key="loss_rules_dl")

# ------------------------------------------------------------------
# Reasons and raw loss history.
# ------------------------------------------------------------------
if reason_col and reason_col in losses.columns:
    st.subheader("Loss Reasons")
    reason = losses[reason_col].astype("string").fillna("Unspecified").value_counts().rename_axis("Reason").reset_index(name="Losses")
    st.dataframe(reason, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(reason.head(30), x="Reason", y="Losses", title="Recorded Loss Reasons"), use_container_width=True)

download_csv("Download Enriched Loss Context", context, "enriched_loss_context.csv", key="loss_context_dl")

st.subheader("Uploaded Loss Table")
st.dataframe(losses.head(500), use_container_width=True, hide_index=True)
