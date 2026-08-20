from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from background.processor import run_background_processing
from common import EXPECTED_COLUMNS, find_local_data, load_and_clean_local, load_csv, clean_finder, save_dataset

st.set_page_config(page_title="Middle Regime Intelligence Studio", page_icon="🧠", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

st.title("🧠 Middle Regime Intelligence Studio")
st.caption("Symbol-first and decision-first research workspace. Every research page now loads the same shared dataset independently, so deployment does not depend on visiting Home first.")

# ------------------------------------------------------------------
# Shared data loading
# ------------------------------------------------------------------
local_path = find_local_data(BASE_DIR)
if local_path is not None:
    try:
        signature = (str(local_path.resolve()), int(local_path.stat().st_mtime_ns))
        if st.session_state.get("_auto_loaded_signature") != signature or st.session_state.get("finder_df") is None:
            raw, missing, extras = load_and_clean_local(signature[0], signature[1])
            if not missing and not raw.empty:
                save_dataset(raw, local_path.name)
                st.session_state["_auto_loaded_signature"] = signature
                st.session_state["_auto_loaded_path"] = str(local_path)
                st.session_state["_auto_extra_cols"] = extras
    except Exception as exc:
        st.warning(f"Automatic local-data load failed: {exc}")

with st.sidebar:
    st.header("Shared data source")
    if local_path is not None:
        st.success(f"Auto-detected: {local_path.name}")
    else:
        st.warning("No Finder dataset found beside app.py.")

    upload = st.file_uploader("Upload Finder CSV/XLSX/XLS", type=["csv", "xlsx", "xls"], key="home_shared_upload")
    if upload is not None:
        try:
            raw = load_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
            df, missing, extras = clean_finder(raw)
            if missing:
                st.error("Missing required Finder columns: " + ", ".join(missing[:8]))
            elif df.empty:
                st.error("The uploaded Finder dataset has no rows.")
            else:
                save_dataset(df, upload.name)
                st.success(f"Shared dataset loaded: {len(df):,} rows")
                if extras:
                    st.caption("Extra fields retained: " + ", ".join(extras[:12]))
        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")

loaded = st.session_state.get("finder_df")
if loaded is None or loaded.empty:
    st.subheader("Start here")
    st.write("Upload a Finder dataset in the sidebar, or place a Finder CSV/XLSX beside app.py. The research pages also contain the same shared uploader, so they can be opened directly on Streamlit Cloud.")
    st.dataframe(pd.DataFrame({"Required Finder field": EXPECTED_COLUMNS}), use_container_width=True, hide_index=True)
    st.stop()

df = loaded.copy()

@st.cache_data(show_spinner="Preparing background analysis data…")
def _background_cache(frame: pd.DataFrame):
    return run_background_processing(frame)

try:
    processed_df, engineered_df = _background_cache(df)
    st.session_state["processed_df"] = processed_df
    st.session_state["engineered_df"] = engineered_df
except Exception as exc:
    st.warning(f"Background feature preparation was skipped: {exc}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Symbols", f"{df['Symbol'].nunique() if 'Symbol' in df else 0:,}")
c3.metric("Timeframes", f"{df['Timeframe'].nunique() if 'Timeframe' in df else 0:,}")
c4.metric("Decisions", f"{df['Strategy Decision'].nunique() if 'Strategy Decision' in df else 0:,}")
c5.metric("Middle States", f"{df['Middle Standard Regime State'].nunique() if 'Middle Standard Regime State' in df else 0:,}")

st.subheader("Research pages")
pages = [
    ("📊 Analysis & Insights", "pages/1_Analysis_Insights.py", "Symbol-first and decision-first statistics, ranking, regime relationships, and latest conditions."),
    ("🧩 Pattern Recognition", "pages/2_Pattern_Recognition.py", "Symbol-specific clusters, anomalies, transitions, and similar historical conditions."),
    ("🌲 Random Forest Lab", "pages/3_Random_Forest_Lab.py", "Chronological Random Forest research for Symbol and Strategy Decision."),
    ("🔮 Prediction Lab", "pages/4_Prediction_Lab.py", "Model comparison and symbol/decision forecasts using the selected symbol or full universe."),
    ("⛏️ Data Mining", "pages/5_Data_Mining.py", "Frequent Symbol/Decision combinations, association strength, pivots and segment insights."),
    ("🧯 Loss Analysis", "pages/6_Loss_Analysis.py", "Upload loss history and compare loss situations against the full Finder history to identify avoidable patterns."),
]
for title, path, desc in pages:
    a, b = st.columns([1, 4])
    with a:
        st.page_link(path, label=title)
    with b:
        st.caption(desc)

st.divider()
st.success("Deployment fix active: each research page independently resolves the shared Finder data, so Prediction and Pattern no longer depend on Home-session state.")
st.info("Loss Analysis supports a separate loss CSV/XLSX and automatically maps common fields such as Symbol, Datetime, Decision, Rank, bias, regime state, and reason/outcome columns when present.")
st.warning("Model outputs are statistical research signals, not guarantees of trading profit. The Finder history itself does not provide realized future P/L unless you upload a loss/outcome history.")
