from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import chronological_model, download_csv, feature_importance, prepare_features, probability_table, require_data, sidebar_filters

st.set_page_config(page_title="Prediction Lab", page_icon="🔮", layout="wide")
st.title("🔮 Prediction Lab")
st.caption("Chronological ML forecasts. The selected symbol gets its own latest observation; universe models are clearly separated from symbol-specific forecasts.")

base = require_data(uploader_key="prediction_shared_upload")
work = sidebar_filters(base, "pred")

symbols = sorted(work["Symbol"].dropna().astype(str).unique()) if "Symbol" in work.columns else []
scope = st.radio("Forecast scope", ["Selected symbol", "All filtered symbols"], horizontal=True, key="pred_scope")
selected_symbol = st.selectbox("Forecast Symbol", symbols, key="pred_symbol") if symbols else None
model_data = work[work["Symbol"].astype(str) == selected_symbol].copy() if scope == "Selected symbol" and selected_symbol else work.copy()

prediction_options = [c for c in ["Strategy Decision", "Decision Direction", "Middle Standard Regime State", "Symbol"] if c in model_data.columns and model_data[c].nunique(dropna=True) >= 2]
if not prediction_options:
    st.warning("No prediction target has at least two classes in the selected data.")
    st.stop()

target = st.selectbox("Prediction Target", prediction_options, key="pred_target")
mode_label = st.radio("Horizon", ["Next recorded class", "Current class"], horizontal=True, key="pred_horizon")
mode = "next" if mode_label.startswith("Next") else "current"

# For a selected symbol, Symbol is constant and therefore cannot be meaningfully classified.
if scope == "Selected symbol" and target == "Symbol":
    st.warning("Symbol is constant inside a selected-symbol model, so choose Strategy Decision, Decision Direction, or Middle Standard Regime State instead.")
    st.stop()

X, y, full_X = prepare_features(model_data, target, mode)
if len(X) < 40 or y.nunique() < 2:
    st.warning(f"This model needs roughly 40+ usable ordered rows and at least two classes. Current usable rows: {len(X):,}; classes: {y.nunique()}.")
    st.stop()

st.caption(f"Training set: {len(X):,} samples • {y.nunique()} classes • {X.shape[1]} features • scope: {scope}{' • ' + selected_symbol if selected_symbol else ''}")

models = []
for model_name in ["Random Forest", "Gradient Boosting", "Logistic Regression"]:
    try:
        r = chronological_model(X, y, model_name, rf_estimators=350, min_leaf=2)
        if r is not None:
            models.append(r)
    except Exception as exc:
        st.warning(f"{model_name} failed: {exc}")
if not models:
    st.warning("No model could be trained on this selection.")
    st.stop()

metrics = pd.DataFrame([{"Model": r.name, "Accuracy": r.accuracy, "Balanced Accuracy": r.balanced_accuracy, "Macro F1": r.f1_macro} for r in models]).sort_values("Balanced Accuracy", ascending=False)
st.subheader("Model Comparison")
st.dataframe(metrics.style.format({"Accuracy": "{:.1%}", "Balanced Accuracy": "{:.1%}", "Macro F1": "{:.3f}"}), use_container_width=True, hide_index=True)
st.plotly_chart(px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group"), use_container_width=True)

st.subheader(f"Latest {target} Forecast")
# full_X is aligned with model_data order; the last row is the latest in the selected scope.
latest_row = full_X.iloc[[-1]]
all_probs = []
for r in models:
    p = probability_table(r, latest_row)
    p["Model"] = r.name
    all_probs.append(p)
prob_df = pd.concat(all_probs, ignore_index=True)

best = metrics.iloc[0]["Model"]
chosen_classes = prob_df.sort_values("Probability", ascending=False).groupby("Model", as_index=False).first()[["Model", "Class", "Probability"]]
c1, c2 = st.columns(2)
c1.metric("Best validation model", best)
c2.metric("Highest model probability", f"{chosen_classes['Probability'].max():.1%}")
st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", color="Model", barmode="group"), use_container_width=True)
st.dataframe(prob_df.style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)

top = chosen_classes.groupby("Class", as_index=False)["Probability"].mean().sort_values("Probability", ascending=False)
st.subheader("Model Consensus")
st.dataframe(chosen_classes.style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
if not top.empty:
    st.success(f"Average probability leader: **{top.iloc[0]['Class']}** ({top.iloc[0]['Probability']:.1%}). This is a statistical forecast for the selected data, not a guarantee.")

chosen = st.selectbox("Inspect Model", [r.name for r in models], key="pred_inspect")
selected = next(r for r in models if r.name == chosen)
a, b = st.columns(2)
with a:
    st.plotly_chart(px.imshow(selected.confusion, text_auto=True, aspect="auto", title=f"{chosen} Confusion Matrix"), use_container_width=True)
with b:
    importance = feature_importance(selected, 20)
    if not importance.empty:
        st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title=f"{chosen} Feature Importance"), use_container_width=True)

# Symbol-specific latest condition audit.
if "Datetime" in model_data.columns:
    latest_cols = [c for c in ["Datetime", "Symbol", "Timeframe", "Strategy Decision", "Decision Rank", "Middle Standard Regime State", "Lower Standard Regime State", "Higher Standard Regime State", "Regime Maturity Score"] if c in model_data.columns]
    if latest_cols:
        st.subheader("Forecast Input Row")
        st.dataframe(model_data.sort_values("Datetime")[latest_cols].tail(1), use_container_width=True, hide_index=True)

download_csv("Download Prediction Probabilities", prob_df, "prediction_model_probabilities.csv", key="prediction_download")
