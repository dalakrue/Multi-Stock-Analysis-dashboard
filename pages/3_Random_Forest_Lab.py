from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    chronological_model,
    feature_importance,
    prepare_features,
    probability_table,
    require_data,
    sidebar_filters,
    download_csv,
)

st.set_page_config(
    page_title="Random Forest Lab",
    page_icon="🌲",
    layout="wide",
)

st.title("🌲 Random Forest Lab")

st.caption(
    "Dedicated Random Forest research for Symbol and Strategy Decision prediction, "
    "with chronological testing, feature importance, probability forecasts, and automatic insights."
)

# =========================================================
# LOAD DATA
# =========================================================

work = sidebar_filters(
    require_data(uploader_key="shared_rf_upload"),
    "rf",
)

# =========================================================
# AVAILABLE TARGETS
# =========================================================

target_options = []

if (
    "Symbol" in work.columns
    and work["Symbol"].nunique(dropna=True) >= 2
):
    target_options.append("Symbol")

if (
    "Strategy Decision" in work.columns
    and work["Strategy Decision"].nunique(dropna=True) >= 2
):
    target_options.append("Strategy Decision")

if not target_options:
    st.warning(
        "No usable classification target was found. "
        "The dataset needs at least two classes in Symbol or Strategy Decision."
    )
    st.stop()

# =========================================================
# MODEL SETTINGS
# =========================================================

left, right = st.columns([2, 1])

with left:

    target = st.selectbox(
        "Prediction Target",
        target_options,
        index=0,
    )

    mode_label = st.radio(
        "Learning Task",
        [
            "Predict next recorded class",
            "Explain / classify current class",
        ],
        horizontal=True,
    )

with right:

    estimators = st.slider(
        "Trees",
        100,
        1200,
        500,
        100,
    )

    depth_choice = st.selectbox(
        "Maximum tree depth",
        [
            "Unlimited",
            5,
            8,
            12,
            16,
            24,
        ],
    )

    min_leaf = st.slider(
        "Minimum samples per leaf",
        1,
        20,
        2,
    )

mode = (
    "next"
    if mode_label.startswith("Predict next")
    else "current"
)

# =========================================================
# PREPARE FEATURES
# =========================================================

X, y, full_X = prepare_features(
    work,
    target,
    mode,
)

st.caption(
    f"Usable samples: {len(X):,} • "
    f"Features: {X.shape[1]:,} • "
    f"Classes: {y.nunique():,}"
)

max_depth = (
    None
    if depth_choice == "Unlimited"
    else int(depth_choice)
)

# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

try:

    result = chronological_model(
        X,
        y,
        "Random Forest",
        rf_estimators=estimators,
        rf_depth=max_depth,
        min_leaf=min_leaf,
    )

except Exception as e:

    st.error(
        f"Random Forest could not be trained: {e}"
    )
    st.stop()

if result is None:

    st.info(
        "Need roughly 40+ ordered samples, at least two classes, "
        "and more than one class in the training period."
    )
    st.stop()

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{result.accuracy:.1%}",
)

m2.metric(
    "Balanced Accuracy",
    f"{result.balanced_accuracy:.1%}",
)

m3.metric(
    "Macro F1",
    f"{result.f1_macro:.3f}",
)

m4.metric(
    "Training / Test Rows",
    f"{result.split_index} / {len(X) - result.split_index}",
)

st.caption(
    "Evaluation uses a chronological split: the earliest 80% of usable observations "
    "train the model and the latest 20% test it. The final probability model is then "
    "refit using all usable history."
)

# =========================================================
# PERFORMANCE INTERPRETATION
# =========================================================

if result.balanced_accuracy >= 0.80:

    performance_label = "Very strong"

elif result.balanced_accuracy >= 0.70:

    performance_label = "Strong"

elif result.balanced_accuracy >= 0.60:

    performance_label = "Moderate"

elif result.balanced_accuracy >= 0.50:

    performance_label = "Weak"

else:

    performance_label = "Very weak"

if result.f1_macro >= 0.80:

    f1_label = "high"

elif result.f1_macro >= 0.65:

    f1_label = "reasonable"

else:

    f1_label = "limited"

# =========================================================
# CONFUSION MATRIX + CLASSIFICATION REPORT
# =========================================================

c1, c2 = st.columns(2)

with c1:

    st.subheader("Confusion Matrix")

    st.plotly_chart(
        px.imshow(
            result.confusion,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Predicted",
                "y": "Actual",
            },
            title=f"{target} Confusion Matrix",
        ),
        use_container_width=True,
    )

    st.dataframe(
        result.confusion,
        use_container_width=True,
    )

with c2:

    st.subheader("Classification Report")

    st.dataframe(
        result.report,
        use_container_width=True,
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🌳 What the Random Forest Is Using")

importance = feature_importance(
    result,
    35,
)

if not importance.empty:

    st.plotly_chart(
        px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Random Forest Feature Importances",
        ),
        use_container_width=True,
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Feature importance is unavailable for this model configuration."
    )

# =========================================================
# LATEST ROW PROBABILITY FORECAST
# =========================================================

st.subheader(
    f"Latest-Row {target} Probability Forecast"
)

latest_row = full_X.iloc[[-1]]

probs = probability_table(
    result,
    latest_row,
).copy()

if probs.empty:

    st.warning(
        "No probability forecast was returned."
    )
    st.stop()

# Make sure probabilities are numeric
if "Probability" in probs.columns:

    probs["Probability"] = pd.to_numeric(
        probs["Probability"],
        errors="coerce",
    )

probs = (
    probs.sort_values(
        "Probability",
        ascending=False,
    )
    .reset_index(drop=True)
)

top_class = probs.iloc[0]["Class"]

top_probability = probs.iloc[0]["Probability"]

second_probability = (
    probs.iloc[1]["Probability"]
    if len(probs) > 1
    else None
)

# =========================================================
# CONFIDENCE INTERPRETATION
# =========================================================

if pd.notna(top_probability):

    if top_probability >= 0.85:

        confidence_label = "Very high"

    elif top_probability >= 0.70:

        confidence_label = "High"

    elif top_probability >= 0.55:

        confidence_label = "Moderate"

    else:

        confidence_label = "Low"

else:

    confidence_label = "Unavailable"

# =========================================================
# PROBABILITY METRICS
# =========================================================

a, b, c = st.columns(3)

with a:

    st.metric(
        "Top Predicted Class",
        str(top_class),
    )

with b:

    st.metric(
        "Top Probability",
        f"{top_probability:.1%}"
        if pd.notna(top_probability)
        else "—",
    )

with c:

    st.metric(
        "Model Confidence",
        confidence_label,
    )

# =========================================================
# PROBABILITY CHART
# =========================================================

st.plotly_chart(
    px.bar(
        probs,
        x="Class",
        y="Probability",
        text_auto=".1%",
        title=f"Random Forest {target} Class Probabilities",
    ),
    use_container_width=True,
)

st.dataframe(
    probs.style.format(
        {
            "Probability": "{:.1%}"
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

st.subheader(
    f"Historical {target} Distribution"
)

class_distribution = (
    y.value_counts(
        normalize=True
    )
    .rename("Share")
    .reset_index()
)

class_distribution.columns = [
    "Class",
    "Share",
]

st.plotly_chart(
    px.bar(
        class_distribution,
        x="Class",
        y="Share",
        text_auto=".1%",
        title=f"Historical Distribution of {target}",
    ),
    use_container_width=True,
)

# =========================================================
# AUTOMATIC RESEARCH INSIGHTS
# =========================================================

st.subheader("🧠 Automatic Model Insights")

insights = []

# ---------------------------------------------------------
# Model quality
# ---------------------------------------------------------

insights.append(
    f"Random Forest performance is **{performance_label.lower()}** "
    f"with {result.balanced_accuracy:.1%} balanced accuracy and "
    f"a macro F1 of {result.f1_macro:.3f}."
)

# ---------------------------------------------------------
# Accuracy vs balanced accuracy
# ---------------------------------------------------------

accuracy_gap = (
    result.accuracy
    - result.balanced_accuracy
)

if abs(accuracy_gap) < 0.03:

    insights.append(
        "Accuracy and balanced accuracy are close, suggesting that "
        "the model is not heavily benefiting from class imbalance."
    )

elif accuracy_gap > 0:

    insights.append(
        f"Accuracy is {accuracy_gap:.1%} above balanced accuracy. "
        "This suggests that some classes may be easier or more frequent "
        "than others, so balanced accuracy is the safer metric to monitor."
    )

else:

    insights.append(
        f"Balanced accuracy is {abs(accuracy_gap):.1%} above raw accuracy. "
        "This indicates that class-level performance should be inspected carefully."
    )

# ---------------------------------------------------------
# Prediction confidence
# ---------------------------------------------------------

if pd.notna(top_probability):

    insights.append(
        f"The latest observation is predicted as **{top_class}** "
        f"with {top_probability:.1%} probability, giving a "
        f"**{confidence_label.lower()} confidence** forecast."
    )

# ---------------------------------------------------------
# Probability separation
# ---------------------------------------------------------

if (
    second_probability is not None
    and pd.notna(top_probability)
    and pd.notna(second_probability)
):

    probability_gap = (
        top_probability
        - second_probability
    )

    if probability_gap >= 0.40:

        insights.append(
            f"The leading class is separated from the second class by "
            f"{probability_gap:.1%}, indicating relatively clear model preference."
        )

    elif probability_gap >= 0.15:

        insights.append(
            f"The leading class is only {probability_gap:.1%} ahead of the "
            "second class, so the forecast has meaningful competition between classes."
        )

    else:

        insights.append(
            f"The top two classes are very close, with only a "
            f"{probability_gap:.1%} probability gap. This is a low-separation "
            "forecast and should be treated as uncertain."
        )

# ---------------------------------------------------------
# Top important features
# ---------------------------------------------------------

if not importance.empty:

    top_features = importance.head(5)[
        ["Feature", "Importance"]
    ].copy()

    feature_names = ", ".join(
        str(x)
        for x in top_features["Feature"].tolist()
    )

    insights.append(
        f"The strongest model inputs are **{feature_names}**. "
        "These variables contribute most to the Random Forest's class separation."
    )

# ---------------------------------------------------------
# Number of classes
# ---------------------------------------------------------

num_classes = y.nunique()

if num_classes <= 3:

    insights.append(
        f"The model is separating {num_classes} classes, so the current "
        "classification problem is relatively compact."
    )

elif num_classes <= 6:

    insights.append(
        f"The model is distinguishing {num_classes} classes, creating "
        "moderate classification complexity."
    )

else:

    insights.append(
        f"The model is distinguishing {num_classes} classes. "
        "Higher class count increases the possibility of overlapping class profiles."
    )

# =========================================================
# RENDER INSIGHTS
# =========================================================

for i, insight in enumerate(insights, start=1):

    st.markdown(
        f"**{i}.** {insight}"
    )

# =========================================================
# AUTOMATIC RESEARCH REPORT
# =========================================================

st.subheader("📋 Automatic Research Report")

latest_datetime = None

if "Datetime" in work.columns:

    try:
        latest_datetime = (
            work["Datetime"]
            .dropna()
            .max()
        )
    except Exception:
        latest_datetime = None

latest_symbol = None

if "Symbol" in work.columns and len(work) > 0:

    try:

        latest_symbol = (
            work.sort_values("Datetime")
            .iloc[-1]["Symbol"]
            if "Datetime" in work.columns
            else work.iloc[-1]["Symbol"]
        )

    except Exception:

        latest_symbol = None

report_parts = []

report_parts.append(
    f"The Random Forest was trained to predict **{target}** using "
    f"{len(X):,} chronologically ordered observations and "
    f"{X.shape[1]:,} model features."
)

report_parts.append(
    f"Chronological out-of-sample performance produced "
    f"{result.accuracy:.1%} accuracy, "
    f"{result.balanced_accuracy:.1%} balanced accuracy, "
    f"and {result.f1_macro:.3f} macro F1."
)

if latest_datetime is not None:

    report_parts.append(
        f"The latest available observation is dated **{latest_datetime}**."
    )

if latest_symbol is not None:

    report_parts.append(
        f"The latest row is associated with symbol **{latest_symbol}**."
    )

if pd.notna(top_probability):

    report_parts.append(
        f"The forward Random Forest forecast favors **{top_class}** "
        f"with a probability of **{top_probability:.1%}**."
    )

if not importance.empty:

    top_feature = importance.iloc[0]["Feature"]

    top_importance_value = importance.iloc[0]["Importance"]

    report_parts.append(
        f"The single most influential model feature is "
        f"**{top_feature}** with a relative importance of "
        f"**{top_importance_value:.4f}**."
    )

# ---------------------------------------------------------
# Automatic conclusion
# ---------------------------------------------------------

if (
    result.balanced_accuracy >= 0.70
    and pd.notna(top_probability)
    and top_probability >= 0.70
):

    conclusion = (
        "Overall, the model shows both reasonable historical classification "
        "skill and a relatively confident latest-row forecast. The strongest "
        "attention should be given to whether the same feature relationships "
        "remain stable in newer observations."
    )

elif (
    result.balanced_accuracy >= 0.60
    and pd.notna(top_probability)
):

    conclusion = (
        "Overall, the model contains useful predictive structure, but the "
        "latest forecast is not strong enough to be treated as a standalone "
        "decision signal. Model probability, class balance, and feature stability "
        "should be evaluated together."
    )

else:

    conclusion = (
        "Overall, the model currently shows limited out-of-sample classification "
        "strength. The prediction should be treated as exploratory rather than "
        "as a reliable standalone signal."
    )

report_parts.append(
    conclusion
)

report_text = "\n\n".join(
    report_parts
)

st.text_area(
    "Generated Research Report",
    report_text,
    height=260,
)

# =========================================================
# RESEARCH SUMMARY TABLE
# =========================================================

st.subheader("Research Summary")

summary_data = {
    "Target": target,
    "Learning Task": mode_label,
    "Samples": len(X),
    "Features": X.shape[1],
    "Classes": y.nunique(),
    "Trees": estimators,
    "Max Depth": (
        "Unlimited"
        if max_depth is None
        else max_depth
    ),
    "Min Leaf": min_leaf,
    "Accuracy": result.accuracy,
    "Balanced Accuracy": result.balanced_accuracy,
    "Macro F1": result.f1_macro,
    "Top Class": top_class,
    "Top Probability": top_probability,
}

summary_df = pd.DataFrame(
    [summary_data]
)

st.dataframe(
    summary_df.style.format(
        {
            "Accuracy": "{:.1%}",
            "Balanced Accuracy": "{:.1%}",
            "Macro F1": "{:.3f}",
            "Top Probability": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# DOWNLOAD
# =========================================================

download_csv(
    "Download Random Forest probabilities",
    probs,
    "random_forest_probabilities.csv",
    key="rf_probs_dl",
)

# =========================================================
# FINAL NOTE
# =========================================================

st.warning(
    "Random Forest classification performance does not automatically mean "
    "profitable trading performance. For direct outcome prediction, use future "
    "return, future price movement, or realized P&L labels rather than relying "
    "only on classification accuracy."
)