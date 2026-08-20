from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    association_rules,
    categorical_columns,
    numeric_columns,
    require_data,
    sidebar_filters,
    download_csv,
)

st.set_page_config(
    page_title="Data Mining",
    page_icon="⛏️",
    layout="wide",
)

st.title("⛏️ Data Mining")

st.caption(
    "Mine recurring Symbol, Ranking, Decision, and Regime relationships, "
    "discover strong associations, compare segments, and generate automatic research insights."
)

work = sidebar_filters(
    require_data(uploader_key="shared_mine_upload"),
    "mine",
)

cats = categorical_columns(
    work,
    max_unique=60,
)

nums = numeric_columns(work)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def strength_label(value: float) -> str:
    if value >= 0.80:
        return "very strong"
    if value >= 0.60:
        return "strong"
    if value >= 0.40:
        return "moderate"
    if value >= 0.20:
        return "weak"
    return "very weak"


# =========================================================
# PAGE OVERVIEW
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Rows",
    f"{len(work):,}",
)

c2.metric(
    "Symbols",
    f"{work['Symbol'].nunique(dropna=True):,}"
    if "Symbol" in work.columns
    else "—",
)

c3.metric(
    "Decisions",
    f"{work['Strategy Decision'].nunique(dropna=True):,}"
    if "Strategy Decision" in work.columns
    else "—",
)

c4.metric(
    "Decision Ranks",
    f"{work['Decision Rank'].nunique(dropna=True):,}"
    if "Decision Rank" in work.columns
    else "—",
)

c5.metric(
    "Regime States",
    f"{work['Middle Standard Regime State'].nunique(dropna=True):,}"
    if "Middle Standard Regime State" in work.columns
    else "—",
)


# =========================================================
# 1. FREQUENT COMBINATIONS
# =========================================================

st.subheader("🔎 Frequent Symbol / Decision Combinations")

default_combo = [
    c
    for c in [
        "Symbol",
        "Strategy Decision",
        "Decision Rank",
        "Middle Standard Regime State",
        "Timeframe",
    ]
    if c in cats
][:4]

combo_cols = st.multiselect(
    "Columns to combine",
    cats,
    default=default_combo,
    max_selections=5,
)

combos = pd.DataFrame()

if combo_cols:

    combos = (
        work.groupby(
            combo_cols,
            dropna=False,
        )
        .size()
        .reset_index(name="Count")
        .sort_values(
            "Count",
            ascending=False,
        )
        .head(100)
    )

    combos["Share"] = (
        combos["Count"] / max(len(work), 1)
    )

    st.dataframe(
        combos.style.format(
            {"Share": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    if len(combo_cols) >= 2:

        plot = combos.head(30).copy()

        plot["Combination"] = (
            plot[combo_cols]
            .astype(str)
            .agg(" | ".join, axis=1)
        )

        st.plotly_chart(
            px.bar(
                plot.sort_values("Count"),
                x="Count",
                y="Combination",
                orientation="h",
                title="Top Frequent Multi-Field Patterns",
            ),
            use_container_width=True,
        )

else:

    st.info(
        "Select at least one categorical field to mine frequent combinations."
    )


# =========================================================
# 2. ASSOCIATION MINING
# =========================================================

st.subheader("🔗 Association Mining")

rule_default = [
    c
    for c in [
        "Symbol",
        "Strategy Decision",
        "Decision Rank",
        "Middle Standard Regime State",
        "RAS Decision",
        "Timeframe",
    ]
    if c in cats
][:5]

rule_cols = st.multiselect(
    "Categorical fields for association rules",
    cats,
    default=rule_default,
    max_selections=6,
)

min_support = st.slider(
    "Minimum support",
    0.01,
    0.50,
    0.05,
    0.01,
)

rules = association_rules(
    work,
    rule_cols,
    min_support=min_support,
    top_n=100,
)

if rules.empty:

    st.info(
        "No associations passed the selected support threshold."
    )

else:

    st.dataframe(
        rules.style.format(
            {
                "Support": "{:.1%}",
                "Confidence A→B": "{:.1%}",
                "Confidence B→A": "{:.1%}",
                "Lift": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        px.scatter(
            rules,
            x="Support",
            y="Lift",
            size="Count",
            hover_data=[
                c
                for c in [
                    "A",
                    "B",
                    "Confidence A→B",
                    "Confidence B→A",
                ]
                if c in rules.columns
            ],
            title="Association Strength: Support vs Lift",
        ),
        use_container_width=True,
    )

    download_csv(
        "Download association rules",
        rules,
        "association_rules.csv",
        key="rules_dl",
    )


# =========================================================
# 3. PIVOT / CROSS-TAB
# =========================================================

st.subheader("📊 Symbol / Decision Pivot Explorer")

if len(cats) >= 2:

    c1, c2 = st.columns(2)

    with c1:

        row_options = [
            c
            for c in cats
        ]

        row = st.selectbox(
            "Rows",
            row_options,
            index=(
                row_options.index("Symbol")
                if "Symbol" in row_options
                else 0
            ),
            key="mine_row",
        )

    with c2:

        col_options = [
            c
            for c in cats
            if c != row
        ]

        default_col_index = (
            col_options.index("Strategy Decision")
            if "Strategy Decision" in col_options
            else 0
        )

        col = st.selectbox(
            "Columns",
            col_options,
            index=default_col_index,
            key="mine_col",
        )

    normalize = st.checkbox(
        "Normalize each row to probability",
        value=True,
    )

    tab = pd.crosstab(
        work[row],
        work[col],
        normalize="index"
        if normalize
        else False,
    )

    st.plotly_chart(
        px.imshow(
            tab,
            text_auto=(
                ".1%"
                if normalize
                else True
            ),
            aspect="auto",
            title=f"{row} × {col}",
        ),
        use_container_width=True,
    )

    if normalize:

        st.dataframe(
            tab.style.format("{:.1%}"),
            use_container_width=True,
        )

    else:

        st.dataframe(
            tab,
            use_container_width=True,
        )

else:

    st.info(
        "At least two categorical fields are required for the pivot explorer."
    )


# =========================================================
# 4. SEGMENT PROFILER
# =========================================================

st.subheader("🧬 Segment Profiler")

if cats:

    segment = st.selectbox(
        "Segment field",
        cats,
        index=(
            cats.index("Symbol")
            if "Symbol" in cats
            else 0
        ),
        key="segment_field",
    )

    default_measures = [
        c
        for c in [
            "Avg Regime Candle Rank",
            "Regime Maturity Score",
            "Candle After Regime Start",
            "Completed Regime Samples",
        ]
        if c in nums
    ]

    chosen_nums = st.multiselect(
        "Numeric measures",
        nums,
        default=default_measures,
    )

    profile = pd.DataFrame()

    if chosen_nums:

        profile = (
            work.groupby(
                segment,
                dropna=False,
            )[chosen_nums]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                ]
            )
            .round(4)
        )

        st.dataframe(
            profile,
            use_container_width=True,
        )

        measure = chosen_nums[0]

        agg = (
            work.groupby(
                segment,
                dropna=False,
            )[measure]
            .mean()
            .reset_index(
                name=f"Mean {measure}"
            )
        )

        st.plotly_chart(
            px.bar(
                agg,
                x=segment,
                y=f"Mean {measure}",
                text_auto=".2f",
                title=f"Average {measure} by {segment}",
            ),
            use_container_width=True,
        )

else:

    st.info(
        "No categorical segment fields are available."
    )


# =========================================================
# 5. CONDITIONAL DECISION MINING
# =========================================================

st.subheader("🎯 Conditional Decision Mining")

condition_fields = [
    c
    for c in [
        "Middle Standard Regime State",
        "Lower Standard Regime State",
        "Higher Standard Regime State",
        "RAS Decision",
        "Timeframe",
        "Symbol",
        "Decision Rank",
    ]
    if c in work.columns
]

outcome_options = [
    c
    for c in [
        "Strategy Decision",
        "Decision Rank",
        "Symbol",
        "Middle Standard Regime State",
        "RAS Decision",
    ]
    if c in work.columns
]

conditional_table = pd.DataFrame()

if condition_fields and outcome_options:

    condition = st.selectbox(
        "Condition on",
        condition_fields,
        key="condition_field",
    )

    valid_outcomes = [
        c
        for c in outcome_options
        if c != condition
    ]

    if valid_outcomes:

        outcome = st.selectbox(
            "Outcome",
            valid_outcomes,
            index=(
                valid_outcomes.index("Strategy Decision")
                if "Strategy Decision" in valid_outcomes
                else 0
            ),
            key="outcome_field",
        )

        conditional_table = pd.crosstab(
            work[condition],
            work[outcome],
            normalize="index",
        )

        conditional_long = (
            conditional_table
            .reset_index()
            .melt(
                id_vars=condition,
                var_name=outcome,
                value_name="Share",
            )
        )

        st.plotly_chart(
            px.bar(
                conditional_long,
                x=condition,
                y="Share",
                color=outcome,
                barmode="stack",
                title=(
                    f"Conditional Distribution of "
                    f"{outcome} Given {condition}"
                ),
            ),
            use_container_width=True,
        )

        st.dataframe(
            conditional_table.style.format("{:.1%}"),
            use_container_width=True,
        )

    else:

        st.info(
            "Choose a different condition field so that an outcome field remains available."
        )

else:

    st.info(
        "No suitable condition/outcome columns are available."
    )


# =========================================================
# 6. AUTOMATIC DATA-MINING INSIGHTS
# =========================================================

st.subheader("🧠 Automatic Mining Insights")

insights: list[str] = []


# ---------------------------------------------------------
# Frequent combination insight
# ---------------------------------------------------------

if not combos.empty:

    top_combo = combos.iloc[0]

    combo_description = " | ".join(
        str(top_combo[c])
        for c in combo_cols
    )

    top_count = int(
        safe_float(top_combo["Count"])
    )

    top_share = safe_float(
        top_combo["Share"]
    )

    insights.append(
        f"The most frequent observed combination is "
        f"**{combo_description}**, appearing in "
        f"**{top_count:,} rows ({top_share:.1%})**."
    )

    if top_share >= 0.20:

        insights.append(
            "This combination has a relatively high concentration in the filtered dataset, "
            "suggesting that it represents a recurring operating state rather than a rare pattern."
        )

    elif top_share < 0.05:

        insights.append(
            "The most common combination still represents a small share of observations, "
            "indicating a relatively diverse set of market situations."
        )


# ---------------------------------------------------------
# Association rule insight
# ---------------------------------------------------------

if not rules.empty:

    strongest_rule = None

    if "Lift" in rules.columns:

        strongest_rule = rules.sort_values(
            "Lift",
            ascending=False,
        ).iloc[0]

        lift = safe_float(
            strongest_rule["Lift"]
        )

        rule_a = str(
            strongest_rule["A"]
        )

        rule_b = str(
            strongest_rule["B"]
        )

        if "Confidence A→B" in rules.columns:

            confidence = safe_float(
                strongest_rule["Confidence A→B"]
            )

        else:

            confidence = 0.0

        insights.append(
            f"The strongest discovered association by lift is "
            f"**{rule_a} → {rule_b}**, with a lift of "
            f"**{lift:.2f}** and approximately "
            f"**{confidence:.1%} confidence**."
        )

        if lift > 2:

            insights.append(
                "The lift is above 2, indicating a substantially stronger "
                "co-occurrence than would be expected from independent occurrence."
            )

        elif lift > 1:

            insights.append(
                "The lift is above 1, indicating positive association between the two mined conditions."
            )

        else:

            insights.append(
                "The leading association does not show strong positive lift, "
                "so the selected conditions may not provide much additional information about each other."
            )


# ---------------------------------------------------------
# Pivot insight
# ---------------------------------------------------------

if not tab.empty:

    try:

        if normalize:

            max_value = tab.max().max()

            min_value = tab.min().min()

            max_cell = tab.stack().idxmax()
            min_cell = tab.stack().idxmin()

            insights.append(
                f"In the **{row} × {col}** relationship, the highest "
                f"conditional share is **{max_value:.1%}** at "
                f"**{max_cell[0]} → {max_cell[1]}**."
            )

            if max_value - min_value >= 0.40:

                insights.append(
                    "The pivot shows substantial distribution differences between groups, "
                    "suggesting that the first field materially changes the composition of the second."
                )

        else:

            highest_cell = tab.stack().idxmax()

            highest_count = safe_float(
                tab.stack().max()
            )

            insights.append(
                f"The largest observed **{row} × {col}** cell is "
                f"**{highest_cell[0]} → {highest_cell[1]}**, "
                f"with {highest_count:,.0f} observations."
            )

    except Exception:
        pass


# ---------------------------------------------------------
# Segment insight
# ---------------------------------------------------------

if (
    not profile.empty
    and chosen_nums
):

    measure = chosen_nums[0]

    try:

        segment_means = (
            work.groupby(
                segment,
                dropna=False,
            )[measure]
            .mean()
            .dropna()
        )

        if len(segment_means) >= 2:

            highest_segment = segment_means.idxmax()
            lowest_segment = segment_means.idxmin()

            highest_value = safe_float(
                segment_means.max()
            )

            lowest_value = safe_float(
                segment_means.min()
            )

            insights.append(
                f"For **{measure}**, **{highest_segment}** has the highest "
                f"average ({highest_value:.2f}), while **{lowest_segment}** "
                f"has the lowest average ({lowest_value:.2f})."
            )

            if lowest_value != 0:

                relative_difference = (
                    abs(highest_value - lowest_value)
                    / max(abs(lowest_value), 1e-9)
                )

                if relative_difference >= 0.50:

                    insights.append(
                        f"The difference between the strongest and weakest segment is "
                        f"approximately {relative_difference:.0%}, indicating meaningful "
                        "heterogeneity across segments."
                    )

    except Exception:
        pass


# ---------------------------------------------------------
# Conditional decision insight
# ---------------------------------------------------------

if not conditional_table.empty:

    try:

        max_share = conditional_table.max().max()

        max_index = conditional_table.stack().idxmax()

        strongest_condition = max_index[0]
        strongest_outcome = max_index[1]

        insights.append(
            f"Under **{condition} = {strongest_condition}**, the most common "
            f"outcome is **{strongest_outcome}** with a conditional share "
            f"of **{max_share:.1%}**."
        )

        if max_share >= 0.70:

            insights.append(
                "This conditional relationship is relatively concentrated and "
                "may represent a useful candidate for further validation."
            )

        elif max_share < 0.40:

            insights.append(
                "The conditional distribution is fairly dispersed, suggesting "
                "that this condition alone does not strongly determine the outcome."
            )

    except Exception:
        pass


# ---------------------------------------------------------
# Overall data-mining interpretation
# ---------------------------------------------------------

if not insights:

    insights.append(
        "Not enough mining output is available to generate automatic insights. "
        "Try selecting more categorical fields or lowering the association support threshold."
    )


for i, insight in enumerate(
    insights,
    start=1,
):

    st.markdown(
        f"**{i}.** {insight}"
    )


# =========================================================
# 7. AUTOMATIC RESEARCH REPORT
# =========================================================

st.subheader("📋 Automatic Data-Mining Research Report")

report_parts: list[str] = []

report_parts.append(
    f"The analysis contains **{len(work):,} observations** "
    f"across **{work['Symbol'].nunique(dropna=True):,} symbols**."
    if "Symbol" in work.columns
    else
    f"The analysis contains **{len(work):,} observations**."
)

if "Strategy Decision" in work.columns:

    report_parts.append(
        f"The dataset contains "
        f"**{work['Strategy Decision'].nunique(dropna=True):,} distinct "
        f"Strategy Decision classes**."
    )

if "Decision Rank" in work.columns:

    report_parts.append(
        f"Decision ranking contains "
        f"**{work['Decision Rank'].nunique(dropna=True):,} distinct ranks**, "
        "allowing the mining process to investigate how symbols and decisions "
        "are distributed across ranking levels."
    )

if not combos.empty:

    top_combo = combos.iloc[0]

    combo_description = " | ".join(
        str(top_combo[c])
        for c in combo_cols
    )

    report_parts.append(
        f"The most frequently repeated multi-field pattern is "
        f"**{combo_description}**, observed "
        f"**{int(safe_float(top_combo['Count'])):,} times**."
    )

if not rules.empty and "Lift" in rules.columns:

    strongest_rule = rules.sort_values(
        "Lift",
        ascending=False,
    ).iloc[0]

    rule_a = str(
        strongest_rule["A"]
    )

    rule_b = str(
        strongest_rule["B"]
    )

    lift = safe_float(
        strongest_rule["Lift"]
    )

    report_parts.append(
        f"Association mining identifies **{rule_a} → {rule_b}** "
        f"as the strongest relationship by lift, with a lift of "
        f"**{lift:.2f}**."
    )

if not conditional_table.empty:

    try:

        max_index = conditional_table.stack().idxmax()
        max_share = conditional_table.stack().max()

        report_parts.append(
            f"Conditional analysis indicates that when "
            f"**{condition} = {max_index[0]}**, the strongest observed "
            f"outcome is **{max_index[1]}**, reaching a conditional share "
            f"of **{max_share:.1%}**."
        )

    except Exception:
        pass

if (
    not profile.empty
    and chosen_nums
):

    measure = chosen_nums[0]

    try:

        means = (
            work.groupby(
                segment,
                dropna=False,
            )[measure]
            .mean()
            .dropna()
        )

        if len(means) >= 2:

            highest_segment = means.idxmax()
            highest_value = means.max()

            lowest_segment = means.idxmin()
            lowest_value = means.min()

            report_parts.append(
                f"Segment profiling shows that **{highest_segment}** has the "
                f"highest average {measure} ({highest_value:.2f}), while "
                f"**{lowest_segment}** has the lowest ({lowest_value:.2f})."
            )

    except Exception:
        pass


# ---------------------------------------------------------
# Overall conclusion
# ---------------------------------------------------------

if not rules.empty and "Lift" in rules.columns:

    max_lift = safe_float(
        rules["Lift"].max()
    )

else:

    max_lift = 0.0


if (
    max_lift >= 2
    and not conditional_table.empty
):

    conclusion = (
        "Overall, the mining results show meaningful recurring relationships "
        "between categorical conditions. The strongest associations and "
        "conditional distributions are suitable candidates for further "
        "chronological validation rather than being treated as causal rules."
    )

elif (
    max_lift > 1
    or not combos.empty
):

    conclusion = (
        "Overall, the dataset contains recurring Symbol, Decision, Ranking, "
        "and Regime structures, but the strength of those relationships varies. "
        "The strongest patterns should be validated on unseen chronological data "
        "before being used as predictive rules."
    )

else:

    conclusion = (
        "Overall, the current mining configuration does not reveal very strong "
        "associations. More observations, alternative categorical combinations, "
        "or a lower support threshold may reveal additional structure."
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
    height=320,
)


# =========================================================
# 8. RESEARCH SUMMARY
# =========================================================

st.subheader("Research Summary")

summary = {
    "Rows": len(work),
    "Symbols": (
        work["Symbol"].nunique(dropna=True)
        if "Symbol" in work.columns
        else None
    ),
    "Strategy Decisions": (
        work["Strategy Decision"].nunique(dropna=True)
        if "Strategy Decision" in work.columns
        else None
    ),
    "Decision Ranks": (
        work["Decision Rank"].nunique(dropna=True)
        if "Decision Rank" in work.columns
        else None
    ),
    "Combination Fields": len(combo_cols),
    "Association Fields": len(rule_cols),
    "Minimum Support": min_support,
    "Association Rules Found": len(rules),
}

summary_df = pd.DataFrame(
    [summary]
)

st.dataframe(
    summary_df.style.format(
        {
            "Minimum Support": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 9. FINAL RESEARCH NOTE
# =========================================================

st.warning(
    "Association, frequency, pivot, and conditional relationships are "
    "descriptive data-mining findings. A strong association does not prove "
    "causation or guarantee future predictive performance. Validate important "
    "patterns using chronological out-of-sample testing."
)