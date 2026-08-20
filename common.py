from __future__ import annotations

import io
import re
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_COLUMNS = [
    "Avg Regime Candle Rank", "Symbol", "Strategy Decision", "Lower Regime Bias",
    "Middle Regime Bias", "Higher Standard Regime Bias", "Candle After Regime Start",
    "RAS Decision", "Regime Maturity Score", "Avg Regime Candle Count Before Regime Change",
    "Completed Regime Samples", "Middle Standard Regime State", "Lower Standard Regime State",
    "Higher Standard Regime State", "Completed Candle", "Timeframe", "Datetime",
]

NUMERIC_CANDIDATES = [
    "Avg Regime Candle Rank", "Candle After Regime Start", "Regime Maturity Score",
    "Avg Regime Candle Count Before Regime Change", "Completed Regime Samples", "Completed Candle",
]

CATEGORICAL_CANDIDATES = [
    "Symbol", "Strategy Decision", "Lower Regime Bias", "Middle Regime Bias",
    "Higher Standard Regime Bias", "RAS Decision", "Middle Standard Regime State",
    "Lower Standard Regime State", "Higher Standard Regime State", "Timeframe",
    "Decision Direction", "Decision Rank",
]

TARGET_OPTIONS = [
    "Middle Standard Regime State", "Strategy Decision", "Decision Direction",
    "Lower Standard Regime State", "Higher Standard Regime State", "RAS Decision",
]

@dataclass
class ModelResult:
    name: str
    eval_pipeline: Pipeline
    full_pipeline: Pipeline
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    report: pd.DataFrame
    confusion: pd.DataFrame
    classes: List[str]
    split_index: int


def normalize_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def load_csv(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last = e
    raise last


def load_local_data(path: Path) -> pd.DataFrame:
    """Load a local Finder dataset from CSV or Excel."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        last = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception as e:
                last = e
        raise last
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def _schema_score(frame: pd.DataFrame) -> int:
    cols = {normalize_col(c) for c in frame.columns}
    return len(cols.intersection(set(EXPECTED_COLUMNS)))


def find_local_data(base_dir: Path) -> Optional[Path]:
    """Find the best Finder dataset beside the application.

    CSV/XLSX files are ranked by filename and schema rather than blindly
    preferring Excel. This prevents a stale workbook from overriding the
    current Finder CSV in Streamlit Cloud.
    """
    base_dir = Path(base_dir)
    candidates = [
        p for p in base_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and not p.name.startswith("~$")
        and p.name.lower() != "finder_schema_template.csv"
    ]
    if not candidates:
        return None

    scored = []
    for path in candidates:
        name = path.name.lower()
        score = 0
        if path.suffix.lower() == ".csv":
            score += 4
        if "middle_regime_finder_" in name:
            score += 8
        elif "finder" in name or "regime" in name or "history" in name:
            score += 5
        if "template" in name or "schema" in name:
            score -= 30
        try:
            # Header-only read keeps deployment fast even for 100k+ row files.
            header = pd.read_csv(path, nrows=0) if path.suffix.lower() == ".csv" else pd.read_excel(path, nrows=0)
            schema = _schema_score(header)
        except Exception:
            schema = 0
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        scored.append((score + schema * 2, schema, mtime, path))
    return max(scored, key=lambda x: (x[0], x[1], x[2]))[-1]


@st.cache_data(show_spinner=False)
def load_and_clean_local(path_str: str, mtime_ns: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    path = Path(path_str)
    raw = load_local_data(path)
    return clean_finder(raw)


def _dataset_signature(df: pd.DataFrame, filename: str = "") -> str:
    payload = f"{filename}|{len(df)}|{','.join(map(str, df.columns))}|"
    if not df.empty:
        sample = df.head(2).astype(str).to_csv(index=False) + df.tail(2).astype(str).to_csv(index=False)
        payload += sample
    return sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def save_dataset(df: pd.DataFrame, filename: str):
    clean = df.copy()
    st.session_state["finder_df"] = clean
    st.session_state["finder_filename"] = filename
    st.session_state["finder_signature"] = _dataset_signature(clean, filename)
    st.session_state.pop("engineered_df", None)
    st.session_state.pop("processed_df", None)


def _load_local_into_session(base_dir: Path) -> Optional[pd.DataFrame]:
    path = find_local_data(base_dir)
    if path is None:
        return None
    try:
        signature = (str(path.resolve()), int(path.stat().st_mtime_ns))
    except OSError:
        return None
    prior = st.session_state.get("_auto_loaded_signature")
    if prior == signature and st.session_state.get("finder_df") is not None:
        return st.session_state.get("finder_df").copy()
    raw, missing, extras = load_and_clean_local(signature[0], signature[1])
    if missing or raw.empty:
        return None
    save_dataset(raw, path.name)
    st.session_state["_auto_loaded_signature"] = signature
    st.session_state["_auto_loaded_path"] = str(path)
    st.session_state["_auto_extra_cols"] = extras
    return raw.copy()


def shared_data_uploader(key: str = "shared_dataset_upload", base_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Render a small shared uploader on any page and make its result global to the session."""
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    current = st.session_state.get("finder_df")
    local = _load_local_into_session(base_dir) if current is None or current.empty else current
    with st.sidebar:
        st.subheader("Shared data")
        if local is not None and not local.empty:
            st.success(f"Loaded: {st.session_state.get('finder_filename', 'Finder dataset')}")
            st.caption(f"{len(local):,} rows • {local['Symbol'].nunique() if 'Symbol' in local else 0} symbols")
        upload = st.file_uploader(
            "Upload Finder CSV/XLSX/XLS",
            type=["csv", "xlsx", "xls"],
            key=key,
            help="This uploader is shared across all research pages; you do not need to return to Home.",
        )
    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                raw = load_csv(upload)
            else:
                raw = pd.read_excel(upload)
            df, missing, extras = clean_finder(raw)
            if missing:
                st.sidebar.error("Missing required Finder columns: " + ", ".join(missing[:8]))
            elif df.empty:
                st.sidebar.error("The uploaded Finder dataset has no rows.")
            else:
                sig = _dataset_signature(df, upload.name)
                if sig != st.session_state.get("finder_signature"):
                    save_dataset(df, upload.name)
                st.sidebar.success(f"Using uploaded dataset: {len(df):,} rows")
                if extras:
                    st.sidebar.caption("Extra fields retained: " + ", ".join(extras[:10]))
        except Exception as exc:
            st.sidebar.error(f"Could not load Finder file: {exc}")
    return st.session_state.get("finder_df")


def require_data(base_dir: Optional[Path] = None, uploader_key: Optional[str] = None) -> pd.DataFrame:
    """Return shared data regardless of which Streamlit page was opened first."""
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    df = st.session_state.get("finder_df")
    if df is None or df.empty:
        df = _load_local_into_session(base_dir)
    if uploader_key:
        df = shared_data_uploader(uploader_key, base_dir)
    if df is None or df.empty:
        st.warning("No Finder dataset is loaded.")
        st.info("Use the Shared data uploader in the sidebar, or place the Finder CSV/XLSX beside app.py and reload.")
        st.stop()
    return df.copy()



def direction_from_decision(v) -> str:
    if pd.isna(v):
        return "UNKNOWN"
    text = str(v).upper()
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    if any(x in text for x in ("HOLD", "WAIT", "NONE", "NEUTRAL")):
        return "NEUTRAL"
    return "OTHER"


def rank_from_decision(v) -> str:
    if pd.isna(v):
        return "None"
    m = re.search(r"\bS[1-9]\b", str(v).upper())
    return m.group(0) if m else "None"


def clean_finder(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in out.columns]
    extras = [c for c in out.columns if c not in EXPECTED_COLUMNS]
    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce", utc=True)
    for c in NUMERIC_CANDIDATES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in CATEGORICAL_CANDIDATES:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip().replace({"": pd.NA})
    if "Datetime" in out.columns:
        sort_cols = [c for c in ["Symbol", "Timeframe", "Datetime"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
        out["Date"] = out["Datetime"].dt.date
        out["Hour"] = out["Datetime"].dt.hour
        out["DayOfWeek"] = out["Datetime"].dt.day_name()
        out["DayOfWeekNo"] = out["Datetime"].dt.dayofweek
        out["Month"] = out["Datetime"].dt.month
        out["Week"] = out["Datetime"].dt.isocalendar().week.astype("Int64")
    if "Strategy Decision" in out.columns:
        out["Decision Direction"] = out["Strategy Decision"].map(direction_from_decision)
        out["Decision Rank"] = out["Strategy Decision"].map(rank_from_decision)
        out["Decision Rank Score"] = pd.to_numeric(
            out["Strategy Decision"].astype("string").str.extract(r"\bS([1-9])\b", expand=False),
            errors="coerce",
        )
    return out, missing, extras

def sidebar_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Dataset filters")
        work = df.copy()
        if "Symbol" in work.columns:
            vals = sorted(work["Symbol"].dropna().astype(str).unique())
            picked = st.multiselect("Symbols", vals, default=vals, key=f"{key}_symbols")
            work = work[work["Symbol"].astype(str).isin(picked)]
        if "Timeframe" in work.columns:
            vals = sorted(work["Timeframe"].dropna().astype(str).unique())
            picked = st.multiselect("Timeframes", vals, default=vals, key=f"{key}_tfs")
            work = work[work["Timeframe"].astype(str).isin(picked)]
        if "Datetime" in work.columns and work["Datetime"].notna().any() and not work.empty:
            dmin, dmax = work["Datetime"].min().date(), work["Datetime"].max().date()
            dr = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"{key}_dates")
            if isinstance(dr, tuple) and len(dr) == 2:
                work = work[(work["Datetime"].dt.date >= dr[0]) & (work["Datetime"].dt.date <= dr[1])]
        st.caption(f"Filtered rows: {len(work):,}")
    if work.empty:
        st.warning("No rows match these filters.")
        st.stop()
    return work.copy()


def download_csv(label: str, df: pd.DataFrame, filename: str, key: Optional[str] = None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv", use_container_width=True, key=key)


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 100) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("Datetime", "Date") and df[c].nunique(dropna=True) <= max_unique]


def transition_matrix(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.dropna(subset=[column]).copy()
    groups = [c for c in ["Symbol", "Timeframe"] if c in work.columns]
    work["Next"] = work.groupby(groups, dropna=False)[column].shift(-1) if groups else work[column].shift(-1)
    work = work.dropna(subset=["Next"])
    counts = pd.crosstab(work[column].astype(str), work["Next"].astype(str))
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return counts, probs


def build_feature_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    blocked = {target, "Date", "Datetime"}
    if target == "Strategy Decision": blocked |= {"Decision Direction", "Decision Rank"}
    if target == "Decision Direction": blocked |= {"Strategy Decision", "Decision Rank"}
    cols = []
    for c in df.columns:
        if c in blocked:
            continue
        if c in EXPECTED_COLUMNS or c in (
            "Decision Rank Score",
            "Hour",
            "DayOfWeek",
            "DayOfWeekNo",
            "Month",
            "Week",
        ):
            # Drop fields that are entirely missing in this selected slice;
            # otherwise sklearn imputation can discard them repeatedly.
            if df[c].notna().any():
                cols.append(c)
    return df[cols].copy()


def prepare_features(df: pd.DataFrame, target: str, mode: str = "next") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    work = df.copy()
    if "Datetime" in work.columns:
        work = work.sort_values([c for c in ["Datetime", "Symbol", "Timeframe"] if c in work.columns]).reset_index(drop=True)
    groups = [c for c in ["Symbol", "Timeframe"] if c in work.columns and target != "Symbol"]
    full_X = build_feature_frame(work, target)
    if mode == "next":
        if groups:
            y = work.groupby(groups, dropna=False)[target].shift(-1)
        else:
            y = work[target].shift(-1)
    else:
        y = work[target]
    valid = y.notna()
    return full_X.loc[valid].reset_index(drop=True), y.loc[valid].astype(str).reset_index(drop=True), full_X


def build_pipeline(X: pd.DataFrame, kind: str, rf_estimators: int = 500, rf_depth: Optional[int] = None, min_leaf: int = 2) -> Pipeline:
    nums = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cats = [c for c in X.columns if c not in nums]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)], remainder="drop")
    if kind == "Random Forest":
        model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth, min_samples_leaf=min_leaf, random_state=42, class_weight="balanced", n_jobs=-1)
    elif kind == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=2500, class_weight="balanced")
    return Pipeline([("prep", prep), ("model", model)])


def chronological_model(X: pd.DataFrame, y: pd.Series, kind: str, **kwargs) -> Optional[ModelResult]:
    if len(X) < 40 or y.nunique() < 2:
        return None
    split = min(max(1, int(len(X) * 0.80)), len(X) - 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2 or len(X_test) < 2:
        return None
    eval_pipe = build_pipeline(X, kind, **kwargs)
    eval_pipe.fit(X_train, y_train)
    pred = eval_pipe.predict(X_test)
    labels = sorted(set(y_train.astype(str)) | set(y_test.astype(str)) | set(pred.astype(str)))
    report = pd.DataFrame(classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels)
    full_pipe = clone(eval_pipe)
    full_pipe.fit(X, y)
    return ModelResult(kind, eval_pipe, full_pipe, float(accuracy_score(y_test, pred)), float(balanced_accuracy_score(y_test, pred)), float(f1_score(y_test, pred, average="macro", zero_division=0)), report, cm, labels, split)


def probability_table(model: ModelResult, row: pd.DataFrame) -> pd.DataFrame:
    pred = str(model.full_pipeline.predict(row)[0])
    if hasattr(model.full_pipeline, "predict_proba"):
        probs = model.full_pipeline.predict_proba(row)[0]
        classes = [str(x) for x in model.full_pipeline.classes_]
        out = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values("Probability", ascending=False)
    else:
        out = pd.DataFrame({"Class": [pred], "Probability": [np.nan]})
    out.insert(0, "Predicted", out["Class"].astype(str).eq(pred))
    return out


def feature_importance(model: ModelResult, top_n: int = 30) -> pd.DataFrame:
    prep = model.full_pipeline.named_steps["prep"]
    clf = model.full_pipeline.named_steps["model"]
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(getattr(clf, "n_features_in_", 0))])
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        values = np.mean(np.abs(coef), axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    out = pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False).head(top_n)
    out["Feature"] = out["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    return out


def anomaly_table(df: pd.DataFrame, contamination: float = 0.05) -> Optional[pd.DataFrame]:
    cols = [c for c in numeric_columns(df) if c not in ("Hour", "Month", "Week", "DayOfWeekNo")]
    if len(df) < 25 or len(cols) < 1: return None
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    flag = iso.fit_predict(X)
    out = df.copy()
    out["Anomaly"] = np.where(flag == -1, "ANOMALY", "Normal")
    out["Anomaly Score"] = iso.decision_function(X)
    return out.sort_values("Anomaly Score")


def cluster_table(df: pd.DataFrame, k: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    cols = [c for c in numeric_columns(df) if c not in ("Hour", "Month", "Week", "DayOfWeekNo")]
    if len(df) < max(20, k * 5) or len(cols) < 2: return None
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    Z = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Z)
    pca = PCA(n_components=2, random_state=42).fit_transform(Z)
    out = df.copy()
    out["Pattern Cluster"] = labels.astype(str)
    out["PCA 1"], out["PCA 2"] = pca[:, 0], pca[:, 1]
    centers = out.groupby("Pattern Cluster")[cols].mean(numeric_only=True).reset_index()
    centers.insert(1, "Rows", out.groupby("Pattern Cluster").size().values)
    return out, centers


def nearest_history(df: pd.DataFrame, n: int = 20) -> Optional[pd.DataFrame]:
    cols = [c for c in numeric_columns(df) if c not in ("Hour", "Month", "Week", "DayOfWeekNo")]
    if len(df) < 4 or not cols: return None
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    Z = StandardScaler().fit_transform(X)
    hist_Z = Z[:-1]
    if len(hist_Z) == 0: return None
    nn = NearestNeighbors(n_neighbors=min(n, len(hist_Z))).fit(hist_Z)
    dists, idx = nn.kneighbors(Z[-1:])
    out = df.iloc[idx[0]].copy()
    out.insert(0, "Similarity Distance", dists[0])
    return out.sort_values("Similarity Distance")


def association_rules(df: pd.DataFrame, cols: List[str], min_support: float = 0.05, top_n: int = 50) -> pd.DataFrame:
    if len(cols) < 2 or df.empty:
        return pd.DataFrame()
    n = len(df)
    rules = []
    value_masks = {}
    for c in cols:
        for v, count in df[c].astype("string").value_counts(dropna=True).items():
            value_masks[(c, str(v))] = (df[c].astype("string") == v)
    for a, b in combinations(cols, 2):
        # Keep mining responsive on large histories: use the most frequent
        # categorical values for each field instead of an unbounded Cartesian product.
        a_vals = [k for k in value_masks if k[0] == a]
        b_vals = [k for k in value_masks if k[0] == b]
        a_vals = sorted(a_vals, key=lambda k: int(value_masks[k].sum()), reverse=True)[:30]
        b_vals = sorted(b_vals, key=lambda k: int(value_masks[k].sum()), reverse=True)[:30]
        for ka in a_vals:
            ma = value_masks[ka]
            pa = ma.mean()
            for kb in b_vals:
                mb = value_masks[kb]
                pb = mb.mean()
                joint = (ma & mb).mean()
                if joint < min_support or pa == 0 or pb == 0: continue
                rules.append({
                    "A": f"{a} = {ka[1]}", "B": f"{b} = {kb[1]}",
                    "Support": joint, "Confidence A→B": joint / pa,
                    "Confidence B→A": joint / pb, "Lift": joint / (pa * pb),
                    "Count": int(round(joint * n)),
                })
    if not rules: return pd.DataFrame()
    return pd.DataFrame(rules).sort_values(["Lift", "Support"], ascending=False).head(top_n)


def engineer_data(df: pd.DataFrame, recipe: str) -> pd.DataFrame:
    out = df.copy()
    nums = [c for c in NUMERIC_CANDIDATES if c in out.columns]
    groups = [c for c in ["Symbol", "Timeframe"] if c in out.columns]
    if recipe in ("Clean & robust", "ML-ready", "Sequential forecasting"):
        for c in nums:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
            median = out[c].median()
            out[c] = out[c].fillna(median if pd.notna(median) else 0.0)
            if recipe in ("Clean & robust", "ML-ready") and out[c].notna().sum() > 4:
                lo, hi = out[c].quantile([0.01, 0.99])
                out[c] = out[c].clip(lo, hi)
    if recipe in ("ML-ready", "Sequential forecasting"):
        for c in nums:
            mean = out[c].mean() if out[c].notna().any() else 0.0
            std = out[c].std() if out[c].notna().sum() > 1 else np.nan
            out[f"Z_{c}"] = (out[c] - mean) / std if std and not np.isnan(std) else 0.0
    if recipe == "Sequential forecasting":
        for c in nums:
            if groups:
                out[f"Lag1_{c}"] = out.groupby(groups, dropna=False)[c].shift(1)
                out[f"Rolling3_{c}"] = out.groupby(groups, dropna=False)[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
                out[f"Delta1_{c}"] = out.groupby(groups, dropna=False)[c].diff()
            else:
                out[f"Lag1_{c}"] = out[c].shift(1)
                out[f"Rolling3_{c}"] = out[c].rolling(3, min_periods=1).mean()
                out[f"Delta1_{c}"] = out[c].diff()
        if "Strategy Decision" in out.columns:
            out["Previous Strategy Decision"] = out.groupby(groups, dropna=False)["Strategy Decision"].shift(1) if groups else out["Strategy Decision"].shift(1)
        if "Middle Standard Regime State" in out.columns:
            out["Previous Middle State"] = out.groupby(groups, dropna=False)["Middle Standard Regime State"].shift(1) if groups else out["Middle Standard Regime State"].shift(1)
    return out
