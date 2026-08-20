# Middle Regime Intelligence Studio — Streamlit Deployment Build

## What is fixed

- **Prediction Lab and Pattern Recognition no longer depend on Home being opened first.** Every research page independently resolves the shared Finder dataset and has the same Finder uploader in its sidebar.
- **Local auto-load is schema-aware.** The app no longer blindly prefers the workbook; it correctly prioritizes the current Finder CSV when it contains the expected Finder schema.
- **Symbol-first analysis is real symbol-specific analysis.** Select a symbol and the calculations, distributions, clusters, anomalies, transitions, similar-history results, and forecast input row use that symbol's actual rows.
- **Decision-first analysis is preserved.** Strategy Decision, Decision Direction, rank, and regime state remain available for research and ML targets.
- **Placeholder zero values are removed from the main dashboards.** A metric is shown as `—` when its source field is not available rather than pretending the result is zero.
- **Prediction forecasting uses the latest observation from the selected scope**, rather than a single global row that can make every symbol appear identical.
- **Loss Analysis has been added as a new research page.** Upload a separate loss/trade-history CSV/XLSX, map non-standard columns when necessary, and compare the loss sample against the complete Finder history.
- **Loss analysis includes:** symbol loss-rate comparison, relative-risk lift, decision/regime/bias condition lift, recurring loss-combination mining, loss-reason counts, automatic contextual insights, optional datetime-based Finder context matching, and downloadable enriched results.
- Background Feature Engineering/Data Processing remain hidden from the visible research navigation and are prepared automatically where supported.
- Large association-rule searches are bounded to the most frequent values so deployment remains responsive.

## Deployment

1. Upload this folder to your Streamlit repository.
2. Keep `app.py`, `common.py`, `requirements.txt`, the `pages/` folder, and the Finder CSV/XLSX in the same repository directory.
3. Streamlit entrypoint: `app.py`.
4. The included Finder CSV is auto-detected on startup. Users may also upload another Finder dataset from any visible research page.
5. For loss analysis, open **🧯 Loss Analysis** and upload the separate loss/trade history file.

## Finder dataset requirements

The Finder file should contain these fields:

- Avg Regime Candle Rank
- Symbol
- Strategy Decision
- Lower Regime Bias
- Middle Regime Bias
- Higher Standard Regime Bias
- Candle After Regime Start
- RAS Decision
- Regime Maturity Score
- Avg Regime Candle Count Before Regime Change
- Completed Regime Samples
- Middle Standard Regime State
- Lower Standard Regime State
- Higher Standard Regime State
- Completed Candle
- Timeframe
- Datetime

## Loss file

There is no single mandatory loss-file schema. The Loss Analysis page detects common aliases for:

- Symbol / ticker / instrument / pair
- Datetime / timestamp / open time / entry time
- P&L / profit / loss / result / outcome
- Loss reason / reason / comment / note
- Strategy Decision / decision / signal
- Decision Rank / rank

The page also provides manual column mapping for non-standard files.

## Important note

The model and pattern outputs are statistical research tools. They describe historical structure and conditional relationships; they are not guarantees of future trading outcomes.
