# ForecastGuard Phase 17 — Release audit and deployment packaging

**Current state:** The six-tab dashboard is working locally. The calibrated next-day view forecasts M5 historical day 1942 (2016-05-23); it does not predict live 2026 sales.

## Contents

- `scripts/audit_release.py` — checks existing backtest cost arithmetic, fold reconciliation, uniqueness, historical uncertainty metrics, and both next-day CSV schemas.
- `scripts/build_demo_snapshot.py` — constructs a standalone, **CSV-only Streamlit demo** in a sibling directory. It does not copy raw Walmart M5 data, notebooks, serialized models, or secrets.
- `tests/test_release_audit.py` — seven additional unit tests.

The add-on does **not** replace your current dashboard or models. Extract it into the existing `ForecastGuard/` project root.

## Run the release checks

From the **ForecastGuard root** in PowerShell:

```powershell
python -m pytest -q
python -m scripts.audit_release
```

The audit needs these files in `results/`:

- `backtest_summary.csv`
- `backtest_predictions.csv`
- `interval_backtest_predictions.csv`
- `next_day_forecast.csv`
- `next_day_calibrated_forecast.csv`

Optional CSVs `interval_backtest_summary.csv`, `interval_product_summary.csv`, and `interval_calibration_margins.csv` are copied when present. The dashboard will still display historical interval results using the predictions CSV alone.

An expected cost result is approximately **$87,976 versus $112,278, 21.64% simulated reduction**. Expected historical interval coverage is approximately **87.19% raw versus 91.29% calibrated**. The audit reports the values it calculates; it does not force these expected values.

## Build a local deployment snapshot

Ensure the Phase 16 dashboard is working and `src/reliability_reporting.py` is present. Install its dashboard dependencies if necessary (`streamlit`, `pandas`, `numpy`, `altair`). Then run:

```powershell
python -m scripts.build_demo_snapshot
```

This creates a **separate sibling directory**:

```text
Projects/
├── ForecastGuard/             # original research project unchanged
└── forecastguard-demo-local/  # standalone local dashboard with exported CSVs
    ├── app/
    ├── src/
    ├── results/
    ├── .streamlit/config.toml
    ├── requirements.txt
    ├── README.md
    ├── MANIFEST.json
    └── .gitignore
```

From the new demo directory:

```powershell
cd ..\forecastguard-demo-local
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

Try each of the six tabs. The snapshot is read-only and relies only on exported historical CSVs; `LightGBM` and `joblib` are not needed to **run the dashboard**.

The builder will stop if the destination folder already exists, so it cannot silently overwrite your previous snapshot. Use `--output ..\forecastguard-demo-local-v2` to create another.

## Publishing to GitHub / Community Cloud

**Do not publish the M5-derived CSV exports until you have verified the applicable Kaggle competition/data redistribution terms and any other applicable rights.** Making code public and making underlying data (or derived records) public are distinct decisions. The builder cannot determine licensing permissions.

Only after that review:

```powershell
python -m scripts.build_demo_snapshot --mode public --confirm-data-terms
```

This creates `../forecastguard-demo-public`. The flag records your deliberate choice; it does not grant a license. Review its contents before committing. No GitHub upload happens automatically.

To deploy on Streamlit Community Cloud after publishing a suitable repository:

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Select **Create app** and the published demo repository.
3. Set the app entrypoint to `app/dashboard.py`.
4. Deploy and inspect all six tabs and the logs.

Streamlit Community Cloud supports `requirements.txt` in the repo root and `.streamlit/config.toml` for theme settings. The generated `requirements.txt` records installed top-level library versions to improve reproducibility.

## Important release limitations

- The 21.64% number is a hypothetical cost simulation using $2 surplus and $8 shortage per unit; it is not documented Walmart profit.
- Historical sales can be censored by stockouts, price listing cannot prove physical stock, and the simulation omits inventory on hand and supply chain lead times.
- The eight windows helped guide model development; they are not an untouched prospective evaluation.
- Historical 91.29% calibrated coverage is not a guarantee for future periods or each product.
- Forecast date 2016-05-23 is the first date after the historical M5 training records, **not a live 2026 forecast**.
- A model trained locally should not be deserialized on a public server from untrusted `.joblib` files. The public CSV-only snapshot deliberately avoids this.

## Next step (Phase 18)

After the release audit and local demo snapshot work, finish a polished project README and GitHub repository. Add an architecture diagram and a transparent model card, then publish only after you have reviewed data redistribution permissions.
