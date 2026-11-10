# Phase 18 — GitHub publication kit

This is an **overlay**, not a complete copy of the user's local ForecastGuard project. It assumes the earlier modules, notebooks, and Streamlit app already exist locally. It does not create a GitHub repository or deploy an app automatically.

## Included

- `README.md`: publication-quality repository landing page grounded in the user-supplied results.
- `docs/TECHNICAL_REPORT.md`, `docs/MODEL_CARD.md`, `docs/PUBLICATION_CHECKLIST.md`.
- `.gitignore`: conservative default excluding datasets, prediction-row exports, trained models and secrets.
- `requirements-research.example.txt`: example dependencies; replace with a tested, version-locked environment before a reproducibility claim.
- `scripts/check_public_release.py`: tracked-file and release-structure preflight check.
- `public_demo/`: independent **aggregated-results-only** Streamlit companion to show in public without distributing raw M5 rows. It is **not** the same as the original full six-tab dashboard.

## Installation

1. Back up any existing README and `.gitignore`; review the overlay files before merging into your local ForecastGuard root.
2. Merge `README.md`, `.gitignore`, `docs/` and `scripts/check_public_release.py` into your existing project.
3. Run `python -m scripts.check_public_release`. It reports missing requirements and risky tracked paths; it does not establish legal rights to share M5 data.
4. Run your **existing** Phase 17 audit and Pytest checks if not already done.
5. Choose publication route: code-only GitHub project, isolated aggregated public demo, or rights-reviewed full historical demo.
6. Follow `docs/PUBLICATION_CHECKLIST.md` to publish after review.

## Reproducing the public companion app locally

```powershell
cd public_demo
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To deploy this companion from Streamlit Community Cloud, `public_demo` must be the **root of its own repository**, with `requirements.txt` and `streamlit_app.py` at the top level. The companion displays aggregate project results shared by the user; it is not live forecasting and it does not rely on serialized ML models.

## Important distinctions

- The user's current **Phase 17 audit completion is unconfirmed** in this chat. Do not claim it passed without checking the output.
- Kaggle lists the M5 competition data license as "Subject to Competition Rules". Review the relevant terms before publishing M5-derived record-level CSVs or screenshots containing potentially restricted data.
- Public GitHub hosting and Streamlit deployment require actions on the user's GitHub/Streamlit accounts; this zip does neither.
