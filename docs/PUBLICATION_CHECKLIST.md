# Publication and deployment checklist

## Preconditions — do not mark complete until verified

- [ ] Phase 17 `python -m scripts.audit_release` completes with no errors **on your local project**. The audit result has not been supplied in this conversation.
- [ ] `python -m pytest -q` passes on the exact commit you will publish.
- [ ] Local `python -m streamlit run app/dashboard.py` renders all six tabs.
- [ ] Historical metrics match your audit: 21.64% cost reduction, 87.19% raw and 91.29% calibrated coverage.
- [ ] Review the M5 competition rules and any permission needed to redistribute raw data or derived row-level records. **The data page marks license as “Subject to Competition Rules.”**
- [ ] No tokens, `.env` files, `.streamlit/secrets.toml`, local user paths, personal data or accidental notebook output in git history.
- [ ] Decide whether to publish only source + documentation or also *explicitly authorized* data files. `.gitignore` excludes data and exported CSVs by default.
- [ ] Confirm any intended license for your **own code** before adding `LICENSE`. (Dataset licensing is separate.)

## Safety checks from the project root

```powershell
python -m scripts.check_public_release
python -m pytest -q
python -m scripts.audit_release

git status --short
git ls-files
```

`check_public_release` checks **tracked filenames** and some likely local-path strings; it cannot certify that a dataset is licensed for sharing or discover all sensitive content. Inspect `git diff --cached --name-only` and `git diff --cached` before committing.

## GitHub setup

Option A — new research-code repository (recommended until data-terms review):

```powershell
# From ForecastGuard project root. Ensure the Phase 18 files have been copied here.
git init
git branch -M main
git add .gitignore README.md docs src scripts tests notebooks app .streamlit
# If any listed path does not exist in your project, omit it.
git diff --cached --name-only
python -m scripts.check_public_release
git commit -m "Publish ForecastGuard research and documentation"
# Create an EMPTY repo named forecastguard on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/forecastguard.git
git push -u origin main
```

If this is already a Git repository, do **not** reinitialize it, replace remotes, or force-push. Add only reviewed files and commit normally.

**Important:** If `data/`, `models/`, or `results/` were already tracked, `.gitignore` does not untrack them. Before publication, carefully review tracked paths and, if appropriate, use `git rm --cached` on the affected tracked files. Purge leaked secrets from history and rotate credentials before making a repository public.

Option B — a separate, lighter public overview app: copy **only** `public_demo/` into a new repository, keeping its own `requirements.txt` at the repo root. This app shows intentionally limited aggregate results; it is not the full six-tab research dashboard. It contains no M5 sales records or model artifacts.

Option C — full historical Streamlit demo: use Phase 17's `build_demo_snapshot` after rights review. A derived CSV snapshot requires a separate decision about redistribution rights, and must be thoroughly reviewed before committing.

## Streamlit Community Cloud

For the **public_demo companion**, publish `public_demo/` as the root of a separate GitHub repository; select `streamlit_app.py` as entrypoint and the included `requirements.txt`. For a reviewed Phase 17 snapshot, choose `app/dashboard.py` as entrypoint, with snapshot root `requirements.txt`.

1. Go to https://share.streamlit.io and connect your GitHub account.
2. Select Create app, then the appropriate repository, branch and entrypoint.
3. Deploy, open the generated `.streamlit.app` URL, and verify the page loads and all relevant tabs render.
4. Add the **actual tested URL**, not a placeholder, to the README and LinkedIn.
5. Confirm that the deployed app does not expose M5 raw/row-level data without authorization.

## Release note wording

Use: **“In eight historical expanding-window backtests over five selected M5 products, a price-filtered pooled Q80 LightGBM policy reduced hypothetical inventory decision costs by 21.64% versus a seasonal-quantile baseline.”**

Avoid: **“Saved Walmart 21.64% in inventory costs”**, **“90% guaranteed uncertainty”**, or **“real-time forecasting”**.
