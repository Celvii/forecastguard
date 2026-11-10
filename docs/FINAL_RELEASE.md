# ForecastGuard — final release checklist

This checklist is for the **existing** GitHub repository `Celvii/forecastguard`. It does not publish files automatically.

## A. Verify local tests and audit

From the root of the local `ForecastGuard` repository in PowerShell:

```powershell
python -m pytest -q
python -m scripts.audit_release
python -m scripts.check_public_release
git ls-files data models results
```

- Keep the full Phase 17 audit output for your own records. The previous publication preflight succeeded, but that is **not** a substitute for the audit.
- `git ls-files data models results` should remain blank; inspect `git status` and `git diff --cached --name-only` before every push.
- Review prior commits for secrets, and confirm the M5 dataset's terms before sharing any raw or row-level derived data. The public demo is aggregate only.

## B. Update the README and screenshots

Copy `README.md` and `docs/images/` from this kit into the project. Keep the already existing `docs/TECHNICAL_REPORT.md`, `docs/MODEL_CARD.md`, and other project code.

Screenshots here are from the previously provided dashboard images. Only aggregate-level screenshots are included in the public patch. Add per-product examples publicly only after reviewing permission implications.

## C. Test the public demo locally

From the project root:

```powershell
python -m pip install -r public_demo/requirements.txt
python -m streamlit run public_demo/streamlit_app.py
```

Check Backtesting, Uncertainty, and Methodology tabs. The companion is a historical results presentation—not live inference. Confirm that the public demo files are present in Git with `git ls-files public_demo`; if not, stage `public_demo/` as part of the following commit.

## D. Commit the documentation update

```powershell
git add README.md docs/images docs/FINAL_RELEASE.md docs/PORTFOLIO_COPY.md public_demo
# Review the exact staged files before the commit
git diff --cached --name-only
git diff --cached --stat
git commit -m "Polish documentation and add dashboard previews"
git push
```

If the commands say there is nothing to commit, inspect `git status` to check whether the files were copied to the correct location.

## E. Deploy from the same GitHub repository

1. Open https://share.streamlit.io and sign in with GitHub.
2. Choose **Create app** → **Yup, I have an app**.
3. Repository: `Celvii/forecastguard`; branch: `main`.
4. Entry point: `public_demo/streamlit_app.py`.
5. Choose a memorable available subdomain if you wish, then deploy.
6. Open the actual deployed URL and verify the three tabs and the 21.64% aggregate metric.
7. **Only after this succeeds**, replace the `Live research demo` placeholder in the README with the exact URL.
8. Commit and push the README update:

```powershell
git add README.md
git commit -m "Link live ForecastGuard demo"
git push
```

Current Streamlit docs support nested entrypoints and dependency files alongside them:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## F. Finish portfolio presentation

Use `docs/PORTFOLIO_COPY.md` for an accurate LinkedIn project description, resume bullets, and repository description. Avoid implying that the simulated costs are verified retailer savings or that day-1942 forecasting is a live 2026 service.

## Completion criteria

- [ ] Automated tests pass locally
- [ ] Phase 17 release audit passes locally
- [ ] Tracked files and Git history reviewed for data and credentials
- [ ] README screenshots render on GitHub
- [ ] Public three-tab demo opens at an independently tested Streamlit URL
- [ ] README links to that exact live demo
- [ ] GitHub repository description and LinkedIn/resume descriptions updated
