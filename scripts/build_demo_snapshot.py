"""Build a self-contained Streamlit dashboard snapshot without raw M5 or joblib.

From ForecastGuard root:
    python -m scripts.build_demo_snapshot

Default mode is LOCAL PREVIEW. Public mode requires explicit data-rights review:
    python -m scripts.build_demo_snapshot --mode public --confirm-data-terms

The builder does NOT git-push or upload anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import shutil

from .audit_release import REQUIRED_FILES, OPTIONAL_FILES, audit_release


CODE_FILES = (
    "app/dashboard.py",
    "app/interval_reliability.py",
    "app/calibrated_recommendations.py",
    "src/reliability_reporting.py",
)

GITIGNORE = """# Public Streamlit snapshot (do not commit secrets or private datasets)
__pycache__/
*.py[cod]
.pytest_cache/
.streamlit/secrets.toml
.env
.env.*
*.joblib
*.pkl
models/
data/
notebooks/
.venv/
venv/
.DS_Store
"""

DEFAULT_THEME = """[theme]
base = \"dark\"
primaryColor = \"#32C8C3\"
backgroundColor = \"#0B1220\"
secondaryBackgroundColor = \"#172337\"
textColor = \"#F1F5F9\"
font = \"sans serif\"
"""


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Install {package} before building the demo snapshot") from exc


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_snapshot(root: Path, output: Path, *, mode: str = "local", confirm_data_terms: bool = False) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve()
    if mode not in ("local", "public"):
        raise ValueError("mode must be local or public")
    if mode == "public" and not confirm_data_terms:
        raise ValueError(
            "Public distribution is blocked until you review the M5 competition/data license "
            "and any applicable rights for derived CSV exports; then use --confirm-data-terms."
        )
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the ForecastGuard project directory")
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}. Choose a new --output path.")

    report = audit_release(root)
    for name in CODE_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required dashboard code missing: {root / name}")

    output.mkdir(parents=True)
    copied = []
    for rel in CODE_FILES:
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, destination)
        copied.append(rel)
    (output / "src" / "__init__.py").touch()
    (output / "app" / "__init__.py").touch()

    for csv in REQUIRED_FILES + OPTIONAL_FILES:
        source = root / "results" / csv
        if source.is_file():
            dest = output / "results" / csv
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(f"results/{csv}")

    theme = root / ".streamlit" / "config.toml"
    target_theme = output / ".streamlit" / "config.toml"
    target_theme.parent.mkdir(parents=True, exist_ok=True)
    if theme.is_file():
        shutil.copy2(theme, target_theme)
    else:
        target_theme.write_text(DEFAULT_THEME, encoding="utf-8")
    copied.append(".streamlit/config.toml")

    # Exact top-level package versions from the environment used to make the snapshot.
    packages = ("streamlit", "pandas", "numpy", "altair")
    requirements = "\n".join(f"{p}=={_version(p)}" for p in packages) + "\n"
    (output / "requirements.txt").write_text(requirements, encoding="utf-8")
    (output / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    b = report["backtest"]
    r = report["reliability"]
    text = f"""# ForecastGuard — Historical Demand Forecasting Demo

> **Historical research demonstration, NOT a live forecasting service.** Forecast day is **2016-05-23** (M5 day 1942). Costs are **simulated**, not proven retailer savings.

## Results from provided historical exports

- Five Walmart M5 food-product series, one California store
- {b['product_day_forecasts']:,} product-day predictions over {b['folds']} expanding windows
- Quantile ML hypothetical inventory cost **${b['ml_cost']:,.0f}** vs same-quantile seasonal baseline **${b['seasonal_cost']:,.0f}**
- Simulated reduction **{b['savings_pct']:.2f}%**, under an assumed $2 surplus / $8 shortage cost per unit
- Historical 90% interval coverage: raw **{r['raw_coverage_pct']:.2f}%**, conformal-calibrated **{r['calibrated_coverage_pct']:.2f}%**

These historical windows informed development; they are not prospective blind evidence. Sales are an imperfect proxy for true demand, since stockouts and inventory are unobserved. The single-day cost simulation omits lead times, inventory on hand, carrying costs, and replenishment dynamics. Conformal coverage is not guaranteed for future nonstationary data.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

All input CSVs are **historical exported predictions**, not private live customer data. The Streamlit application only reads CSVs. The public snapshot intentionally does not contain Walmart raw files or `joblib` model artifacts.

## Methodology

- Pooled LightGBM quantile model for Q80 with product-level lag and calendar features
- Same-weekday historical quantile benchmark
- Expanding-window one-day-ahead simulated historical evaluation
- Separate product-level historical calibration reports and calibrated day-1942 forecasts
- Dashboard surfaces interval width, empirical coverage and inventory cost tradeoffs

## Data and redistribution review

The underlying input was the M5 forecasting competition dataset. **You must verify its license and competition terms before redistributing any CSV derived from it**. This local snapshot builder does not decide redistribution rights. In the repository, link to the official data source and explain how to reproduce results. Do not commit secrets, raw data or serialized models by default.

## Demo files

- `app/dashboard.py` — Streamlit entrypoint
- `app/interval_reliability.py`, `app/calibrated_recommendations.py` — visualization components
- `src/reliability_reporting.py` — reliability metrics
- `results/` — audited historical CSV exports

## Scope

This is a portfolio research prototype. It does not connect to live Walmart systems or generate real-time inventory purchase orders.
"""
    (output / "README.md").write_text(text, encoding="utf-8")

    manifest = {"mode": mode, "source": "local ForecastGuard CSV exports", "files": {}}
    for file in sorted(output.rglob("*")):
        if file.is_file() and file.name != "MANIFEST.json":
            manifest["files"][file.relative_to(output).as_posix()] = _sha256(file)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any((output / p).exists() for p in ("models", "data", "notebooks")):
        raise RuntimeError("Unexpected raw data/model directory copied into snapshot")
    return {"output": str(output), "files": len(manifest["files"]), "mode": mode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "public"), default="local")
    parser.add_argument("--confirm-data-terms", action="store_true",
                        help="Confirm you have reviewed redistribution rights for exported M5-derived CSVs")
    parser.add_argument("--output", type=Path,
                        help="New sibling folder (default: forecastguard-demo-local or forecastguard-demo-public)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root.parent / f"forecastguard-demo-{args.mode}"
    try:
        result = build_snapshot(root, output, mode=args.mode, confirm_data_terms=args.confirm_data_terms)
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as exc:
        print("BUILD BLOCKED:", exc)
        raise SystemExit(1) from exc
    print(f"Snapshot created: {result['output']}")
    print(f"Files: {result['files']}; mode: {result['mode']}")
    print(f"Try: cd \"{result['output']}\" && python -m streamlit run app/dashboard.py")
    print("No remote deployment or git upload was performed.")


if __name__ == "__main__":
    main()
