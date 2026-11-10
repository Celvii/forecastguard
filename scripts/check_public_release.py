"""Preflight scan before publishing a public ForecastGuard source repository.

Usage: python -m scripts.check_public_release
This is a conservative structural check, NOT a secrets or copyright certification.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DANGEROUS_PREFIXES = ("data/", "models/", "results/")
DANGEROUS_SUFFIXES = (".csv", ".parquet", ".joblib", ".pkl", ".pickle", ".onnx", ".pem", ".key", ".zip")
DANGEROUS_NAMES = {".env", "secrets.toml"}


def inspect_tracked(tracked):
    findings=[]
    for entry in tracked:
        normalized=entry.replace(chr(92),'/').lstrip('./')
        if normalized.startswith(DANGEROUS_PREFIXES) or normalized.endswith(DANGEROUS_SUFFIXES) or normalized.split('/')[-1] in DANGEROUS_NAMES:
            findings.append(normalized)
    return findings


def git_tracked_files(root):
    try:
        result=subprocess.run(["git","ls-files","-z"],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return [n.decode('utf-8',errors='replace') for n in result.stdout.split(bytes([0])) if n]


def main():
    print("ForecastGuard public release preflight")
    print("Project root:", ROOT)
    required=["README.md", ".gitignore", "docs/MODEL_CARD.md", "docs/TECHNICAL_REPORT.md"]
    missing=[path for path in required if not (ROOT/path).exists()]
    if missing:
        print("FAIL — missing publication files:", ", ".join(missing))
        return 1
    tracked=git_tracked_files(ROOT)
    if tracked is None:
        print("NOTICE — not currently a Git repository; tracked-file check deferred until after git init/add")
    else:
        risky=inspect_tracked(tracked)
        if risky:
            print("FAIL — potentially restricted or sensitive tracked files:")
            for name in risky: print("  -",name)
            print("Do not publish until reviewed and untracked; if already committed, also inspect git history.")
            return 1
        print(f"PASS — no obvious restricted data/model paths in {len(tracked)} tracked filenames")
    print("PASS — publication document structure present")
    print("MANUAL REVIEW REQUIRED — Phase 17 audit, dataset terms, secrets/history, and actual Streamlit deployment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
