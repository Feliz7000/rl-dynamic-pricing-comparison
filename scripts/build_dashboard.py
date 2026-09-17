"""Export the dashboard data, build the single-file dashboard, and copy it to
docs/index.html (GitHub Pages: Settings -> Pages -> branch master, folder
/docs). Requires Node 18+.

Usage:
    python scripts/build_dashboard.py            # export + build + copy
    python scripts/build_dashboard.py --skip-export
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
DOCS = ROOT / "docs"


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true", help="reuse dashboard/data/dashboard_data.json")
    parser.add_argument("--data-source", choices=["synthetic", "raw"], default="synthetic")
    args = parser.parse_args()

    if not args.skip_export:
        run([sys.executable, str(ROOT / "scripts" / "export_dashboard_data.py"), "--data-source", args.data_source], cwd=ROOT)

    if not (DASH / "node_modules").exists():
        run(["npm", "install", "--no-audit", "--no-fund"], cwd=DASH)
    run(["npm", "run", "build"], cwd=DASH)

    built = DASH / "dist" / "index.html"
    DOCS.mkdir(exist_ok=True)
    shutil.copyfile(built, DOCS / "index.html")
    print(f"Dashboard written to {DOCS / 'index.html'} ({built.stat().st_size / 1e6:.2f} MB) -- open it directly in a browser.")


if __name__ == "__main__":
    main()
