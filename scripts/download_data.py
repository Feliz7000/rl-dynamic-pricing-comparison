"""Attempt to download the real Kaggle "Retail Price Optimization" dataset.

Falls back to printing manual instructions if the Kaggle CLI/credentials
aren't available (e.g. no internet access in this environment). This never
blocks the rest of the pipeline -- scripts/generate_synthetic_data.py
produces a schema-identical dataset that everything downstream is built and
tested against.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET = "suddharshan/retail-price-optimization"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

MANUAL_INSTRUCTIONS = f"""
Could not download the dataset automatically (no Kaggle CLI/credentials
found, or no internet access in this environment).

To use the real dataset instead of the synthetic fallback:
  1. On a machine with internet access, log into https://www.kaggle.com
  2. Go to: https://www.kaggle.com/datasets/{DATASET}
  3. Download the dataset zip and extract it.
  4. Copy the resulting CSV to:
       {RAW_DIR / 'retail_price.csv'}
  5. Re-run: python scripts/fit_demand_model.py --source raw

Until then, the project runs fully on the synthetic dataset at
data/synthetic/retail_price.csv (see scripts/generate_synthetic_data.py) --
no code changes are needed to switch sources later, since data/data_loader.py
is schema-driven.
"""


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"Downloaded dataset to {RAW_DIR}")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"Kaggle download failed: {exc}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
