"""
Notebook-style script: EDA & Preprocessing (Task 1)
Run this interactively in Jupyter or as a plain Python script.

Usage:
    python notebooks/01_eda_preprocessing.py --zip data/raw/complaint.csv.zip
"""

import sys
import argparse
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import main, load_data, run_eda, filter_data, clean_narratives, save_filtered, PROCESSED

def run(zip_path=None, csv_path=None):
    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 1 — Load Data")
    print("█"*60)
    df_raw = load_data(zip_path=zip_path, csv_path=csv_path)

    # ── Step 2: EDA ───────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 2 — Exploratory Data Analysis")
    print("█"*60)
    run_eda(df_raw)

    # ── Step 3: Filter ────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 3 — Filter to Target Products & Remove Empty Narratives")
    print("█"*60)
    df_filtered = filter_data(df_raw)

    # ── Step 4: Clean ─────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 4 — Clean Narratives")
    print("█"*60)
    df_clean = clean_narratives(df_filtered)

    # ── Step 5: Save ──────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 5 — Save Processed Data")
    print("█"*60)
    out_path = save_filtered(df_clean)

    print("\n🎉 Task 1 complete!")
    print(f"   Output → {out_path}")
    print(f"   Shape  → {df_clean.shape}")
    print(f"\n   Product breakdown:\n{df_clean['product_category'].value_counts().to_string()}")
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: EDA & Preprocessing")
    parser.add_argument("--zip", help="Path to complaint.csv.zip", default=None)
    parser.add_argument("--csv", help="Path to complaint.csv", default=None)
    args = parser.parse_args()
    run(zip_path=args.zip, csv_path=args.csv)
