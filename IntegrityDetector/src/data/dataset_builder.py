"""dataset_builder.py — assemble the final, reproducible training dataset.

Pipeline:
  1. Load Kaggle datasets        (dataset_loader.load_all)
  2. Load generated dataset      (data/generated/dataset_generated.csv, if present)
  3. Merge
  4. Clean / normalise           (loader already normalises; re-asserted here)
  5. Remove duplicates           (hash-based, across ALL sources)
  6. Balance classes             (downsample majority to minority, capped to target)
  7. Shuffle                     (fixed seed)
  8. Split 80 / 10 / 10          (stratified; dedup-before-split ⇒ no leakage)

Outputs:
  data/final_dataset.csv                 (text,label,source,split)
  data/splits/{train,val,test}.csv

Run:  python -m src.data.dataset_builder
      python scripts/build_dataset.py --no-generated --target 80000
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.config import (BALANCE_CLASSES, FINAL_DATASET, GENERATED_CSV, SEED,
                        SPLITS_DIR, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
                        TARGET_TOTAL)
from src.data.dataset_loader import load_all
from src.data.text_cleaning import normalize_text, text_hash
from src.logging_utils import get_logger

log = get_logger("builder")


def _load_generated() -> pd.DataFrame:
    """Load every generated corpus in data/generated/ (templates + diverse).

    Picks up both the original ``dataset_generated.csv`` and the large
    ``diverse_corpus.csv`` (casual / short / slang / goofy + polished-AI), so a
    single build merges all synthetic augmentation alongside the real datasets.
    """
    from src.config import GENERATED_DIR
    frames = []
    for path in sorted(GENERATED_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["text", "label", "source"])
        except (ValueError, KeyError):
            df = pd.read_csv(path)
            if not {"text", "label"}.issubset(df.columns):
                log.warning("Skipping %s (missing text/label columns)", path.name)
                continue
            if "source" not in df.columns:
                df["source"] = path.stem
            df = df[["text", "label", "source"]]
        df["label"] = df["label"].astype(int)
        log.info("Loaded generated corpus %-22s: %d rows", path.name, len(df))
        frames.append(df)
    if not frames:
        log.info("No generated corpora in %s (run diverse_corpus / dataset_generator)",
                 GENERATED_DIR)
        return pd.DataFrame(columns=["text", "label", "source"])
    return pd.concat(frames, ignore_index=True)


def balance_classes(df: pd.DataFrame, target_total: int | None) -> pd.DataFrame:
    """Downsample the majority class so both classes are equal, capped to target."""
    g0 = df[df["label"] == 0]
    g1 = df[df["label"] == 1]
    per_class = min(len(g0), len(g1))
    if target_total:
        per_class = min(per_class, target_total // 2)
    if per_class == 0:
        log.warning("A class is empty (human=%d ai=%d) — skipping balance", len(g0), len(g1))
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    g0 = g0.sample(n=per_class, random_state=SEED)
    g1 = g1.sample(n=per_class, random_state=SEED)
    out = pd.concat([g0, g1]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log.info("Balanced to %d per class (total %d)", per_class, len(out))
    return out


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'split' column: train/val/test, stratified by label.

    Texts are already deduplicated, so an identical document cannot leak across
    splits. sklearn handles stratification + the fixed seed gives reproducibility.
    """
    from sklearn.model_selection import train_test_split

    idx = df.index.to_numpy()
    y = df["label"].to_numpy()
    train_idx, temp_idx = train_test_split(
        idx, test_size=(SPLIT_VAL + SPLIT_TEST), random_state=SEED, stratify=y)
    rel_test = SPLIT_TEST / (SPLIT_VAL + SPLIT_TEST)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, random_state=SEED, stratify=y[temp_idx])

    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def _log_split_stats(df: pd.DataFrame) -> None:
    for split in ("train", "val", "test"):
        s = df[df["split"] == split]
        log.info("  %-5s n=%-7d human=%-6d ai=%-6d",
                 split, len(s), int((s["label"] == 0).sum()), int((s["label"] == 1).sum()))


def build(use_generated: bool = True, target_total: int | None = TARGET_TOTAL,
          balance: bool = BALANCE_CLASSES) -> pd.DataFrame:
    config.ensure_dirs()
    config.set_seed()

    # 1-3. load + merge
    kaggle = load_all(dedup=False)                      # dedup once, after merge
    parts = [kaggle]
    if use_generated:
        gen = _load_generated()
        if not gen.empty:
            parts.append(gen)
    df = pd.concat(parts, ignore_index=True)
    log.info("Merged total (pre-clean): %d", len(df))

    # 4. clean / normalise (idempotent re-assert)
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]

    # 5. hash-based dedup across everything
    before = len(df)
    df["_h"] = df["text"].map(text_hash)
    df = df.drop_duplicates("_h").drop(columns="_h").reset_index(drop=True)
    log.info("Removed %d duplicates (merge-wide) -> %d unique", before - len(df), len(df))

    # 6. balance + 7. shuffle
    if balance:
        df = balance_classes(df, target_total)
    else:
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 8. split
    df = stratified_split(df)

    # write outputs
    FINAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DATASET, index=False)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df[df["split"] == split][["text", "label", "source"]].to_csv(
            SPLITS_DIR / f"{split}.csv", index=False)

    log.info("Wrote final dataset -> %s (%d rows)", FINAL_DATASET, len(df))
    _log_split_stats(df)
    return df


def _cli():
    p = argparse.ArgumentParser(description="Build the final dataset (reproducible).")
    p.add_argument("--no-generated", action="store_true", help="skip synthetic data")
    p.add_argument("--target", type=int, default=TARGET_TOTAL,
                   help="max balanced size pre-split (0 = no cap)")
    p.add_argument("--no-balance", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    build(use_generated=not a.no_generated,
          target_total=(a.target or None),
          balance=not a.no_balance)
