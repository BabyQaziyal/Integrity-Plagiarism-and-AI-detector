"""CLI: run the full data pipeline -> data/final_dataset.csv

    python scripts/build_dataset.py
    python scripts/build_dataset.py --no-generated --target 80000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_builder import _cli, build  # noqa: E402

if __name__ == "__main__":
    a = _cli()
    build(use_generated=not a.no_generated,
          target_total=(a.target or None),
          balance=not a.no_balance)
