"""CLI: generate the synthetic dataset -> data/generated/dataset_generated.csv

    python scripts/generate_dataset.py                 # GPT-2 if available
    python scripts/generate_dataset.py --no-gpt2       # templates only (fast)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_generator import _cli, main  # noqa: E402

if __name__ == "__main__":
    a = _cli()
    main(n_human=a.human, n_ai=a.ai, use_gpt2=not a.no_gpt2)
