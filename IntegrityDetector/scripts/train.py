"""CLI: fine-tune the DistilBERT AI detector (CUDA/AMP).

    python scripts/train.py
    python scripts/train.py --epochs 3 --batch 16 --max-train 40000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_ai_detector import _cli, train  # noqa: E402

if __name__ == "__main__":
    a = _cli()
    train(epochs=a.epochs, batch_size=a.batch, lr=a.lr, max_len=a.max_len,
          max_train=a.max_train, max_val=a.max_val)
