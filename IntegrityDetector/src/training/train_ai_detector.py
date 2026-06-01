"""train_ai_detector.py — fine-tune DistilBERT (AI vs human) on the built dataset.

STRICT NVIDIA CUDA + mixed-precision training, per the project spec:
    device = torch.device("cuda")
    assert torch.cuda.is_available()
Uses torch AMP autocast + GradScaler for fp16 training on the RTX 4060 Ti.

Expects data/splits/{train,val}.csv (run scripts/build_dataset.py first).
Saves the fine-tuned model + tokenizer to models/checkpoints/ai_detector/.

Run:  python -m src.training.train_ai_detector
      python scripts/train.py --epochs 3 --batch 16 --max-train 40000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.config import (AI_DETECTOR_BASE_MODEL, AI_DETECTOR_DIR, EPOCHS, EVAL_BATCH,
                        LR, MAX_SEQ_LEN, SEED, SPLITS_DIR, TRAIN_BATCH, USE_AMP,
                        WARMUP_RATIO, WEIGHT_DECAY)
from src.logging_utils import get_logger

log = get_logger("train")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _build_dataset(tokenizer, texts, labels, max_len):
    import torch

    enc = tokenizer(list(texts), truncation=True, max_length=max_len)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": int(labels[i])}

    return _DS()


def _load_split(name: str, max_n: int | None):
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first: python scripts/build_dataset.py")
    df = pd.read_csv(path, usecols=["text", "label"])
    df = df.dropna(subset=["text"])
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=SEED)
    return df["text"].tolist(), df["label"].astype(int).tolist()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _evaluate(model, loader, device, use_amp):
    import torch
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits.float()
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_p.append(p)
            all_y.append(labels.numpy())
    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(epochs=EPOCHS, batch_size=TRAIN_BATCH, lr=LR, max_len=MAX_SEQ_LEN,
          max_train=None, max_val=None, base_model=AI_DETECTOR_BASE_MODEL,
          out_dir: Path = AI_DETECTOR_DIR):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, get_linear_schedule_with_warmup)

    config.ensure_dirs()
    config.set_seed()

    # ---- STRICT CUDA -------------------------------------------------------
    device = config.get_device(require_cuda=True)        # raises if no GPU (unless REQUIRE_CUDA=0)
    print(f"Using device: {device.type}")
    if device.type == "cuda":
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    use_amp = USE_AMP and device.type == "cuda"

    # ---- data --------------------------------------------------------------
    tr_texts, tr_labels = _load_split("train", max_train)
    va_texts, va_labels = _load_split("val", max_val)
    log.info("Train=%d  Val=%d  (max_len=%d, batch=%d, amp=%s)",
             len(tr_texts), len(va_texts), max_len, batch_size, use_amp)

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=str(config.CACHE_DIR))
    collator = DataCollatorWithPadding(tokenizer)
    train_ds = _build_dataset(tokenizer, tr_texts, tr_labels, max_len)
    val_ds = _build_dataset(tokenizer, va_texts, va_labels, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False,
                            collate_fn=collator, num_workers=0, pin_memory=True)

    # ---- model / optim -----------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, cache_dir=str(config.CACHE_DIR)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    from tqdm import tqdm
    best_f1, best_metrics = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(**batch, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{running/step:.4f}")

        metrics = _evaluate(model, val_loader, device, use_amp)
        log.info("epoch %d done in %.0fs | train_loss=%.4f | val=%s",
                 epoch, time.time() - t0, running / max(1, len(train_loader)), metrics)

        if metrics["f1"] > best_f1:
            best_f1, best_metrics = metrics["f1"], metrics
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            log.info("  -> new best (f1=%.4f) saved to %s", best_f1, out_dir)

    log.info("Training complete. Best val metrics: %s", best_metrics)
    return best_metrics


def _cli():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT AI detector (CUDA/AMP).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=TRAIN_BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--max-train", type=int, default=None, help="cap train rows (speed)")
    p.add_argument("--max-val", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    train(epochs=a.epochs, batch_size=a.batch, lr=a.lr, max_len=a.max_len,
          max_train=a.max_train, max_val=a.max_val)
