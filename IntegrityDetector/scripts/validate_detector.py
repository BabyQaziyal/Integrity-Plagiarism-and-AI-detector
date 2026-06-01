"""validate_detector.py — sanity + metrics check for the AI detector.

Three checks, in order of what actually matters for the reported bug:

  1. QUALITATIVE — a curated battery of short / casual / slang / goofy HUMAN
     inputs (which used to be flagged as AI) plus some obvious AI inputs. Prints
     P(AI) and the end-to-end verdict for each.
  2. TEST-SET METRICS — accuracy / precision / recall / F1 on data/splits/test.csv.
  3. PER-SOURCE ACCURACY — so we can see that *both* formal human essays and
     casual human text score as human, and AI essays still score as AI (i.e. the
     model didn't just learn "short = human, long = AI").

Run:  python scripts/validate_detector.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import SPLITS_DIR  # noqa: E402
from src.detection.ai_detector import predict_proba  # noqa: E402
from src.detection.analyzer import analyze_submission  # noqa: E402

HUMAN_CASUAL = [
    "nigga ahhhhhh",
    "heoo hee haa haa",
    "lol bruh this is so funny i cant even",
    "omg i cant believe she said that to me like wtf bro thats so messed up",
    "asdkfjalskdjf",
    "ahhhhhhhhhh",
    "bruh bruh bruh stoppp",
    "i went to the store yesterday and bought some milk it was kinda expensive lol",
    "today i woke up late and missed the bus again ugh worst morning ever",
    "this restaurant was honestly trash the service was so slow 2/5",
    "does anyone know how to fix my wifi its been down all day smh",
    "um so like i was thinking maybe we could just chill this weekend idk",
    "milk, eggs, bread, call mom, pay rent",
    "haha lol xd yeet",
]
AI_LIKE = [
    "Certainly! Here's a clear and concise overview of the key benefits of time management for everyone.",
    "In today's fast-paced world, artificial intelligence has become increasingly important. "
    "Firstly, it provides numerous benefits. Moreover, it plays a vital role in society. "
    "In conclusion, its importance cannot be overstated.",
    "To improve your productivity, consider the following steps: 1. Define your goal clearly. "
    "2. Break the task into smaller steps. 3. Monitor your progress regularly.",
]


def _verdict(text):
    r = analyze_submission(text, run_classifier=True, run_perplexity=False,
                           run_plagiarism=False, run_stylometry=False)
    ai = r["ai_content"]
    return ai.get("ai_percent"), ai.get("low_confidence"), r["verdict"]


def qualitative():
    print("\n=== QUALITATIVE (human casual/short/goofy — want LOW AI / abstain) ===")
    for t in HUMAN_CASUAL:
        p = predict_proba(t)
        pct, low, v = _verdict(t)
        print(f"  p_ai={str(p.get('p_ai')):>6}  ai%={str(pct):>6}  low_conf={str(low):>5}  "
              f"verdict={v:<28} | {t[:48]}")
    print("\n=== QUALITATIVE (AI-like — want HIGH AI) ===")
    for t in AI_LIKE:
        p = predict_proba(t)
        pct, low, v = _verdict(t)
        print(f"  p_ai={str(p.get('p_ai')):>6}  ai%={str(pct):>6}  low_conf={str(low):>5}  "
              f"verdict={v:<28} | {t[:48]}")


def test_metrics(max_n=8000):
    path = SPLITS_DIR / "test.csv"
    if not path.exists():
        print("\n(no test split — skipping metrics)")
        return
    df = pd.read_csv(path).dropna(subset=["text"])
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=42)
    print(f"\n=== TEST-SET METRICS (n={len(df)}) ===")
    preds, probs = [], []
    for t in df["text"].tolist():
        r = predict_proba(t)
        p = r.get("p_ai")
        p = 0.5 if p is None else p
        probs.append(p)
        preds.append(int(p >= 0.5))
    y = df["label"].astype(int).to_numpy()
    pred = np.array(preds)
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    print(f"  accuracy ={accuracy_score(y, pred):.4f}")
    print(f"  precision={precision_score(y, pred, zero_division=0):.4f}")
    print(f"  recall   ={recall_score(y, pred, zero_division=0):.4f}")
    print(f"  f1       ={f1_score(y, pred, zero_division=0):.4f}")

    df = df.assign(pred=pred, correct=(pred == y))
    print("\n=== PER-SOURCE ACCURACY (label / acc / n) ===")
    g = (df.groupby("source")
           .agg(label=("label", "first"), acc=("correct", "mean"), n=("correct", "size"))
           .sort_values("n", ascending=False).head(25))
    for src, row in g.iterrows():
        cls = "human" if row["label"] == 0 else "ai"
        print(f"  {src:<26} {cls:<6} acc={row['acc']:.3f}  n={int(row['n'])}")


if __name__ == "__main__":
    qualitative()
    test_metrics()
