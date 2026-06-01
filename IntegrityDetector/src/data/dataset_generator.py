"""dataset_generator.py — offline synthetic data augmentation.

Produces ``data/generated/dataset_generated.csv`` with:
  * HUMAN samples (label 0): template Wikipedia / academic-essay / news prose.
  * AI samples (label 1): local **GPT-2** sampling (preferred) with a
    deterministic template fallback when GPT-2/torch is unavailable.

Fully offline + reproducible (fixed seeds). No APIs, no internet at runtime.

Run:  python -m src.data.dataset_generator            (defaults from config)
      python scripts/generate_dataset.py --no-gpt2    (templates only, fast)
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from src import config
from src.config import (GEN_AI_MIN, GEN_HUMAN_MIN, GEN_MAX_NEW_TOKENS,
                        GENERATED_CSV, GPT2_GEN_MODEL, SEED)
from src.data.text_cleaning import normalize_text, text_hash, word_count
from src.logging_utils import get_logger

log = get_logger("generator")

# --------------------------------------------------------------------------- #
# Vocabulary / template banks
# --------------------------------------------------------------------------- #
SUBJECTS = [
    "photosynthesis", "the French Revolution", "machine learning", "climate change",
    "the stock market", "the Roman Empire", "black holes", "the immune system",
    "supply and demand", "renewable energy", "the printing press", "plate tectonics",
    "DNA replication", "the Cold War", "urbanization", "antibiotics", "the internet",
    "democracy", "evolution by natural selection", "vaccination", "the water cycle",
    "Renaissance art", "quantum mechanics", "globalization", "the human brain",
    "volcanic activity", "the Industrial Revolution", "coral reef ecosystems",
    "inflation", "the solar system", "artificial intelligence", "civil rights",
    "ocean currents", "the nervous system", "social media", "nuclear energy",
    "the Great Depression", "genetic engineering", "public health policy",
    "the theory of relativity", "deforestation", "the electoral process",
]
_ADJ   = ["significant", "complex", "fundamental", "controversial", "essential",
          "remarkable", "intricate", "influential", "dynamic", "far-reaching"]
_FIELD = ["science", "history", "economics", "biology", "physics", "sociology",
          "technology", "medicine", "politics", "ecology", "engineering"]

_HUMAN_WIKI = [
    "{T} is a {adj} subject within {field} that has attracted sustained scholarly attention.",
    "The study of {t} dates back many decades and continues to develop as new evidence emerges.",
    "Most researchers agree that {t} plays a {adj} role in shaping outcomes in {field}.",
    "A number of factors influence {t}, among them environmental, social, and economic conditions.",
    "Critics have argued that the conventional account of {t} tends to oversimplify its nuances.",
    "Historically, attitudes toward {t} have shifted considerably with each generation of scholarship.",
    "While {t} is often discussed in isolation, its effects are deeply intertwined with {field}.",
    "Empirical work on {t} suggests that small changes can produce disproportionately large results.",
]
_HUMAN_ACADEMIC = [
    "This essay examines {t} and argues that its importance is frequently underestimated.",
    "Although some scholars downplay {t}, the available evidence points to a more {adj} picture.",
    "To understand {t}, one must first consider the historical context from which it arose.",
    "However, the relationship between {t} and {field} is rarely as straightforward as it appears.",
    "Consequently, any rigorous analysis of {t} must weigh competing explanations carefully.",
    "Taken together, these observations suggest that {t} cannot be reduced to a single cause.",
]
_HUMAN_NEWS = [
    "According to researchers, recent developments in {t} could reshape parts of {field}.",
    "Experts warn that ignoring {t} may carry consequences that are difficult to reverse.",
    "Officials said this week that {t} would remain a priority for the foreseeable future.",
    "Local communities have reported mixed experiences with {t}, the survey found.",
    "Analysts note that the debate over {t} shows little sign of settling soon.",
]
_AI_TEMPLATE = [
    "In today's fast-paced world, {t} has become increasingly important for everyone.",
    "It is important to note that {t} affects many different aspects of our daily lives.",
    "Firstly, {t} provides numerous benefits that simply cannot be ignored.",
    "Secondly, the impact of {t} on society is both significant and far-reaching.",
    "Moreover, understanding {t} is absolutely crucial for future generations.",
    "There are several key factors to consider when we discuss {t} in detail.",
    "Additionally, {t} plays a vital role in promoting growth and development.",
    "Overall, the importance of {t} truly cannot be overstated in any way.",
    "In conclusion, {t} remains a vital topic that deserves our full and undivided attention.",
    "Furthermore, experts agree that {t} will continue to shape the world in meaningful ways.",
]


def _fill(frame: str, topic: str, rng: random.Random) -> str:
    return frame.format(t=topic, T=topic[0].upper() + topic[1:],
                        adj=rng.choice(_ADJ), field=rng.choice(_FIELD))


def _compose(frames: list[str], topic: str, rng: random.Random,
             min_sent: int = 6, max_sent: int = 10) -> str:
    n = rng.randint(min_sent, max_sent)
    picks = [rng.choice(frames) for _ in range(n)]
    return " ".join(_fill(f, topic, rng) for f in picks)


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def generate_human(n: int) -> list[tuple]:
    """Template human prose split across wiki / academic / news styles."""
    styles = [("synthetic_human_wiki", _HUMAN_WIKI),
              ("synthetic_human_academic", _HUMAN_ACADEMIC + _HUMAN_WIKI),
              ("synthetic_human_news", _HUMAN_NEWS + _HUMAN_WIKI)]
    out = []
    for i in range(n):
        rng = random.Random(SEED * 7919 + i)            # deterministic per sample
        source, frames = styles[i % len(styles)]
        topic = SUBJECTS[rng.randrange(len(SUBJECTS))]
        text = normalize_text(_compose(frames, topic, rng))
        if word_count(text) >= 40:
            out.append((text, 0, source))
    log.info("Generated %d human (template) samples", len(out))
    return out


def generate_ai_templates(n: int) -> list[tuple]:
    """Formulaic AI-style prose (classic LLM-essay tells)."""
    out = []
    for i in range(n):
        rng = random.Random(SEED * 104729 + i)
        topic = SUBJECTS[rng.randrange(len(SUBJECTS))]
        text = normalize_text(_compose(_AI_TEMPLATE, topic, rng, 6, 11))
        if word_count(text) >= 40:
            out.append((text, 1, "synthetic_ai_template"))
    log.info("Generated %d AI (template) samples", len(out))
    return out


def generate_ai_gpt2(n: int) -> list[tuple]:
    """Generate AI samples with a local GPT-2 model on the GPU when available."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        log.warning("GPT-2 deps unavailable (%s) — falling back to templates", exc)
        return generate_ai_templates(n)

    try:
        device = config.get_device(require_cuda=False)
        log.info("GPT-2 generation device: %s", device)
        tok = AutoTokenizer.from_pretrained(GPT2_GEN_MODEL, cache_dir=str(config.CACHE_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            GPT2_GEN_MODEL, cache_dir=str(config.CACHE_DIR)).to(device).eval()
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load GPT-2 '%s' (%s) — falling back to templates",
                    GPT2_GEN_MODEL, exc)
        return generate_ai_templates(n)

    prompts_pool = (
        [f"Write a short essay about {t}." for t in SUBJECTS]
        + [f"Discuss the importance of {t}." for t in SUBJECTS]
        + [f"Explain {t} in a few paragraphs." for t in SUBJECTS]
    )
    rng = random.Random(SEED)
    out, batch_size, attempts, max_attempts = [], 16, 0, n * 3
    torch.manual_seed(SEED)

    from tqdm import tqdm
    pbar = tqdm(total=n, desc="GPT-2 AI samples")
    while len(out) < n and attempts < max_attempts:
        batch_prompts = [rng.choice(prompts_pool) for _ in range(batch_size)]
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, top_p=0.95, temperature=0.95,
                max_new_tokens=GEN_MAX_NEW_TOKENS, no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )
        for prompt, ids in zip(batch_prompts, gen):
            decoded = tok.decode(ids, skip_special_tokens=True)
            body = normalize_text(decoded[len(prompt):] if decoded.startswith(prompt) else decoded)
            attempts += 1
            if word_count(body) >= 60:
                out.append((body, 1, "synthetic_ai_gpt2"))
                pbar.update(1)
                if len(out) >= n:
                    break
    pbar.close()
    log.info("Generated %d AI (GPT-2) samples in %d attempts", len(out), attempts)
    if len(out) < n:                                    # top up if generation under-delivered
        out.extend(generate_ai_templates(n - len(out)))
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(n_human: int = GEN_HUMAN_MIN, n_ai: int = GEN_AI_MIN,
         use_gpt2: bool = config.GEN_USE_GPT2) -> pd.DataFrame:
    config.ensure_dirs()
    config.set_seed()
    log.info("Synthetic generation: %d human + %d AI (gpt2=%s)", n_human, n_ai, use_gpt2)

    rows = generate_human(n_human)
    rows += generate_ai_gpt2(n_ai) if use_gpt2 else generate_ai_templates(n_ai)

    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    before = len(df)
    df["_h"] = df["text"].map(text_hash)
    df = df.drop_duplicates("_h").drop(columns="_h")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log.info("Removed %d duplicate synthetic rows", before - len(df))

    GENERATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GENERATED_CSV, index=False)
    log.info("Wrote %d synthetic samples -> %s (human=%d ai=%d)",
             len(df), GENERATED_CSV,
             int((df["label"] == 0).sum()), int((df["label"] == 1).sum()))
    return df


def _cli():
    p = argparse.ArgumentParser(description="Generate synthetic dataset (offline).")
    p.add_argument("--human", type=int, default=GEN_HUMAN_MIN)
    p.add_argument("--ai", type=int, default=GEN_AI_MIN)
    p.add_argument("--no-gpt2", action="store_true", help="use templates only (fast)")
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    main(n_human=a.human, n_ai=a.ai, use_gpt2=not a.no_gpt2)
