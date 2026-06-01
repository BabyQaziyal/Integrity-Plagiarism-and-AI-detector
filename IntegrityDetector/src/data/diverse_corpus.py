"""diverse_corpus.py — large, *diverse* offline corpus generator.

The original detector was trained almost entirely on long, formal essays, so it
had never seen casual, short, slangy, or plain goofy text and resolved all of
that out-of-distribution input toward "AI". This module fixes the data side by
generating a very large, deduplicated corpus that spans **registers** and
**lengths** in BOTH classes, so the model learns that the AI/human boundary is
about *style*, not *length* or *formality*:

  HUMAN (label 0)
    * casual / social / chat / slang          (texting, DMs, comments)
    * short interjections / goofy / gibberish  ("ahhhh", "lol", keyboard mash)
    * reviews & opinions                       (varied sentiment, informal)
    * personal / journal / diary
    * questions / forum posts                  (typos, lowercase, run-ons)
    * spoken / transcript style                (fillers: um, like, you know)
    * notes / lists                            (groceries, todos, fragments)
    * informal essays with human imperfection  (typos, hedging, tangents)

  AI (label 1)
    * polished assistant answers (short→long)  ("Certainly! Here's ...")
    * structured listicles / step-by-steps     (numbered, bulleted, headers)
    * formulaic essay tells                     ("In today's fast-paced world")
    * AI imitating casual tone (still tidy)

Crucially, SHORT and LONG samples appear in *both* classes so length is not a
shortcut feature. Everything is generated from large combinatoric banks with
deterministic per-sample seeding → reproducible, fully offline, and easily
scaled to hundreds of thousands of unique rows.

Run:  python -m src.data.diverse_corpus --human 120000 --ai 120000
      python scripts/generate_dataset.py --diverse        (see scripts)
"""
from __future__ import annotations

import argparse
import random
import string

import pandas as pd

from src import config
from src.config import GENERATED_DIR, SEED
from src.data.text_cleaning import normalize_text, text_hash
from src.logging_utils import get_logger

log = get_logger("diverse")

DIVERSE_CSV = GENERATED_DIR / "diverse_corpus.csv"

# --------------------------------------------------------------------------- #
# Shared word banks (kept big so combinatorics yield lots of unique rows)
# --------------------------------------------------------------------------- #
_PEOPLE = ["my mom", "my dad", "my bro", "my sister", "this guy", "that girl",
           "my teacher", "my boss", "the cashier", "my roommate", "my coach",
           "this dude", "my bestie", "my ex", "my neighbor", "some random guy",
           "the new kid", "my coworker", "grandma", "my landlord", "the waiter"]
_PLACES = ["the mall", "school", "work", "the gym", "the park", "the store",
           "my house", "the beach", "the library", "the cafeteria", "the club",
           "downtown", "the bus stop", "the kitchen", "the office", "class",
           "the parking lot", "the airport", "the dentist", "practice"]
_THINGS = ["the new phone", "this game", "that movie", "the food", "her dress",
           "the homework", "this song", "the weather", "the traffic", "my car",
           "the wifi", "the pizza", "this show", "the test", "the meeting",
           "his haircut", "the concert", "the app", "this book", "the coffee"]
_FEELINGS = ["so tired", "kinda annoyed", "honestly fine", "super happy",
             "low key stressed", "bored out of my mind", "actually excited",
             "kinda sad", "hyped", "exhausted", "chill", "mad", "confused",
             "starving", "sleepy", "pumped", "over it", "vibing", "nervous"]
_SLANG = ["ngl", "fr", "tbh", "lowkey", "highkey", "deadass", "no cap", "frfr",
          "istg", "bruh", "lmaooo", "lol", "smh", "ong", "iykyk", "periodt",
          "literally", "honestly", "bro", "dude", "fam", "vibes", "sheesh"]
_INTENS = ["so", "super", "really", "kinda", "lowkey", "mad", "hella", "very",
           "actually", "literally", "crazy", "wayyy", "soooo", "pretty"]
_REACT = ["that was wild", "i cant even", "im dead", "no way", "im screaming",
          "thats crazy", "im weak", "make it make sense", "the audacity",
          "im obsessed", "im done", "i cant believe it", "this is everything",
          "i'm crying", "best day ever", "worst day ever", "im so confused"]
_EMOJ = ["lol", "lmao", "haha", "hahaha", "xd", "omg", "ugh", "yikes", "oof",
         "yay", "ayo", "sheesh", "bruhhh", "welp", "huh", "eh", "meh", "woo"]
_OPINION_OBJ = ["this restaurant", "the new update", "that phone", "this app",
                "the movie", "this hotel", "the service", "this product",
                "the game", "this place", "the show", "this car", "the album",
                "this laptop", "the burger joint", "this airline", "the gym"]
_OPINION_POS = ["actually really good", "way better than i expected",
                "totally worth it", "pretty solid", "honestly amazing",
                "decent for the price", "surprisingly nice", "great overall"]
_OPINION_NEG = ["a total waste of money", "way overpriced", "super disappointing",
                "not worth it at all", "kinda mid", "honestly trash",
                "a letdown", "worse than the old one", "really frustrating"]
_OPINION_MID = ["okay i guess", "fine but nothing special", "hit or miss",
                "decent but slow", "alright, could be better", "just average"]
_TOPICS_Q = ["python", "my wifi", "this error", "my essay", "the assignment",
             "javascript", "my resume", "excel", "my plant", "this recipe",
             "my pc", "minecraft", "my phone", "the printer", "my budget"]
_FILLERS = ["um", "like", "you know", "i mean", "so basically", "kinda",
            "i guess", "honestly", "right", "and stuff", "or whatever"]

# AI-tells banks
_AI_OPENERS = ["Certainly!", "Of course!", "Absolutely!", "Great question!",
               "Sure thing!", "Happy to help!", "Here's a clear overview:",
               "Let's break this down.", "I'd be glad to explain.",
               "That's an excellent point.", "Here's what you need to know:"]
_AI_CONNECT = ["It's important to note that", "Additionally,", "Furthermore,",
               "Moreover,", "In essence,", "Ultimately,", "As a result,",
               "On the other hand,", "More specifically,", "Notably,",
               "To summarize,", "In conclusion,", "Overall,", "First and foremost,"]
_AI_HEDGE = ["can vary depending on the context", "offers several key benefits",
             "plays a crucial role in modern society", "is a multifaceted topic",
             "has both advantages and disadvantages", "requires careful consideration",
             "is essential for long-term success", "continues to evolve rapidly",
             "depends on a variety of factors", "is widely regarded as significant"]
_AI_TOPICS = ["time management", "renewable energy", "effective communication",
              "machine learning", "a healthy lifestyle", "financial planning",
              "remote work", "climate change", "personal productivity",
              "digital marketing", "sustainable living", "data privacy",
              "leadership", "the benefits of exercise", "learning a new language",
              "artificial intelligence", "mental health", "team collaboration"]
_AI_STEPS = ["Define your goal clearly.", "Gather the necessary resources.",
             "Create a structured plan.", "Break the task into smaller steps.",
             "Monitor your progress regularly.", "Stay consistent and disciplined.",
             "Review and adjust as needed.", "Celebrate small milestones.",
             "Seek feedback from others.", "Reflect on what you've learned.",
             "Prioritize the most important tasks.", "Set a realistic timeline."]


def _rng(i: int, salt: int) -> random.Random:
    return random.Random(SEED * salt + i * 2654435761 % (2 ** 31))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _maybe_typo(text: str, rng: random.Random, p: float = 0.15) -> str:
    """Inject light human-like noise: drop a letter, double one, lowercase."""
    if rng.random() > p:
        return text
    words = text.split()
    if not words:
        return text
    j = rng.randrange(len(words))
    w = words[j]
    if len(w) > 3 and rng.random() < 0.5:
        k = rng.randrange(1, len(w) - 1)
        w = w[:k] + w[k + 1:]                       # drop a char
    elif len(w) > 2:
        k = rng.randrange(len(w))
        w = w[:k] + w[k] + w[k:]                     # double a char
    words[j] = w
    return " ".join(words)


# --------------------------------------------------------------------------- #
# HUMAN generators
# --------------------------------------------------------------------------- #
def _gen_casual(i: int) -> str:
    r = _rng(i, 1001)
    parts = []
    n = r.randint(1, 3)
    for _ in range(n):
        form = r.randrange(6)
        if form == 0:
            parts.append(f"{r.choice(_PEOPLE)} was {r.choice(_INTENS)} {r.choice(_FEELINGS)} at {r.choice(_PLACES)} {r.choice(_SLANG)}")
        elif form == 1:
            parts.append(f"omg {r.choice(_PEOPLE)} said {r.choice(_THINGS)} was {r.choice(_OPINION_NEG)} {r.choice(_REACT)}")
        elif form == 2:
            parts.append(f"i was at {r.choice(_PLACES)} and {r.choice(_THINGS)} was {r.choice(_INTENS)} {r.choice(_FEELINGS)}")
        elif form == 3:
            parts.append(f"{r.choice(_EMOJ)} {r.choice(_REACT)} {r.choice(_SLANG)}")
        elif form == 4:
            parts.append(f"cant believe {r.choice(_PEOPLE)} did that at {r.choice(_PLACES)} {r.choice(_EMOJ)}")
        else:
            parts.append(f"{r.choice(_THINGS)} hits different when youre {r.choice(_FEELINGS)} {r.choice(_SLANG)}")
    return _maybe_typo(" ".join(parts), r, 0.3)


def _gen_goofy(i: int) -> str:
    """Short goofy / interjection / mild-gibberish text — the exact failure cases."""
    r = _rng(i, 1009)
    form = r.randrange(7)
    if form == 0:                                   # stretched interjection
        base = r.choice(["ah", "he", "ha", "lo", "wo", "ye", "no", "ee", "uh", "yo", "br"])
        return (base * r.randint(1, 2)) + r.choice("aeiou") * r.randint(2, 8) + r.choice(["", "h", "hh", "!!"])
    if form == 1:                                   # laughter chains
        return " ".join(r.choice(["haha", "hee", "haa", "lol", "lmao", "hehe", "xd", "heh", "ha"]) for _ in range(r.randint(2, 8)))
    if form == 2:                                   # keyboard mash
        return "".join(r.choice("asdfghjklqwertyuiop") for _ in range(r.randint(4, 14)))
    if form == 3:                                   # goofy phrase
        return f"{r.choice(_EMOJ)} {r.choice(['ahhhhh','heee haa haa','blah blah','yadda yadda','wee woo','beep boop','yeet','oof ouch','goofy ahh'])} {r.choice(_SLANG)}"
    if form == 4:                                   # repeated word
        w = r.choice(["bruh", "no", "stop", "why", "help", "lol", "ok", "what", "huh", "yes"])
        return " ".join([w] * r.randint(2, 6)) + r.choice(["", "!!", "..."])
    if form == 5:                                   # single reaction
        return r.choice(_REACT) + " " + r.choice(_EMOJ)
    return r.choice(_EMOJ) + r.choice("?!.") * r.randint(1, 4)


def _gen_review(i: int) -> str:
    r = _rng(i, 1013)
    obj = r.choice(_OPINION_OBJ)
    bucket = r.choice([_OPINION_POS, _OPINION_NEG, _OPINION_MID])
    s = f"{_cap(obj)} was {r.choice(bucket)}."
    if r.random() < 0.7:
        s += f" The {r.choice(['staff','service','quality','price','vibe','wait time','location','setup'])} was {r.choice(['great','terrible','okay','slow','fast','friendly','rude','fine'])}"
    if r.random() < 0.5:
        s += f" and i'd {r.choice(['definitely','probably','never','maybe'])} {r.choice(['go back','recommend it','buy again','tell my friends'])}"
    if r.random() < 0.5:
        s += f". {r.randint(1, 5)}/5 {r.choice(_EMOJ)}"
    return _maybe_typo(s, r, 0.2)


def _gen_journal(i: int) -> str:
    r = _rng(i, 1019)
    did = r.choice(["woke up late", "missed the bus", "aced my test", "cried a little",
                    "ate way too much", "went for a run", "stayed in bed",
                    f"called {r.choice(_PEOPLE)}", "skipped breakfast", "got lost downtown"])
    parts = [f"today i {did}"]
    if r.random() < 0.8:
        parts.append(f"and it was {r.choice(_INTENS)} {r.choice(_FEELINGS)}")
    if r.random() < 0.6:
        parts.append(f". {r.choice(['honestly','idk','not gonna lie','to be fair'])} {r.choice(_PEOPLE)} {r.choice(['helped a lot','was being weird','made my day','annoyed me','texted me back finally'])}")
    if r.random() < 0.5:
        parts.append(f". gonna {r.choice(['sleep early','try again tomorrow','just relax','study more','call it a night'])} {r.choice(_EMOJ)}")
    return _maybe_typo(" ".join(parts), r, 0.25)


def _gen_question(i: int) -> str:
    r = _rng(i, 1021)
    t = r.choice(_TOPICS_Q)
    forms = [
        f"does anyone know how to fix {t}? its been {r.choice(['broken','acting weird','crashing','so slow'])} all day",
        f"how do i {r.choice(['install','reset','speed up','update','uninstall'])} {t}?? pls help im lost",
        f"is it just me or is {t} {r.choice(['super buggy','really confusing','not working','down rn'])} today",
        f"quick q — whats the best way to {r.choice(['learn','start with','debug','organize'])} {t} for a beginner",
        f"why does {t} keep {r.choice(['freezing','giving errors','restarting','lagging'])}?? been at this for hours smh",
    ]
    return _maybe_typo(r.choice(forms), r, 0.2)


def _gen_spoken(i: int) -> str:
    r = _rng(i, 1031)
    n = r.randint(2, 5)
    chunks = []
    for _ in range(n):
        chunks.append(r.choice(_FILLERS))
        chunks.append(r.choice([
            f"i was thinking we could go to {r.choice(_PLACES)}",
            f"{r.choice(_PEOPLE)} told me about {r.choice(_THINGS)}",
            "it just doesnt really make sense to me",
            f"we should probably {r.choice(['leave early','wait a bit','ask someone','double check'])}",
            "thats not what i meant though",
            f"i dont know if {r.choice(_THINGS)} is worth it",
        ]))
    return " ".join(chunks)


def _gen_notes(i: int) -> str:
    r = _rng(i, 1033)
    items = r.sample([
        "milk", "eggs", "call dentist", "pay rent", "laundry", "bread", "gym",
        "email prof", "buy gift", "water plants", "fix bike", "return books",
        "bananas", "coffee", "vacuum", "text mom", "renew pass", "study ch 4",
        "charger", "batteries", "dog food", "submit form", "book flight"],
        k=r.randint(2, 6))
    sep = r.choice([", ", " - ", "\n", " / ", "; "])
    prefix = r.choice(["", "todo: ", "grocery: ", "remember: ", "list: "])
    return prefix + sep.join(items)


def _gen_short_factual_human(i: int) -> str:
    """Plain, slightly-imperfect short human statements (not formulaic)."""
    r = _rng(i, 1039)
    forms = [
        f"i think {r.choice(_THINGS)} is {r.choice(_OPINION_MID)} but my friends disagree",
        f"we went to {r.choice(_PLACES)} last weekend and it rained the whole time",
        f"my favorite part of {r.choice(_THINGS)} is honestly just the ending",
        f"{_cap(r.choice(_PEOPLE))} keeps telling me to try {r.choice(_THINGS)} but idk",
        f"got home from {r.choice(_PLACES)}, made some food, watched {r.choice(_THINGS)}",
    ]
    return _maybe_typo(r.choice(forms), r, 0.2)


def _gen_informal_essay(i: int) -> str:
    """Longer human writing with imperfections, tangents, hedging, typos."""
    r = _rng(i, 1049)
    topic = r.choice(_AI_TOPICS + [t.replace("the ", "") for t in _THINGS])
    sents = [
        f"so i've been thinking about {topic} a lot lately and honestly it's complicated.",
        f"like, some people swear by it but in my experience it kinda depends?",
        f"when i tried it myself {r.choice(['it went okay','i messed up a bunch','it actually helped','i gave up halfway'])}.",
        "anyway the point is theres no one right answer and thats fine.",
        f"my {r.choice(_PEOPLE)} thinks {r.choice(['im overthinking it','i have a point','its a waste of time','i should keep going'])} lol.",
        "i guess what im trying to say is you just gotta figure out what works for you.",
        f"also side note, {r.choice(_THINGS)} has nothing to do with this but it was on my mind.",
    ]
    r.shuffle(sents)
    text = " ".join(sents[:r.randint(3, 7)])
    return _maybe_typo(text, r, 0.4)


# --------------------------------------------------------------------------- #
# AI generators
# --------------------------------------------------------------------------- #
def _gen_ai_assistant(i: int) -> str:
    r = _rng(i, 2003)
    topic = r.choice(_AI_TOPICS)
    parts = [r.choice(_AI_OPENERS)]
    n = r.randint(1, 4)
    for _ in range(n):
        parts.append(f"{r.choice(_AI_CONNECT)} {topic} {r.choice(_AI_HEDGE)}.")
    if r.random() < 0.6:
        parts.append(f"By focusing on consistency and clear goals, you can achieve meaningful results with {topic}.")
    return " ".join(parts)


def _gen_ai_listicle(i: int) -> str:
    r = _rng(i, 2011)
    topic = r.choice(_AI_TOPICS)
    head = r.choice([f"Here are some effective tips for {topic}:",
                     f"To improve {topic}, consider the following steps:",
                     f"Below are key strategies for {topic}:"])
    steps = r.sample(_AI_STEPS, k=r.randint(3, 6))
    style = r.randrange(3)
    if style == 0:
        body = " ".join(f"{j+1}. {s}" for j, s in enumerate(steps))
    elif style == 1:
        body = " ".join(f"- {s}" for s in steps)
    else:
        body = " ".join(steps)
    tail = r.choice(["", " By following these steps, you'll be well on your way to success.",
                     " These strategies can help you stay on track and achieve your goals."])
    return f"{head} {body}{tail}"


def _gen_ai_formulaic(i: int) -> str:
    r = _rng(i, 2017)
    topic = r.choice(_AI_TOPICS)
    frames = [
        f"In today's fast-paced world, {topic} has become increasingly important for everyone.",
        f"It is important to note that {topic} affects many different aspects of our daily lives.",
        f"Firstly, {topic} provides numerous benefits that simply cannot be ignored.",
        f"Moreover, understanding {topic} is absolutely crucial for future generations.",
        f"Additionally, {topic} plays a vital role in promoting growth and development.",
        f"In conclusion, {topic} remains a vital topic that deserves our full attention.",
    ]
    r.shuffle(frames)
    return " ".join(frames[:r.randint(2, 6)])


def _gen_ai_casual(i: int) -> str:
    """LLM imitating a casual tone — still tidy, grammatical, slightly stiff."""
    r = _rng(i, 2027)
    topic = r.choice(_AI_TOPICS)
    forms = [
        f"Great question! Honestly, {topic} can be a lot of fun once you get the hang of it. The key is to start small and stay consistent.",
        f"Oh, I totally get it! {_cap(topic)} can feel overwhelming at first, but with a little practice it becomes much easier. You've got this!",
        f"That's a really interesting point. When it comes to {topic}, the best approach is usually to keep things simple and build up gradually.",
        f"Absolutely! Many people find {topic} rewarding. Just remember to be patient with yourself and enjoy the process along the way.",
    ]
    return r.choice(forms)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_HUMAN_GENS = [
    ("diverse_casual", _gen_casual, 0.20),
    ("diverse_goofy", _gen_goofy, 0.18),
    ("diverse_review", _gen_review, 0.12),
    ("diverse_journal", _gen_journal, 0.12),
    ("diverse_question", _gen_question, 0.10),
    ("diverse_spoken", _gen_spoken, 0.07),
    ("diverse_notes", _gen_notes, 0.06),
    ("diverse_short_human", _gen_short_factual_human, 0.07),
    ("diverse_informal_essay", _gen_informal_essay, 0.08),
]
_AI_GENS = [
    ("diverse_ai_assistant", _gen_ai_assistant, 0.34),
    ("diverse_ai_listicle", _gen_ai_listicle, 0.26),
    ("diverse_ai_formulaic", _gen_ai_formulaic, 0.22),
    ("diverse_ai_casual", _gen_ai_casual, 0.18),
]


def _run_bank(gens, total: int, label: int) -> list[tuple]:
    rows: list[tuple] = []
    offset = 0
    for source, fn, frac in gens:
        n = max(1, int(total * frac))
        for k in range(n):
            text = normalize_text(fn(offset + k))
            if text:
                rows.append((text, label, source))
        offset += n
    return rows


def generate(n_human: int = 120000, n_ai: int = 120000) -> pd.DataFrame:
    config.ensure_dirs()
    config.set_seed()
    log.info("Diverse generation target: %d human + %d AI", n_human, n_ai)

    rows = _run_bank(_HUMAN_GENS, n_human, 0) + _run_bank(_AI_GENS, n_ai, 1)
    df = pd.DataFrame(rows, columns=["text", "label", "source"])

    before = len(df)
    df["_h"] = df["text"].map(text_hash)
    df = df.drop_duplicates("_h").drop(columns="_h")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log.info("Removed %d duplicate diverse rows -> %d unique", before - len(df), len(df))

    DIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIVERSE_CSV, index=False)
    h = int((df["label"] == 0).sum()); a = int((df["label"] == 1).sum())
    log.info("Wrote %d diverse rows -> %s (human=%d ai=%d)", len(df), DIVERSE_CSV, h, a)
    log.info("By source:\n%s", df["source"].value_counts().to_string())
    # quick length sanity
    wl = df["text"].str.split().str.len()
    log.info("Word length: min=%d median=%d p90=%d max=%d",
             int(wl.min()), int(wl.median()), int(wl.quantile(0.9)), int(wl.max()))
    return df


def _cli():
    p = argparse.ArgumentParser(description="Generate a large, diverse offline corpus.")
    p.add_argument("--human", type=int, default=120000)
    p.add_argument("--ai", type=int, default=120000)
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    generate(n_human=a.human, n_ai=a.ai)
