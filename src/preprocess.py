"""Area B (classical NLP) preprocessing for chat moderation:

- regex normalization of noisy text (leetspeak, repeated letters, @USER/URL)
- an edit-distance lexicon matcher used as a RULE baseline
"""
import os
import re
from difflib import SequenceMatcher

# --- leetspeak / common character substitutions ---
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
})

USER_URL = re.compile(r"@USER|URL|http\S+|www\.\S+", re.I)
NON_WORD = re.compile(r"[^\wçğıöşü ]+", re.I)


def normalize(text: str) -> str:
    """Lowercase, strip @USER/URL, undo leetspeak, collapse repeated letters."""
    text = text.lower()
    text = USER_URL.sub(" ", text)
    text = text.translate(LEET_MAP)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)   # "çoook" -> "çook"
    text = NON_WORD.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- edit-distance lexicon matcher (rule baseline) ---
LEXICON_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                            "offensive_lexicon.txt")


def load_lexicon(path=LEXICON_PATH):
    """Load offensive seed words, one per line (you fill this file in)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip() and not w.startswith("#")}


def _similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def rule_predict(text, lexicon, threshold=0.85):
    """Flag as offensive (1) if any token closely matches a lexicon word."""
    for token in normalize(text).split():
        for word in lexicon:
            if token == word or _similar(token, word) >= threshold:
                return 1
    return 0


if __name__ == "__main__":
    print("normalized:", normalize("@USER çooook kötüüü bir mesaj http://x.co"))
    lex = load_lexicon()
    print(f"lexicon size: {len(lex)}  (fill data/offensive_lexicon.txt)")
