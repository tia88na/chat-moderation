"""Standalone Area B script: tf-idf + Logistic Regression baseline.

Self-contained (no datasets/pyarrow). Put it anywhere and run:

    python baseline_standalone.py

Prints macro-F1, offensive-class F1, model size and per-message latency
on the OffensEval-TR test set.
"""
import os
import re
import time
import pickle
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

LABEL_NAMES = ["NOT", "OFF"]
LABEL_RE = re.compile(r"^__label__(NOT|OFF)\s+")
BASE = "https://huggingface.co/datasets/stefan-it/offenseval2020_tr/resolve/main/"

# --- Area B text normalization ---
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                      "5": "s", "7": "t", "@": "a", "$": "s"})
USER_URL = re.compile(r"@USER|URL|http\S+|www\.\S+", re.I)
NON_WORD = re.compile(r"[^\wçğıöşü ]+", re.I)


def normalize(t):
    t = t.lower()
    t = USER_URL.sub(" ", t)
    t = t.translate(LEET)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = NON_WORD.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_split(fname):
    req = urllib.request.Request(BASE + fname, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        lines = r.read().decode("utf-8").splitlines()
    X, y = [], []
    for line in lines:
        if not line.strip():
            continue
        m = LABEL_RE.match(line)
        y.append(LABEL_NAMES.index(m.group(1)) if m else 0)
        X.append(LABEL_RE.sub("", line, count=1))
    return X, y


def measure_latency(model, samples, n_runs=200):
    pool = (samples * (n_runs // len(samples) + 1))[:n_runs]
    t0 = time.perf_counter()
    for s in pool:
        model.predict([s])
    return (time.perf_counter() - t0) / n_runs * 1000


def main():
    print("Downloading data...")
    X_train, y_train = load_split("train.txt")
    X_test, y_test = load_split("test.txt")
    X_train = [normalize(t) for t in X_train]
    X_test = [normalize(t) for t in X_test]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open("baseline_tfidf.pkl", "wb") as f:
        pickle.dump(model, f)
    size_mb = os.path.getsize("baseline_tfidf.pkl") / 1e6

    print("\n=== Classical baseline (tf-idf + LR) ===")
    print(f"Macro-F1:     {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Offensive F1: {f1_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"Size:         {size_mb:.2f} MB")
    print(f"Latency:      {measure_latency(model, X_test[:50]):.3f} ms/msg (CPU)")
    print()
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()
