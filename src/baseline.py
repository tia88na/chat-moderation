"""Classical baseline (Area B): tf-idf + Logistic Regression.

Reports macro-F1 (classes are imbalanced), offensive-class F1, model size and
per-message latency on the shared OffensEval-TR test set (homework R3/R4).
"""
import os
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

from data import load_offenseval, LABEL_NAMES
from preprocess import normalize


def measure_latency(model, samples, n_runs=200):
    """Average single-message prediction latency in ms, over >=100 runs (R4)."""
    pool = (samples * (n_runs // len(samples) + 1))[:n_runs]
    t0 = time.perf_counter()
    for s in pool:
        model.predict([s])
    return (time.perf_counter() - t0) / n_runs * 1000


def main():
    data = load_offenseval()
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    # normalize up front so tf-idf sees clean text (Area B preprocessing)
    X_train = [normalize(t) for t in X_train]
    X_test = [normalize(t) for t in X_test]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=10,
                                   class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    off_f1 = f1_score(y_test, y_pred, pos_label=1)
    latency_ms = measure_latency(model, X_test[:50])

    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/baseline_tfidf.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_mb = os.path.getsize(path) / 1e6

    print("=== Classical baseline (tf-idf + LR) ===")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Offensive F1:    {off_f1:.4f}")
    print(f"Size:            {size_mb:.2f} MB")
    print(f"Latency:         {latency_ms:.3f} ms/msg (CPU — state hardware in report)")
    print()
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()
