"""Standalone Area D script: dynamic INT8 quantization of the fine-tuned BERTurk.

Self-contained. Put this file in the SAME folder as your extracted
`berturk-offenseval` model folder, then run:

    python quantize_standalone.py

It prints FP32 vs INT8 size, latency and macro-F1 on the OffensEval-TR test set.

Note: the test set is downloaded directly as a plain text file (no `datasets`
library), to avoid the pyarrow DLL that Windows Application Control blocks.
"""
import os
import re
import time
import urllib.request
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score

MODEL_DIR = "berturk-offenseval"      # the folder you extracted from the zip
LABEL_NAMES = ["NOT", "OFF"]
LABEL_RE = re.compile(r"^__label__(NOT|OFF)\s+")
TEST_URL = ("https://huggingface.co/datasets/stefan-it/offenseval2020_tr/"
            "resolve/main/test.txt")


def load_test():
    """Download test.txt (fastText '__label__XXX message' format) and parse it."""
    req = urllib.request.Request(TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
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


def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs) / 1e6


def evaluate(model, tok, X, y, label):
    model.eval()
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for text in X:
            enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
            preds.append(int(model(**enc).logits.argmax(-1)))
    latency_ms = (time.perf_counter() - t0) / len(X) * 1000
    macro_f1 = f1_score(y, preds, average="macro")
    off_f1 = f1_score(y, preds, pos_label=1)
    print(f"\n=== {label} ===")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Offensive F1: {off_f1:.4f}")
    print(f"Latency:      {latency_ms:.1f} ms/msg (CPU)")
    return macro_f1


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("Downloading test set...")
    X_test, y_test = load_test()
    print(f"Test messages: {len(X_test)}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    fp32_mb = dir_size_mb(MODEL_DIR)
    evaluate(model, tok, X_test, y_test, f"FP32 BERTurk ({fp32_mb:.0f} MB)")

    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(qmodel.state_dict(), "berturk-int8.pt")
    int8_mb = os.path.getsize("berturk-int8.pt") / 1e6
    evaluate(qmodel, tok, X_test, y_test, f"Quantized INT8 BERTurk ({int8_mb:.0f} MB)")

    print("\n--- size comparison ---")
    print(f"FP32: {fp32_mb:.0f} MB  ->  INT8: {int8_mb:.0f} MB "
          f"({fp32_mb / int8_mb:.1f}x smaller)")
    print("Saved quantized weights to berturk-int8.pt")


if __name__ == "__main__":
    main()
