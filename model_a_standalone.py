"""Standalone Area A script: own word2vec + feedforward network (SGD + backprop).

Self-contained (no datasets/pyarrow). Put it anywhere and run:

    python model_a_standalone.py

Prints macro-F1, offensive-class F1, model size and per-message latency
on the OffensEval-TR test set.
"""
import os
import re
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, classification_report

LABEL_NAMES = ["NOT", "OFF"]
LABEL_RE = re.compile(r"^__label__(NOT|OFF)\s+")
BASE = "https://huggingface.co/datasets/stefan-it/offenseval2020_tr/resolve/main/"

EMB_DIM = 100
HIDDEN = 128
EPOCHS = 20
LR = 0.5          # SGD learning rate
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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


def tokenize(t):
    return normalize(t).split()


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


def message_vector(tokens, w2v):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def main():
    print("Downloading data...")
    X_train_raw, y_train = load_split("train.txt")
    X_test_raw, y_test = load_split("test.txt")
    train_tokens = [tokenize(t) for t in X_train_raw]
    test_tokens = [tokenize(t) for t in X_test_raw]

    print("Training word2vec on training messages...")
    w2v = Word2Vec(sentences=train_tokens, vector_size=EMB_DIM, window=5,
                   min_count=2, workers=4, seed=SEED, epochs=10)

    Xtr = np.vstack([message_vector(t, w2v) for t in train_tokens])
    Xte = np.vstack([message_vector(t, w2v) for t in test_tokens])
    ytr = np.array(y_train)
    yte = np.array(y_test)

    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)

    model = FeedForward(EMB_DIM, HIDDEN, 2)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print("Training feedforward net (SGD + backprop)...")
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(Xte)).argmax(1).numpy()

    def predict_one(text):
        v = message_vector(tokenize(text), w2v)
        with torch.no_grad():
            return model(torch.tensor(v).unsqueeze(0)).argmax(1).item()

    pool = (X_test_raw[:50] * 4)[:200]
    t0 = time.perf_counter()
    for s in pool:
        predict_one(s)
    latency_ms = (time.perf_counter() - t0) / len(pool) * 1000

    w2v.save("word2vec.model")
    torch.save(model.state_dict(), "model_a_ffn.pt")
    size_mb = (os.path.getsize("word2vec.model")
               + os.path.getsize("model_a_ffn.pt")) / 1e6

    print("\n=== Area A (own word2vec + feedforward) ===")
    print(f"Macro-F1:     {f1_score(yte, y_pred, average='macro'):.4f}")
    print(f"Offensive F1: {f1_score(yte, y_pred, pos_label=1):.4f}")
    print(f"Size:         {size_mb:.2f} MB (word2vec + net)")
    print(f"Latency:      {latency_ms:.3f} ms/msg (CPU)")
    print()
    print(classification_report(yte, y_pred, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()
