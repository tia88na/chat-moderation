"""Area A (deep learning): own word2vec + feedforward network.

Trains a word2vec model on the training messages (gensim), averages word vectors
per message, and feeds them to a small feedforward net trained with SGD and
backpropagation. Evaluated on the SAME OffensEval-TR test set (R3), reporting
macro-F1, model size and per-message latency (R4).
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, classification_report

from data import load_offenseval, LABEL_NAMES
from preprocess import normalize

EMB_DIM = 100
HIDDEN = 128
EPOCHS = 20
LR = 0.5          # SGD learning rate
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def tokenize(text):
    return normalize(text).split()


def message_vector(tokens, w2v):
    """Average of the word vectors present in the message (zeros if none)."""
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    data = load_offenseval()
    X_train_raw, y_train = data["train"]
    X_test_raw, y_test = data["test"]

    train_tokens = [tokenize(t) for t in X_train_raw]
    test_tokens = [tokenize(t) for t in X_test_raw]

    # --- train our OWN word2vec on the training messages ---
    print("Training word2vec on training messages...")
    w2v = Word2Vec(sentences=train_tokens, vector_size=EMB_DIM, window=5,
                   min_count=2, workers=4, seed=SEED, epochs=10)

    Xtr = np.vstack([message_vector(t, w2v) for t in train_tokens])
    Xte = np.vstack([message_vector(t, w2v) for t in test_tokens])
    ytr = np.array(y_train)
    yte = np.array(y_test)

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    # class weights to handle imbalance
    counts = np.bincount(ytr, minlength=2)
    weights = torch.tensor((counts.sum() / (2 * counts)), dtype=torch.float32)

    model = FeedForward(EMB_DIM, HIDDEN, 2)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print("Training feedforward net (SGD + backprop)...")
    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d}  loss {loss.item():.4f}")

    # --- evaluate on the shared test set ---
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(Xte)).argmax(1).numpy()

    macro_f1 = f1_score(yte, y_pred, average="macro")
    off_f1 = f1_score(yte, y_pred, pos_label=1)

    # per-message latency incl. embedding lookup (R4, >=100 runs)
    def predict_one(text):
        v = message_vector(tokenize(text), w2v)
        with torch.no_grad():
            return model(torch.tensor(v).unsqueeze(0)).argmax(1).item()

    pool = (X_test_raw[:50] * 4)[:200]
    t0 = time.perf_counter()
    for s in pool:
        predict_one(s)
    latency_ms = (time.perf_counter() - t0) / len(pool) * 1000

    os.makedirs("artifacts", exist_ok=True)
    w2v.save("artifacts/word2vec.model")
    torch.save(model.state_dict(), "artifacts/model_a_ffn.pt")
    size_mb = (os.path.getsize("artifacts/word2vec.model")
               + os.path.getsize("artifacts/model_a_ffn.pt")) / 1e6

    print("\n=== Area A (own word2vec + feedforward) ===")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Offensive F1: {off_f1:.4f}")
    print(f"Size:         {size_mb:.2f} MB (word2vec + net)")
    print(f"Latency:      {latency_ms:.3f} ms/msg (CPU — state hardware in report)")
    print()
    print(classification_report(yte, y_pred, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()
