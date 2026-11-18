# Real-Time Chat Moderation (SWE015 — Use Case 4)

An intelligent application that flags offensive Turkish chat messages **live** on a
CPU-only budget. Each message is labeled (clean / offensive) with a confidence score,
offensive messages are masked, and a side panel logs the response time of every message.

Four models are compared on the **same** OffensEval-TR 2020 test set:

| # | Model | Area |
|---|-------|------|
| 1 | tf-idf + Logistic Regression (classical baseline) | B |
| 2 | Own word2vec + feedforward / 1D-CNN | A |
| 3 | BERTurk fine-tuned for classification | C |
| 4 | Quantized (INT8) BERTurk — runs in the app | D |

Metrics: **macro-F1** (classes are imbalanced) and **latency per message**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 1. sanity-check the data
python src/data.py

# 2. train + evaluate the classical baseline
python src/baseline.py

# 3. launch the app (added later)
# python app.py
```

## Notes
- Every number in the report comes from our own runs (homework rule).
- Model weights are **not** committed — see `.gitignore`; link to Hugging Face
  Hub or Google Drive instead.
- `data/offensive_lexicon.txt` holds the seed words for the rule baseline — fill it in.

## AI usage
AI coding assistant used for scaffolding and debugging — see the AI Usage slide.
