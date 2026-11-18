# Real-Time Chat Moderation (SWE015 — Use Case 4)

An intelligent application that flags offensive Turkish chat messages **live** on a
CPU-only budget. Each message is labeled (clean / offensive) with a confidence score,
offensive messages are masked, and a side panel logs the response time of every message.

Four models are compared on the **same** OffensEval-TR 2020 test set:

| # | Model | Area |
|---|-------|------|
| 1 | tf-idf + Logistic Regression (classical baseline) | B |
| 2 | Own word2vec + feedforward network | A |
| 3 | BERTurk fine-tuned for classification | C |
| 4 | Quantized (INT8) BERTurk — runs in the app | D |

## Results (our own runs, CPU)

| Model | Macro-F1 | Size | Latency |
|-------|---------|------|---------|
| tf-idf + LR (B) | 0.7246 | 1.77 MB | 0.50 ms/msg |
| word2vec + FFN (A) | 0.5432 | 25.5 MB | 0.27 ms/msg |
| BERTurk FP32 (C) | 0.8120 | 443 MB | 42.0 ms/msg |
| BERTurk INT8 (D) | 0.8056 | 186 MB | 38.9 ms/msg |

Metric: **macro-F1** (classes are imbalanced). BERTurk wins; INT8 quantization keeps
almost the same accuracy at ~2.4x smaller size. The word2vec+FFN model underperforms
the tf-idf baseline — a valid finding discussed in the report.

## Model weights

The fine-tuned BERTurk model is on the Hugging Face Hub (not committed here, see R7):
**https://huggingface.co/tia88na/berturk-offenseval-tr**

## Setup

```bash
pip install torch transformers scikit-learn gensim
```

## Run

```bash
# Area B — classical baseline
python baseline_standalone.py

# Area A — own word2vec + feedforward
python model_a_standalone.py

# Area D — quantize the fine-tuned BERTurk
#   first download the model from the Hugging Face link above into ./berturk-offenseval
python quantize_standalone.py

# Area C — BERTurk fine-tuning: open the notebook in Google Colab (GPU)
#   c_berturk_finetune.ipynb
```

The scripts download the OffensEval-TR test/train splits directly, so no `datasets`
library is required.

## AI usage

AI coding assistant used for scaffolding, debugging and data-loading fixes —
see the AI Usage slide.
