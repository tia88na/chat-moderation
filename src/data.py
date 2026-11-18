"""OffensEval-TR 2020 data loading for the chat moderation app.

Provides ONE shared train/val/test split that every model is evaluated on,
so the four models in the ladder are comparable (homework R3).

Labels: 0 = NOT (clean), 1 = OFF (offensive).
Source: https://huggingface.co/datasets/stefan-it/offenseval2020_tr
"""
from datasets import load_dataset

LABEL_NAMES = ["NOT", "OFF"]  # 0 = clean, 1 = offensive


def _to_int_label(value):
    """Accept either an int (0/1) or a string ('NOT'/'OFF')."""
    if isinstance(value, int):
        return value
    return LABEL_NAMES.index(value)


def load_offenseval():
    """Return data = {"train": (X, y), "val": (X, y), "test": (X, y)}.

    X: list[str] tweets, y: list[int] labels (0 = NOT, 1 = OFF).
    """
    ds = load_dataset("stefan-it/offenseval2020_tr")

    # this version has train / validation / test; fall back if names differ
    split_map = {"train": "train", "val": "validation", "test": "test"}
    if "validation" not in ds:
        split_map["val"] = "train"  # no dev set -> reuse train for quick checks

    def split_to_xy(hf_split):
        texts = ds[hf_split]["tweet"]
        labels = [_to_int_label(v) for v in ds[hf_split]["subtask_a"]]
        return list(texts), labels

    return {k: split_to_xy(v) for k, v in split_map.items()}


if __name__ == "__main__":
    data = load_offenseval()
    for split, (X, y) in data.items():
        off = sum(y) / len(y) * 100
        print(f"{split:5s}: {len(X)} msgs | {off:.1f}% offensive")
