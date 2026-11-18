"""Gradio app: real-time Turkish chat moderation (SWE015 UC4).

Runs the quantized (INT8) BERTurk by default; the user can switch to FP32 to
compare speed. Each message is labeled clean / offensive with a confidence
score, offensive messages are masked, and the live feed logs the per-message
latency of every message.

Run:
    pip install gradio torch transformers
    python app.py
Then open the local URL it prints (http://127.0.0.1:7860).
"""
import re
import time
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "tia88na/berturk-offenseval-tr"
LABELS = ["clean", "offensive"]

# Area B normalization applied before the model, to resist evasions
# (UPPERCASE, leetspeak, stretched letters). Turkish-aware lowercasing.
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                      "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize(text):
    text = text.replace("İ", "i").replace("I", "ı").lower()  # Turkish lower
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)               # çoook -> çook
    text = text.translate(LEET)                              # leetspeak -> letters
    return text

print("Loading model from Hugging Face (first run downloads it)...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
int8 = torch.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
MODELS = {"INT8 (quantized)": int8, "FP32": fp32}

# warm up both models so the FIRST real message isn't slow (one-time kernel setup)
with torch.no_grad():
    for _m in MODELS.values():
        _m(**tok("ısınma", return_tensors="pt"))
print("Ready.")


def classify(message, model_name, log):
    log = log or []
    if message and message.strip():
        model = MODELS[model_name]
        t0 = time.perf_counter()
        enc = tok(normalize(message), truncation=True, max_length=128,
                  return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        latency = (time.perf_counter() - t0) * 1000
        lid = int(probs.argmax())
        conf = float(probs[lid])
        label = LABELS[lid]
        shown = message if label == "clean" else "⟨hidden⟩"
        status = "🟢 clean" if label == "clean" else "🔴 offensive (masked)"
        log = log + [[shown, status, f"{conf:.0%}", f"{latency:.1f}", model_name]]
    return log, log, ""


with gr.Blocks(title="Turkish Chat Moderation") as demo:
    gr.Markdown(
        "# 🛡️ Real-Time Turkish Chat Moderation\n"
        "Type a message and press Enter. Offensive messages are **masked (█)**; "
        "every message's response time (latency) is logged live."
    )
    with gr.Row():
        with gr.Column(scale=1):
            model_sel = gr.Dropdown(list(MODELS), value="INT8 (quantized)",
                                    label="Model (switch to compare speed)")
            msg = gr.Textbox(placeholder="Mesaj yaz ve Enter'a bas...",
                             label="Message", autofocus=True)
            gr.Examples(
                ["merhaba nasılsın", "bu oyun gerçekten çok eğlenceli",
                 "aptal herif", "iyi oynadın tebrikler"],
                inputs=msg, label="Örnek mesajlar")
        with gr.Column(scale=2):
            gr.Markdown("### Live feed + latency log")
            table = gr.Dataframe(
                headers=["Message", "Status", "Conf", "Latency (ms)", "Model"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False, wrap=True)

    log_state = gr.State([])
    msg.submit(classify, [msg, model_sel, log_state], [table, log_state, msg])

if __name__ == "__main__":
    demo.launch()
