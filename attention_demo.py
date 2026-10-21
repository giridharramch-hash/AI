"""
Self-attention demo (for explaining the concept in the report / viva).

    python attention_demo.py            # toy example in pure NumPy (no download)
    python attention_demo.py --real     # real attention maps from the DistilBERT QA model

Scaled dot-product attention:   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv, scale=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # every token -> query, key, value
    d_k = K.shape[-1]
    scores = Q @ K.T                                  # how well token i's query matches token j's key
    if scale:
        scores = scores / np.sqrt(d_k)                # keeps scores from growing with d_k
    weights = softmax(scores)                         # each row sums to 1
    return weights @ V, weights                       # weighted sum of value vectors


def heatmap(w, tokens, title, path):
    fig, ax = plt.subplots(figsize=(max(4, 0.45 * len(tokens)), max(4, 0.45 * len(tokens))))
    ax.imshow(w, cmap="Blues")
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
    ax.set_xlabel("attended-to token (key)")
    ax.set_ylabel("current token (query)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


def toy():
    rng = np.random.default_rng(0)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    d_model, d_k = 8, 4
    X = rng.normal(size=(len(tokens), d_model))                       # toy embeddings
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) for _ in range(3))
    out, w = self_attention(X, Wq, Wk, Wv)
    np.set_printoptions(precision=2, suppress=True)
    print("attention weights (rows = query tokens):\n", w)
    print("row sums:", w.sum(axis=1))
    print("output shape:", out.shape, "(one new vector per token, mixing information from all tokens)")
    # why divide by sqrt(d_k)?  with large d_k the un-scaled softmax becomes almost one-hot
    Q, K = rng.normal(size=(6, 256)), rng.normal(size=(6, 256))     # independent queries and keys
    for scale in (False, True):
        scores = Q @ K.T / (np.sqrt(256) if scale else 1.0)
        print(f"d_k=256, scaled={scale!s:5}: mean of max weight per row = {softmax(scores).max(axis=1).mean():.2f}")
    heatmap(w, tokens, "Toy self-attention (random weights)", "attention_toy.png")


def real(model_name="distilbert-base-cased-distilled-squad"):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, attn_implementation="eager").eval()
    q = "Which company released BERT?"
    c = "BERT is an encoder-only model pretrained with masked language modeling. It was released by Google in 2018."
    enc = tok(q, c, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k in tok.model_input_names}, output_attentions=True)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    for layer in (0, -1):
        w = out.attentions[layer][0].mean(0).numpy()                  # average over the heads
        name = "first" if layer == 0 else "last"
        heatmap(w, tokens, f"DistilBERT {name} layer, head-averaged attention", f"attention_real_{name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    real() if ap.parse_args().real else toy()
