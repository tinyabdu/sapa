"""
text_gen/train_transformer.py

Trains the MiniGPT character-level Transformer on a text/code corpus
of your choosing.

DESIGNED FOR GOOGLE COLAB:
1. Upload this file + transformer_model.py, and your training text file
   (e.g. corpus.txt) to Colab
2. Runtime -> Change runtime type -> GPU (T4, free tier)
3. Run: !python train_transformer.py
4. Checkpoint saves to checkpoints/model_final.pth - download that and
   the vocab.json it produces, both needed for deployment on Railway.

GETTING TRAINING DATA:
- For CODE: clone a few permissively-licensed small repos in your
  language of choice and concatenate their source files into one
  corpus.txt, OR use a Kaggle "code dataset" (e.g. Python code corpus).
- For TEXT: any large plain-text file works (books, articles, your own
  writing samples for a personalized style).
- Aim for at least 1-5MB of text for anything resembling coherent
  output; more is better. A few KB will just memorize and overfit.
"""

import os
import json
import torch
import torch.optim as optim

from transformer_model import MiniGPT

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CORPUS_PATH = "corpus.txt"     # <-- point this at your training text/code file
BLOCK_SIZE = 256                # how many characters of context the model sees
BATCH_SIZE = 64
EMBED_DIM = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------
# Data + tokenizer (simple character-level - each unique character is one token)
# ---------------------------------------------------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

with open("vocab.json", "w") as f:
    json.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f)

print(f"Corpus length: {len(text):,} characters, vocab size: {vocab_size}")

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------
# Model + training loop
# ---------------------------------------------------------------------
model = MiniGPT(
    vocab_size=vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS,
    n_layers=N_LAYERS, block_size=BLOCK_SIZE, dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.2f}M")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for it in range(MAX_ITERS):
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), "checkpoints/model_latest.pth")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "checkpoints/model_final.pth")
print("\nTraining complete. Deploy checkpoints/model_final.pth + vocab.json to Railway.")

# ---------------------------------------------------------------------
# Quick sample after training
# ---------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
generated_text = "".join([itos[i] for i in generated[0].tolist()])
print("\n--- Sample generation ---")
print(generated_text)
