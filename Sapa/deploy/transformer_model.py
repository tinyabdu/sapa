"""
text_gen/transformer_model.py

A small GPT-style Transformer, built from scratch in PyTorch (attention
mechanism, positional embeddings, and all — nothing imported from
Hugging Face). Character-level: it predicts the next character given
previous characters, then samples repeatedly to generate text.

Honest scope: trained on a modest text/code corpus with a free Colab
GPU, this learns the STYLE and PATTERNS of the training text (syntax
shapes, common phrases, indentation habits) well enough to generate
plausible-looking snippets. It will not reliably generate correct,
runnable code — that requires vastly more data and parameters than is
practical here. Treat it as a creative pattern-completion toy, not a
coding assistant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttentionHead(nn.Module):
    """One head of causal (can't look ahead) self-attention."""

    def __init__(self, embed_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        # causal mask: position i can only attend to positions <= i
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        attn_scores = q @ k.transpose(-2, -1) * (C ** -0.5)  # scaled dot-product
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v = self.value(x)
        return attn_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = embed_dim // n_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One Transformer block: self-attention -> feedforward, each with a
    residual connection and layer norm (pre-norm, like GPT-2)."""

    def __init__(self, embed_dim, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_heads=8, n_layers=6,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate `max_new_tokens` new characters,
        appended to the input `idx`."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
