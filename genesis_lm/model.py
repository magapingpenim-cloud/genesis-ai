"""A compact decoder-only Transformer for next-byte prediction.

This is an educational starter model, not a pretrained assistant. It uses UTF-8
bytes as tokens so training does not depend on a downloaded tokenizer vocabulary.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 256

class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, steps, self.heads, width // self.heads)
        q, k, v = [t.view(shape).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                  nn.Linear(4 * width, width), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class GenesisLM(nn.Module):
    def __init__(self, width=192, layers=6, heads=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(VOCAB_SIZE, width)
        self.position_embedding = nn.Embedding(block_size, width)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        batch, steps = tokens.shape
        if steps > self.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.block_size}")
        positions = torch.arange(steps, device=tokens.device)
        x = self.drop(self.token_embedding(tokens) + self.position_embedding(positions))
        logits = self.lm_head(self.norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=200, temperature=0.8, top_k=40):
        if temperature <= 0: raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
