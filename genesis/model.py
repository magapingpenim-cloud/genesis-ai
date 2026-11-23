"""Compact decoder-only Transformer for the GENESIS research prototype."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)

class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(), nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        z = self.norm1(x)
        a, _ = self.attention(z, z, z, is_causal=True, need_weights=False)
        x = x + self.dropout(a)
        return x + self.ff(self.norm2(x))

class GenesisLanguageModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.position_embedding = nn.Embedding(c.context_length, c.width)
        self.blocks = nn.ModuleList([CausalBlock(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens, targets=None):
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError("Input exceeds configured context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
