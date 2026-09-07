from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.query_key_value = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.output = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = inputs.shape
        query, key, value = self.query_key_value(inputs).split(channels, dim=2)
        query = query.view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        key = key.view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        value = value.view(batch, steps, self.n_head, channels // self.n_head).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * (key.size(-1) ** -0.5)
        attention = attention.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)
        outputs = attention @ value
        outputs = outputs.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.output(outputs))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm_attention(inputs))
        return inputs + self.mlp(self.norm_mlp(inputs))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, steps = tokens.shape
        if steps > self.config.block_size:
            raise ValueError("input is longer than the model block size")
        positions = torch.arange(steps, device=tokens.device)
        hidden = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        logits = self.lm_head(self.norm(self.blocks(hidden)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probabilities = F.softmax(logits, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probabilities, num_samples=1)), dim=1)
        return tokens

    def checkpoint(self):
        return {"config": asdict(self.config), "model": self.state_dict()}
