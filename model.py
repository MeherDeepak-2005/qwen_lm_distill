"""10.6M-parameter decoder and compact Core ML inference wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 fake quantization with a straight-through estimator."""
    reduce_dims = tuple(range(1, weight.ndim))
    scale = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = (weight / scale).round().clamp(-127, 127) * scale
    return weight + (quantized - weight).detach()


class QLinear(nn.Linear):
    qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant_per_channel(self.weight) if self.qat_enabled else self.weight
        return F.linear(x, weight, self.bias)


class QEmbedding(nn.Embedding):
    qat_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant_per_channel(self.weight) if self.qat_enabled else self.weight
        return F.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 6000
    d_model: int = 384
    layers: int = 6
    heads: int = 6
    ffn_dim: int = 1024
    max_length: int = 32
    dropout: float = 0.05

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: value[k] for k in allowed if k in value})


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        if d % cfg.heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = cfg.heads
        self.head_dim = d // cfg.heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = QLinear(d, 3 * d)
        self.proj = QLinear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = QLinear(d, cfg.ffn_dim)
        self.ff2 = QLinear(cfg.ffn_dim, d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(width, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attention = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = (attention + causal).softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.proj(out))
        ff = self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))
        return x + self.dropout(ff)


class KeyboardDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embedding = QEmbedding(cfg.vocab_size, cfg.d_model)
        self.position = QEmbedding(cfg.max_length, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = QLinear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.register_buffer("position_ids", torch.arange(cfg.max_length), persistent=False)
        self.register_buffer(
            "causal",
            torch.triu(torch.full((cfg.max_length, cfg.max_length), float("-inf")), 1),
            persistent=False,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config.max_length:
            raise ValueError(f"length {length} exceeds {self.config.max_length}")
        x = self.embedding(tokens) + self.position(self.position_ids[:length])
        mask = self.causal[:length, :length]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

    def hidden_fixed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Core ML path: fixed max-length inputs avoid unsupported dynamic slicing."""
        x = self.embedding(tokens) + self.position(self.position_ids)
        for block in self.blocks:
            x = block(x, self.causal)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(tokens))

    def enable_qat(self, enabled: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, (QLinear, QEmbedding)):
                module.qat_enabled = enabled

    def checkpoint_metadata(self) -> dict:
        return {"model_config": asdict(self.config), "parameters": count_parameters(self)}


class CompactInferenceModel(nn.Module):
    score_slots = 8

    def __init__(self, model: KeyboardDecoder):
        super().__init__()
        self.model = model
        self.width = model.config.d_model

    def _gather(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        index = positions.to(torch.int64).unsqueeze(-1).expand(-1, -1, self.width)
        return hidden.gather(1, index)

    def forward(self, tokens, score_positions, target_ids, score_mask, next_positions):
        hidden = self.model.hidden_fixed(tokens)
        selected_hidden = self._gather(hidden, score_positions)
        selected_logits = self.model.head(selected_hidden)
        selected_log_probs = selected_logits.log_softmax(dim=-1)
        selected = selected_log_probs.gather(
            2, target_ids.to(torch.int64).unsqueeze(-1)
        ).squeeze(-1)
        candidate_log_probs = (selected * score_mask).sum(dim=1)
        # Candidate scoring may use a dynamic batch of trie/beam hypotheses, but
        # next-word logits are needed only once for their shared context.
        next_hidden = self._gather(hidden[:1], next_positions[:1])
        next_logits = self.model.head(next_hidden).squeeze(1)
        return candidate_log_probs, next_logits


class NextTokenInferenceModel(nn.Module):
    """Low-latency graph used by trie-constrained beam search."""

    def __init__(self, model: KeyboardDecoder):
        super().__init__()
        self.model = model
        self.width = model.config.d_model

    def forward(self, tokens, next_positions):
        hidden = self.model.hidden_fixed(tokens)
        index = next_positions.to(torch.int64).unsqueeze(-1).expand(-1, -1, self.width)
        selected = hidden.gather(1, index).squeeze(1)
        return self.model.head(selected)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
