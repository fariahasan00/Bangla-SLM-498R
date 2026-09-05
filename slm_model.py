"""
Bangla SLM — decoder-only transformer (~15M params).

Modern-but-minimal: RMSNorm, RoPE, SwiGLU, Flash attention via SDPA,
tied embeddings. Written to be readable next to nanoGPT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SLMConfig:
    vocab_size: int = 8000
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0        # keep 0.0 while data-limited; raise if you overfit
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(head_dim: int, max_seq: int, base: float, device, dtype=torch.float32):
    """Returns cos/sin of shape (max_seq, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, T, D) with D even. cos/sin: (T, D//2)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)  # round up to multiple of 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)   # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class BanglaSLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_base, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init on residual output projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, ignore_index: int = -100):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size"
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    # ---- optimizer with correct weight-decay grouping -------------------
    def configure_optimizers(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cuda"):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 **({"fused": True} if fused_ok else {}))

    # ---- inference ------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    @torch.no_grad()
    def sequence_logprob(self, ids, prefix_len: int = 0):
        """
        Sum log P(token) over ids[prefix_len:]. Used for perplexity reranking
        in the GEC ensemble. `ids` is a 1-D LongTensor on the model's device.
        """
        self.eval()
        ids = ids[: self.cfg.block_size].unsqueeze(0)
        logits, _ = self(ids)
        logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = ids[:, 1:]
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        start = max(prefix_len - 1, 0)
        scored = tok_lp[start:]
        return scored.sum().item(), scored.numel()


if __name__ == "__main__":
    cfg = SLMConfig(vocab_size=8000)  # Your actual vocab
    m = BanglaSLM(cfg)
    print(f"total={m.num_params()/1e6:.2f}M non-emb={m.num_params(True)/1e6:.2f}M")
    
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = m(x, x)
    print("logits", tuple(logits.shape), "loss", round(loss.item(), 3))
    
    out = m.generate(x[:, :4], max_new_tokens=5)
    print("generate ->", tuple(out.shape))