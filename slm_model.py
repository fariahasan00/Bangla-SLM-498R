
import math
from dataclasses import dataclass #decorator class, eliminates writing same codes for classes

import torch 
import torch.nn as nn #stateful (class), holds internal parameters, declared inside __init__. Also imports nn.Module
import torch.nn.functional as F #stateless, contains functions like relu, cross entropy, called dynamically in forward


@dataclass
class SLMConfig:
    vocab_size: int = 8000
    block_size: int = 512 #context window (row size)
    n_layer: int = 8 #transformer layers
    n_head: int = 6 #multihead attention
    n_embd: int = 384 #embedding dimension per token
    dropout: float = 0.0        # keep 0.0 while data-limited; raise if you overfit
    rope_base: float = 10000.0 #PE formula, rotates Q and K based in position, relative position
    tie_embeddings: bool = True #output, input share same weight tensor

    @property #decorator, turns class method into a getter attribute, head_dim acts like a variable
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0 #tests if this is true
        return self.n_embd // self.n_head

#nn.Module -- base class for all neural network components
class RMSNorm(nn.Module): #normalization
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) #torch.ones creates a 1d tensor of 384 (1.0s), nn.Parameter shows its a trainable weight and registers in model.parameters()

#x is the input tensor [32, 512, 384]
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) #keepdim true preserves original no. of dim
        return (x.to(dtype)) * self.weight

#custom function (64, 512, 10000.0,, cpu, torch.float32). torch.arrange creates a 1d tensor of values like range() but in tensors
def build_rope_cache(head_dim: int, max_seq: int, base: float, device, dtype=torch.float32):
    #inv_freq = 1/base^(2i/head_dim) eg. 1.0/10000^[0,2,4,6,8..62]/64
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float() # 512
    freqs = torch.outer(t, inv_freq) #outer multiplies t with inv freq
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


#multihead attention block
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.n_head = cfg.n_head #6
        self.head_dim = cfg.head_dim #64
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False) #context vector concatenated, needs linear 
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape # B = 32 T = 512, C = 384 [b,512, 384]
        q, k, v = self.qkv(x).split(C, dim=2) #[b, 512, 1152] -> [b, 512, 384]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) #.view() --> [B, 512, 384] -> [B, 512, 6, 64] -> [B, 6, 512, 64]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T]) 
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0, #if train --> dropout, if eval no dropout.
            is_causal=True, #prevents looking at future tokens, masked attention
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

#feed forward nn
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

#transformer block
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

#everything assembled
class BanglaSLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__() #initializes parent class (nn.Module)
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  #maps each token id to a 384 dim vector [batch, token, 384]
        self.drop = nn.Dropout(cfg.dropout) 
        #nn.ModuleList container in Pytorch, acts like list. 
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]) #creates 8 transformer blocks and stores them in the list
        self.ln_f = RMSNorm(cfg.n_embd) #final normalization layer 
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) #output projection mapping final embedding to vocab size
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_base, "cpu") #RoPE rotations for all positions [512,32]
        self.register_buffer("rope_cos", cos, persistent=False) #stores tensors as buffers
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights) #weight initialization on every module 
        # scaled init on residual output projections 
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #fills tensor w random numbers bc at init, weights cant start at 0. Sets initial weights
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #all linear layers get a normal 
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

#utility function used to check model size
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters()) #counts total parameters 
        if non_embedding:
            n -= self.tok_emb.weight.numel() #subtracts embedding parameters
        return n

    def forward(self, idx, targets=None, ignore_index: int = -100):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size"
        #takes first T positions of cache and moves to same device as input
        cos = self.rope_cos[:T].to(idx.device) 
        sin = self.rope_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx)) #applies dropout
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x) #logits - raw, unnormalized score outputs

#Loss calc 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), #flatten sequence
                targets.reshape(-1), #flatten targets
                ignore_index=ignore_index, #skip padding tokens
            )
        return logits, loss

    # ---- optimizer with correct weight-decay grouping -------------------
    def configure_optimizers(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cuda"):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: #requires_grad = False
                continue
            (decay if p.dim() >= 2 else no_decay).append(p) #must be 2D/2D+, 1D are biases and normalization layers
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 **({"fused": True} if fused_ok else {}))

    # ---- inference ------------------------------------------------------
    @torch.no_grad() #decorator disabling grad computation
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