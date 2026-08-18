"""
Pretraining loop for the Bangla SLM. Designed for Colab: assume you WILL be
disconnected, so state is checkpointed to Drive and resume is automatic.

Run:  python train_slm.py
"""

import math
import os
import time

import numpy as np
import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

from slm_model import SLMConfig, BanglaSLM

# ----------------------------- config ---------------------------------
ROOT = "/content/drive/MyDrive/Bangla_SLM_15M"
TRAIN_BIN = "/content/local_data/train.bin"
VAL_BIN = "/content/local_data/val.bin"
SP_MODEL = f"{ROOT}/data/tokenizer/bangla_sp.model"
CKPT_DIR = f"{ROOT}/checkpoints"

BLOCK_SIZE = 512
MICRO_BATCH = 32          # per step; lower to 16 if you OOM on a T4
GRAD_ACCUM = 16           # effective batch = 32*16*512 ≈ 262k tokens
MAX_STEPS = 20_000        # set from your token budget: see print below
WARMUP_STEPS = 400
LR = 6e-4
MIN_LR = 6e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_EVERY = 500
EVAL_BATCHES = 60
SAVE_EVERY = 500
LOG_EVERY = 20
COMPILE = True
SEED = 1337

os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "No GPU. Runtime > Change runtime type > T4/L4 GPU."
# bf16 on Ampere+ (L4/A100), fp16 on T4
use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"device={device} dtype={amp_dtype}")


# ----------------------------- data -----------------------------------
class MemMapDataset(Dataset):
    def __init__(self, bin_path, block_size=BLOCK_SIZE, dtype=np.uint16):
        self.bin_path, self.block_size, self.dtype = bin_path, block_size, dtype
        self.total_tokens = os.path.getsize(bin_path) // np.dtype(dtype).itemsize
        self.num_samples = (self.total_tokens - 1) // block_size
        self.data = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        s = idx * self.block_size
        chunk = torch.from_numpy(
            np.array(self.data[s: s + self.block_size + 1], dtype=np.int64)
        )
        return chunk[:-1], chunk[1:]


train_ds = MemMapDataset(TRAIN_BIN)
val_ds = MemMapDataset(VAL_BIN)

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)
VOCAB = sp.get_piece_size()

tokens_per_step = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE
print(f"train tokens : {train_ds.total_tokens/1e6:.1f}M")
print(f"val tokens   : {val_ds.total_tokens/1e6:.1f}M")
print(f"tokens/step  : {tokens_per_step/1e3:.0f}k")
print(f"total budget : {MAX_STEPS*tokens_per_step/1e6:.0f}M tokens "
      f"(= {MAX_STEPS*tokens_per_step/train_ds.total_tokens:.2f} epochs)")


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH, shuffle=False,
                        num_workers=2, pin_memory=True,
                        persistent_workers=True, drop_last=True)
train_iter = infinite(train_loader)


# ----------------------------- model ----------------------------------
cfg = SLMConfig(vocab_size=VOCAB, block_size=BLOCK_SIZE,
                n_layer=6, n_head=6, n_embd=384, dropout=0.0)
model = BanglaSLM(cfg).to(device)
print(f"params: {model.num_params()/1e6:.2f}M "
      f"({model.num_params(True)/1e6:.2f}M non-embedding)")

optimizer = model.configure_optimizers(WEIGHT_DECAY, LR, device_type=device)
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

raw_model = model
if COMPILE:
    model = torch.compile(model)


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    prog = min(prog, 1.0)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))


# ------------------------- checkpoint / resume ------------------------
LAST = os.path.join(CKPT_DIR, "last.pt")
BEST = os.path.join(CKPT_DIR, "best.pt")


def save(path, step, best_val):
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": cfg.__dict__,
    }, tmp)
    os.replace(tmp, path)   # atomic: a disconnect mid-save can't corrupt it


start_step, best_val = 0, float("inf")
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location=device)
    raw_model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    start_step, best_val = ck["step"] + 1, ck["best_val"]
    print(f"resumed from step {start_step} (best val {best_val:.4f})")


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= EVAL_BATCHES:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(prompt="বাংলাদেশের", n=80):
    ids = torch.tensor([sp.encode(prompt)], device=device)
    out = raw_model.generate(ids, n, temperature=0.8, top_k=50)
    model.train()
    return sp.decode(out[0].tolist())


# ----------------------------- train ----------------------------------
model.train()
t0 = time.time()
for step in range(start_step, MAX_STEPS):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        total += loss.item()

    scaler.unscale_(optimizer)
    gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()

    if step % LOG_EVERY == 0:
        dt = time.time() - t0
        tps = LOG_EVERY * tokens_per_step / dt if step > start_step else 0
        print(f"step {step:6d} | loss {total:.4f} | ppl {math.exp(total):8.1f} "
              f"| lr {lr:.2e} | gnorm {gnorm:.2f} | {tps/1e3:.1f}k tok/s")
        t0 = time.time()

    if step > 0 and step % EVAL_EVERY == 0:
        vl = evaluate()
        print(f"  >> val loss {vl:.4f} | val ppl {math.exp(vl):.1f}")
        print(f"  >> sample: {sample()[:200]}")
        if vl < best_val:
            best_val = vl
            save(BEST, step, best_val)
            print("  >> new best, saved")
        t0 = time.time()

    if step > 0 and step % SAVE_EVERY == 0:
        save(LAST, step, best_val)

save(LAST, MAX_STEPS - 1, best_val)
print("done.")
