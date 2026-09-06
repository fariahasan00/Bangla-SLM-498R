# Bangla-SLM-498R

A small decoder-only language model (~17M parameters) pretrained from scratch on Bangla
text, built to support a **Bangla Grammatical Error Correction (GEC)** evaluation study.

The project has two halves:

1. **`slm_model.py` + `train_slm.py`** — the model and its pretraining pipeline.
   A modern-but-minimal transformer (RMSNorm, RoPE, SwiGLU, flash attention,
   tied embeddings), written to be readable next to nanoGPT.
2. **`gec_eval.py`** — a system-agnostic GEC evaluation harness. Every system
   (this SLM, a fine-tuned seq2seq, a prompted LLM, an ensemble) implements one
   `correct(sources) -> hypotheses` interface, so the metric code is written once
   and never touched again.

---

## Results

Pretraining run completed on a single NVIDIA RTX 3060 (8 GB).

| Metric | Start | Final |
| --- | --- | --- |
| Validation loss | 8.9874 | **3.2483** |
| Validation perplexity | 8002 | **25.75** |

| Training detail | Value |
| --- | --- |
| Optimizer steps | 5,750 |
| Tokens per step | 262,144 (32 micro-batch x 16 grad-accum x 512 block) |
| Total tokens seen | 1.51 B (approx. 4 epochs over a 377 M-token corpus) |
| Wall-clock time | approx. 7 hours |
| Throughput | approx. 60k tokens/sec |
| Final train/val gap | 0.056 — **no overfitting**; validation was still improving at the last eval |

The small train/val gap means the model had not saturated when the run ended, so a
longer run would likely improve perplexity further.

### Sample output

Prompted with `বাংলাদেশের প্রধানমন্ত্রী`:

> বাংলাদেশের প্রধানমন্ত্রী দেশবাসীকে শুভেচ্ছা জানিয়েছেন। প্রধানমন্ত্রীর প্রেস সচিব ইহসানুল করিম
> জানান, প্রধানমন্ত্রী শেখ হাসিনা আজ বিকেলে ... সৌজন্য সাক্ষাৎ করতে পারবেন বলে জানান।

The model produces fluent, grammatical Bangla in a newspaper register. It is **not**
factually reliable — at 17M parameters it learned the language, not the world.

---

## Running It Yourself (No Local GPU? Start Here)

Everything below runs inside a free Google Colab notebook — no local install, no
CUDA setup needed. Each numbered block is one cell.

### How to run it

**1. Clone the repo:**
```python
!git clone https://github.com/fariahasan00/Bangla-SLM-498R.git
%cd Bangla-SLM-498R
!git checkout farhana
!pip install sentencepiece huggingface_hub -q
```

**2. Download the tokenizer** (public, no login needed):
```python
from huggingface_hub import hf_hub_download
tok_path = hf_hub_download(
    repo_id="faria00/bangla-slm-pretrain-data",
    filename="bangla_sp.model",
    repo_type="dataset",
)
```

**3. Get `best.pt`** from whoever trained it — it's not in this repo or the public
dataset (see "Repository contents" below) — then upload it:
```python
from google.colab import files
uploaded = files.upload()   # choose the checkpoint zip
```
```bash
!unzip -o "best.pt.zip"
```

**4. If that didn't leave you a single `best.pt` file** — it likely extracted into a
folder of loose files (`data.pkl`, `data/0`, `data/1`, ...) instead. A `.pt` file is
itself a zip archive internally, and re-zipping one for transfer explodes that
structure. Fix it:
```bash
!mkdir -p repack/archive && cp -r best.pt/* repack/archive/ \
  && cd repack && zip -r -q ../best_fixed.pt archive && cd .. && rm -rf repack
```

**5. Generate:**
```python
import torch, sentencepiece as spm
from slm_model import BanglaSLM, SLMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("best_fixed.pt", map_location=device)
model = BanglaSLM(SLMConfig(**ck["config"])).to(device)
model.load_state_dict(ck["model"])
model.eval()

sp = spm.SentencePieceProcessor()
sp.load(tok_path)

ids = torch.tensor([sp.encode("বাংলাদেশের প্রধানমন্ত্রী")], device=device)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=50)
print(sp.decode(out[0].tolist()))
```

### What to expect

- **A confirmation line first** — loading the checkpoint prints its training step
  and validation loss (step 5,750, loss ≈3.2483 for the run reported above).
- **Fluent, grammatical Bangla**, continuing whatever prompt you give it, similar in
  register to the sample under "Results" above. It won't be word-for-word identical
  to that sample every time — generation samples randomly (see `temperature` /
  `top_k`), so the same prompt gives different but similarly fluent output on each run.
- **Don't expect factual accuracy.** At 17M parameters, the model learned Bangla
  grammar and style, not world knowledge — treat anything it states as fact as
  unverified.
- **Don't expect long-range coherence.** The 512-token context window and small
  parameter count make it reliable over a sentence or two, not a full article.
- **If you see gibberish or repeated characters instead** — double-check the
  checkpoint and tokenizer came from the same training run. Mixing mismatched ones
  won't throw an error, it'll just silently produce nonsense.
- A Colab disconnect (common after ~90 minutes idle) wipes everything uploaded or
  downloaded in this section — just redo steps 1–4.

## Repository contents

| File | Purpose |
| --- | --- |
| `slm_model.py` | Model definition (`SLMConfig`, `BanglaSLM`). Run directly for a shape/parameter self-test. |
| `train_slm.py` | Pretraining script: downloads data, trains, evaluates, checkpoints, auto-resumes. |
| `gec_eval.py` | GEC evaluation harness — P/R/F0.5, GLEU, chrF, exact match, over-correction rate. |
| `modelArchitecture.ipynb` | Architecture walkthrough with sanity checks. |
| `bangla_NLP.ipynb` | Corpus exploration and tokenizer training. |
| `BanglaSLM.ipynb`, `BanglaSLM15M.ipynb` | Earlier prototyping notebooks. |

**Not in this repository** (too large for Git — see `.gitignore`): model checkpoints
(~207 MB each), tokenized data (~750 MB), and the raw corpus (~6 GB). The training
data downloads automatically — see below.

---

## Requirements

- **Python 3.10+** (developed on 3.14)
- **An NVIDIA GPU with CUDA.** The script deliberately refuses to run on CPU — a model
  this size would take weeks. About 4 GB of VRAM suffices at the default settings.
- Roughly 1 GB free disk for the dataset, plus ~450 MB per pair of checkpoints.

### Install

Install PyTorch with CUDA support first, matching your driver
(see <https://pytorch.org/get-started/locally/>):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece numpy huggingface_hub
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Quick start

### 1. Pretrain

```bash
python train_slm.py
```

That is the entire command. On first run it downloads `train.bin`, `val.bin`, and the
SentencePiece tokenizer (~750 MB total) from the public Hugging Face dataset
[`faria00/bangla-slm-pretrain-data`](https://huggingface.co/datasets/faria00/bangla-slm-pretrain-data)
into `./data`, then starts training.

To reproduce the run reported above:

```bash
python train_slm.py --max-steps 5750 --save-every 200
```

To confirm the setup works before committing hours to it, run a handful of steps:

```bash
python train_slm.py --max-steps 6 --log-every 1 --eval-every 5 --eval-batches 3
```

**If the data is already on disk**, point at it instead of downloading:

```bash
python train_slm.py --data-dir /path/to/folder
```

That folder must contain `train.bin`, `val.bin`, and `tokenizer/bangla_sp.model`.

### 2. Generate text from a trained checkpoint

```python
import sys, torch, sentencepiece as spm
from slm_model import BanglaSLM, SLMConfig

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # needed on Windows

ck = torch.load("checkpoints/best.pt", map_location="cuda")
model = BanglaSLM(SLMConfig(**ck["config"])).cuda()
model.load_state_dict(ck["model"])
model.eval()

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/bangla_sp.model")

ids = torch.tensor([sp.encode("বাংলাদেশের প্রধানমন্ত্রী")], device="cuda")
out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=50)
print(sp.decode(out[0].tolist()))
```

The Windows console defaults to an encoding that cannot represent Bangla characters,
which is why `reconfigure` comes first. `train_slm.py` already does this internally.

### 3. Run the GEC evaluation harness

```bash
python gec_eval.py
```

This runs a built-in self-test over four toy sentences and prints a metrics table
(`oracle` at F0.5 = 1.000, `identity` at 0.000 as the floor). It needs no GPU, no
dataset, and no trained model — it exists to verify the metric implementations.

To evaluate on a real dataset, supply JSONL with one object per line:

```json
{"incorrect": "সে গতকাল স্কুলে যায় ।", "correct": "সে গতকাল স্কুলে গিয়েছিল ।"}
```

```python
from gec_eval import Benchmark, Identity

bench = Benchmark.from_jsonl("your_gec_data.jsonl")
bench.run(Identity())        # baseline floor — any real system must beat this
bench.report()
bench.to_csv("results.csv")
```

---

## Model architecture

| Setting | Value |
| --- | --- |
| Parameters | 17.23 M total, 14.16 M excluding embeddings |
| Layers | 8 |
| Attention heads | 6 (head dimension 64) |
| Embedding dimension | 384 |
| Context length | 512 tokens |
| Vocabulary | 8,000 SentencePiece (unk=0, bos=1, eos=2, pad=3) |
| Normalisation | RMSNorm, pre-norm |
| Positional encoding | RoPE (rotary, base 10000) |
| Feed-forward | SwiGLU, hidden dimension 1024 |
| Attention kernel | `F.scaled_dot_product_attention`, causal |
| Embeddings | Tied between input and output |
| Dropout | 0.0 |

Verify the architecture loads and runs:

```bash
python slm_model.py
```

### Training data

| Split | Tokens |
| --- | --- |
| Train | 377,235,487 |
| Validation | 6,380,700 |

Bangla Common Crawl text, tokenized with a purpose-built 8k SentencePiece model and
stored as flat `uint16` arrays.

---

## Key training options

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-steps` | 20000 | **Keep constant across resumes** — the LR schedule is derived from it. |
| `--micro-batch` | 32 | Lower this first if you hit CUDA out-of-memory. |
| `--grad-accum` | 16 | Effective batch = micro-batch x grad-accum x block-size. |
| `--lr` / `--min-lr` | 6e-4 / 6e-5 | Linear warmup (400 steps), then cosine decay. |
| `--save-every` | 500 | Checkpoint interval. Lower it if power is unreliable. |
| `--eval-every` | 500 | Validation pass plus a generated sample. |
| `--preload` / `--no-preload` | preload on | Read token files into RAM vs. memory-map from disk. |
| `--num-workers` | 0 | DataLoader worker processes. |
| `--data-dir` | `./data` | Where `train.bin`, `val.bin`, and `tokenizer/` live. |
| `--skip-download` | off | Never contact Hugging Face; fail if files are missing. |

Full list: `python train_slm.py --help`

---

## Checkpoints and resuming

Training writes two files into `./checkpoints`:

- **`last.pt`** — most recent state (weights, optimizer, step). Used for **resuming**.
- **`best.pt`** — lowest validation loss seen so far. Used for **inference**. Only
  overwritten on a genuine improvement.

Both contain `model`, `optimizer`, `scaler`, `step`, `best_val_loss`, and `config`.

**Resuming is automatic.** If `checkpoints/last.pt` exists, training continues from it.
If the run is interrupted — including by a power cut — simply re-run the same command.

Each checkpoint is written to a temporary file and then atomically renamed, so a crash
mid-save leaves either the old checkpoint or the new one, never a corrupt file.

Two things to watch:

- **To start a fresh run, delete `checkpoints/` first.** Otherwise training silently
  resumes, and a changed `--n-layer` or `--n-embd` will fail with a confusing
  `load_state_dict` error.
- **Pass the same `--max-steps` on every resume.** It is not stored in the checkpoint,
  and the cosine learning-rate schedule is computed from it.

To train *further* after a run completes, resume with a higher ceiling — the learning
rate warm-restarts and decays again:

```bash
python train_slm.py --max-steps 8630 --save-every 200   # 6 epochs total
```

At 262,144 tokens per step over a 377 M-token corpus, one epoch is **1,439 steps**.

---

## Troubleshooting

**`SystemError: No CUDA GPU was detected`**
Expected on CPU-only machines. This model needs a GPU; there is no CPU fallback.

**`CUDA out of memory`**
Lower `--micro-batch` (try 16, then 8) and raise `--grad-accum` to match, keeping their
product constant so the effective batch size does not change.

**`RuntimeError: DataLoader worker exited unexpectedly`, or a crash with Windows
exception code `0xc0000006` (`STATUS_IN_PAGE_ERROR`)**
This occurs when the token files are memory-mapped from a **mechanical hard drive**.
Shuffling turns every batch into a random ~1 KB read, and an HDD cannot sustain that
seek rate; one stalled page-in kills the process. The default `--preload` avoids it by
reading the files into RAM once. Use `--no-preload` only if `train.bin` does not fit in
RAM, and preferably only on an SSD.

**`UnicodeEncodeError` when printing samples on Windows**
Already handled inside `train_slm.py`, which reconfigures stdout to UTF-8 at startup.
Apply the same fix in your own scripts.

**Download fails, or `Could not resolve host`**
If you are behind a VPN or DNS proxy (for example Cloudflare WARP), disable it and
retry, or download the three files manually from the dataset page and pass
`--data-dir` together with `--skip-download`.

---

## Project status

- [x] Corpus collection, cleaning, and 8k SentencePiece tokenizer
- [x] Model architecture with sanity checks
- [x] Pretraining pipeline: checkpointing, auto-resume, mixed precision
- [x] Pretraining run — 4 epochs, validation perplexity 25.75
- [x] GEC evaluation harness, metrics verified against a self-test
- [ ] Bangla GEC evaluation dataset (`incorrect` / `correct` JSONL pairs)
- [ ] GEC benchmark results

### Intended role for the SLM in GEC

A 17M-parameter model will not out-generate a fine-tuned BanglaT5 or a prompted LLM,
and the harness is designed around that reality. The intended contribution is
`PerplexityRerank` in `gec_eval.py`: larger systems propose correction candidates, and
this SLM scores each one with length-normalised log-probability via
`BanglaSLM.sequence_logprob`, then picks the winner. That requires **no fine-tuning** —
the pretrained checkpoint works as-is — and offers a cheap, reproducible alternative to
LLM-as-judge reranking.

The alternative path, `SLMCorrector`, would require fine-tuning the model on
`<incorrect> [SEP] <correct>` pairs. No fine-tuning script exists in this repository yet.

---

## License and attribution

Course project (498R). Training corpus derived from Bangla Common Crawl.
