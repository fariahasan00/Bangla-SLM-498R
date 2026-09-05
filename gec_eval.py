"""
Bangla GEC evaluation harness.

Everything — your SLM, a fine-tuned seq2seq, a prompted LLM, an ensemble —
implements one interface:

    class System:
        name: str
        def correct(self, sources: list[str]) -> list[str]

so you write the evaluation code once and never touch it again.

Metrics: edit-level P/R/F0.5 (the standard GEC metric), GLEU, chrF,
exact match, and over-correction rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Protocol

CACHE_DIR = "gec_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================================
# 1. Tokenisation + edit extraction
# =====================================================================
_PUNCT = r"।,;:!?\"'()\[\]{}—–\-…"


def tokenize(s: str) -> list[str]:
    """Whitespace + punctuation split. '।' is the Bangla full stop (danda)."""
    s = s.strip()
    s = re.sub(rf"([{_PUNCT}])", r" \1 ", s)
    return s.split()


def extract_edits(source: str, target: str) -> set[tuple]:
    """
    Edits as (start, end, replacement) over source token indices.
    Two systems that produce the same correction produce the same edit set,
    which is what makes P/R/F0.5 comparable across systems.
    """
    src, tgt = tokenize(source), tokenize(target)
    edits = set()
    sm = SequenceMatcher(None, src, tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.add((i1, i2, " ".join(tgt[j1:j2])))
    return edits


# =====================================================================
# 2. Metrics
# =====================================================================
def prf(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p == 0 and r == 0:
        return p, r, 0.0
    b2 = beta ** 2
    f = (1 + b2) * p * r / (b2 * p + r)
    return p, r, f


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def gleu(source: str, hyp: str, ref: str, max_n: int = 4) -> float:
    """
    GLEU (Napoles et al.) — BLEU that penalises n-grams the system kept from
    the source but which the reference changed. The standard GEC fluency metric.
    """
    s, h, r = tokenize(source), tokenize(hyp), tokenize(ref)
    if not h:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hn, rn, sn = _ngrams(h, n), _ngrams(r, n), _ngrams(s, n)
        if not hn:
            precisions.append(0.0)
            continue
        # reward overlap with ref, penalise overlap with source-not-in-ref
        overlap = sum((hn & rn).values())
        penalty = sum((hn & (sn - rn)).values())
        precisions.append(max(overlap - penalty, 0) / sum(hn.values()))
    if min(precisions) == 0:
        geo = 0.0
    else:
        prod = 1.0
        for p in precisions:
            prod *= p
        geo = prod ** (1 / max_n)
    bp = 1.0 if len(h) > len(r) else pow(2.718281828, 1 - len(r) / max(len(h), 1))
    return bp * geo


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score — robust for morphologically rich Bangla."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        hk = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rk = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        inter = sum((hk & rk).values())
        ps.append(inter / sum(hk.values()) if sum(hk.values()) else 0.0)
        rs.append(inter / sum(rk.values()) if sum(rk.values()) else 0.0)
    p, r_ = sum(ps) / n, sum(rs) / n
    if p + r_ == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r_ / (b2 * p + r_)


@dataclass
class Result:
    system: str
    n: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f05: float = 0.0
    gleu: float = 0.0
    chrf: float = 0.0
    exact_match: float = 0.0
    overcorrection: float = 0.0   # edits made on already-correct sentences
    unchanged_rate: float = 0.0   # how often the system does nothing


def evaluate(name: str, sources: list[str], hyps: list[str],
             refs: list[str]) -> Result:
    tp = fp = fn = 0
    gl, cf, em = 0.0, 0.0, 0
    clean_total = clean_touched = 0
    unchanged = 0

    for src, hyp, ref in zip(sources, hyps, refs):
        gold = extract_edits(src, ref)
        pred = extract_edits(src, hyp)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

        gl += gleu(src, hyp, ref)
        cf += chrf(hyp, ref)
        em += int(hyp.strip() == ref.strip())
        if not pred:
            unchanged += 1
        if not gold:                       # source was already correct
            clean_total += 1
            clean_touched += int(bool(pred))

    n = len(sources)
    p, r, f = prf(tp, fp, fn)
    return Result(
        system=name, n=n, precision=p, recall=r, f05=f,
        gleu=gl / n, chrf=cf / n, exact_match=em / n,
        overcorrection=(clean_touched / clean_total) if clean_total else 0.0,
        unchanged_rate=unchanged / n,
    )


def table(results: list[Result]) -> str:
    hdr = (f"{'system':<32}{'P':>7}{'R':>7}{'F0.5':>8}{'GLEU':>7}"
           f"{'chrF':>7}{'EM':>7}{'OverC':>7}{'NoOp':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(results, key=lambda x: -x.f05):
        lines.append(f"{r.system:<32}{r.precision:>7.3f}{r.recall:>7.3f}"
                     f"{r.f05:>8.3f}{r.gleu:>7.3f}{r.chrf:>7.3f}"
                     f"{r.exact_match:>7.3f}{r.overcorrection:>7.3f}"
                     f"{r.unchanged_rate:>7.3f}")
    return "\n".join(lines)


# =====================================================================
# 3. System interface + caching
# =====================================================================
class System(Protocol):
    name: str
    def correct(self, sources: list[str]) -> list[str]: ...


def cached(system: System) -> Callable[[list[str]], list[str]]:
    """Never pay for the same LLM call twice."""
    def run(sources):
        key = hashlib.md5(
            (system.name + "||" + "||".join(sources)).encode("utf-8")
        ).hexdigest()
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
        out = system.correct(sources)
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        return out
    return run


# =====================================================================
# 4. Systems
# =====================================================================
class Identity:
    """Do-nothing baseline. Its F0.5 is your floor; beat it or the system
    is actively harmful."""
    name = "identity (no-op)"

    def correct(self, sources):
        return list(sources)


class SLMCorrector:
    """Your pretrained SLM, fine-tuned on <incorrect> [SEP] <correct> [EOS]."""
    name = "bangla-slm-15m (ft)"

    def __init__(self, model, sp, device="cuda", sep="\t", max_new=128):
        self.model, self.sp, self.device = model, sp, device
        self.sep, self.max_new = sep, max_new

    def correct(self, sources):
        import torch
        out = []
        for s in sources:
            ids = torch.tensor([self.sp.encode(s + self.sep)], device=self.device)
            gen = self.model.generate(ids, self.max_new, temperature=0.1, top_k=1,
                                      eos_id=self.sp.eos_id())
            text = self.sp.decode(gen[0].tolist())
            out.append(text.split(self.sep)[-1].strip() or s)
        return out


class HFSeq2Seq:
    """BanglaT5 / mT5 fine-tuned baseline."""
    def __init__(self, model_name, prefix="সংশোধন করো: ", device="cuda"):
        self.name = f"hf:{model_name}"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.prefix, self.device = prefix, device

    def correct(self, sources, batch_size=16):
        import torch
        outs = []
        for i in range(0, len(sources), batch_size):
            batch = [self.prefix + s for s in sources[i:i + batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=128, num_beams=4)
            outs += self.tok.batch_decode(gen, skip_special_tokens=True)
        return outs


# ---- prompting strategies (this is a whole axis of your paper) --------
PROMPTS = {
    "zero_shot": lambda s, ex: (
        "নিচের বাংলা বাক্যটির ব্যাকরণগত ভুল সংশোধন করো। "
        "শুধু সংশোধিত বাক্যটি লেখো, অন্য কিছু নয়। বাক্যে ভুল না থাকলে "
        "বাক্যটি অপরিবর্তিত রাখো।\n\n"
        f"বাক্য: {s}\nসংশোধিত:"
    ),
    "few_shot": lambda s, ex: (
        "নিচের বাংলা বাক্যগুলোর ব্যাকরণগত ভুল সংশোধন করো।\n\n"
        + "\n\n".join(f"বাক্য: {a}\nসংশোধিত: {b}" for a, b in ex)
        + f"\n\nবাক্য: {s}\nসংশোধিত:"
    ),
    "cot": lambda s, ex: (
        "নিচের বাংলা বাক্যটি বিশ্লেষণ করো। ধাপে ধাপে ভুলগুলো চিহ্নিত করো, "
        "তারপর শেষ লাইনে 'সংশোধিত: ' লিখে সংশোধিত বাক্যটি দাও।\n\n"
        f"বাক্য: {s}"
    ),
    # Vaiyakarana's 12 error classes — give the model the taxonomy
    "error_aware": lambda s, ex: (
        "বাংলা ব্যাকরণে সাধারণ ভুলের ধরন: বিভক্তি, কারক, সমাস, ক্রিয়ার কাল, "
        "বচন, লিঙ্গ, সন্ধি, বানান, পদক্রম, যতিচিহ্ন, বাহুল্য দোষ, অন্বয় দোষ।\n"
        "এই ধরনগুলো মাথায় রেখে নিচের বাক্যটি সংশোধন করো। শুধু সংশোধিত বাক্য লেখো।\n\n"
        f"বাক্য: {s}\nসংশোধিত:"
    ),
}


class LLMCorrector:
    """Wrap any chat LLM. `call_fn(prompt) -> str` is yours to supply."""
    def __init__(self, model_id, call_fn, strategy="zero_shot", exemplars=None):
        self.name = f"{model_id} [{strategy}]"
        self.call_fn, self.strategy = call_fn, strategy
        self.exemplars = exemplars or []

    def correct(self, sources):
        out = []
        for s in sources:
            raw = self.call_fn(PROMPTS[self.strategy](s, self.exemplars))
            out.append(self._parse(raw, s))
        return out

    @staticmethod
    def _parse(raw, fallback):
        raw = raw.strip()
        if "সংশোধিত:" in raw:
            raw = raw.split("সংশোধিত:")[-1]
        raw = raw.strip().strip('"').split("\n")[0].strip()
        return raw or fallback


class SelfRefine:
    """Two-pass: correct, then critique-and-revise. A prompting-strategy result."""
    def __init__(self, base: LLMCorrector, call_fn):
        self.name = base.name.replace("]", " + self-refine]")
        self.base, self.call_fn = base, call_fn

    def correct(self, sources):
        first = self.base.correct(sources)
        out = []
        for src, cand in zip(sources, first):
            p = ("মূল বাক্য: " + src + "\nপ্রস্তাবিত সংশোধন: " + cand +
                 "\n\nসংশোধনটি কি সঠিক এবং সর্বনিম্ন প্রয়োজনীয় পরিবর্তন? "
                 "অপ্রয়োজনীয় পরিবর্তন থাকলে বাতিল করো। "
                 "শেষ লাইনে 'সংশোধিত: ' লিখে চূড়ান্ত বাক্য দাও।")
            out.append(LLMCorrector._parse(self.call_fn(p), cand))
        return out


# ---- ensembling strategies (the other axis) ---------------------------
class EditVoteEnsemble:
    """
    Majority vote at the EDIT level, not the sentence level. Each member
    proposes edits; keep those proposed by >= threshold members. Raises
    precision sharply, which is what F0.5 rewards.
    """
    def __init__(self, members: list[System], threshold: int | None = None):
        self.members = members
        self.threshold = threshold or (len(members) // 2 + 1)
        self.name = f"ensemble:vote(k={self.threshold}/{len(members)})"

    def correct(self, sources):
        all_hyps = [cached(m)(sources) for m in self.members]
        out = []
        for i, src in enumerate(sources):
            votes = Counter()
            for hyps in all_hyps:
                for e in extract_edits(src, hyps[i]):
                    votes[e] += 1
            keep = [e for e, c in votes.items() if c >= self.threshold]
            out.append(apply_edits(src, keep))
        return out


class PerplexityRerank:
    """
    THE role for a 15M SLM. Members generate candidates; the SLM scores each
    with length-normalised log-prob and picks the winner. Small model, real
    contribution — and you can report it as a cheap alternative to LLM-as-judge.
    """
    def __init__(self, members: list[System], scorer, sp, device="cuda",
                 alpha=0.0):
        self.members, self.scorer, self.sp = members, scorer, sp
        self.device, self.alpha = device, alpha
        self.name = f"ensemble:slm-rerank({len(members)})"

    def _score(self, text):
        import torch
        ids = torch.tensor(self.sp.encode(text), device=self.device)
        lp, n = self.scorer.sequence_logprob(ids)
        return lp / max(n, 1)

    def correct(self, sources):
        all_hyps = [cached(m)(sources) for m in self.members]
        out = []
        for i, src in enumerate(sources):
            cands = {src} | {h[i] for h in all_hyps}   # source is always a candidate
            best, best_s = src, -1e18
            for c in cands:
                s = self._score(c) - self.alpha * len(extract_edits(src, c))
                if s > best_s:
                    best, best_s = c, s
            out.append(best)
        return out


class Pipeline:
    """Detector gates corrector: only sentences flagged as erroneous get edited.
    Directly attacks over-correction."""
    def __init__(self, detector: Callable[[str], bool], corrector: System):
        self.detector, self.corrector = detector, corrector
        self.name = f"pipeline:detect->{corrector.name}"

    def correct(self, sources):
        flags = [self.detector(s) for s in sources]
        idxs = [i for i, f in enumerate(flags) if f]
        fixed = self.corrector.correct([sources[i] for i in idxs]) if idxs else []
        out = list(sources)
        for i, h in zip(idxs, fixed):
            out[i] = h
        return out


def apply_edits(source: str, edits: list[tuple]) -> str:
    toks = tokenize(source)
    for i1, i2, rep in sorted(edits, key=lambda e: -e[0]):
        toks[i1:i2] = rep.split() if rep else []
    return " ".join(toks)


# =====================================================================
# 5. Runner
# =====================================================================
@dataclass
class Benchmark:
    sources: list[str]
    references: list[str]
    results: list[Result] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path, src_key="incorrect", ref_key="correct"):
        srcs, refs = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                srcs.append(d[src_key])
                refs.append(d[ref_key])
        return cls(srcs, refs)

    def run(self, system: System) -> Result:
        hyps = cached(system)(self.sources)
        r = evaluate(system.name, self.sources, hyps, self.references)
        self.results.append(r)
        print(f"{r.system:<32} F0.5={r.f05:.3f}  GLEU={r.gleu:.3f}  "
              f"overcorrection={r.overcorrection:.3f}")
        return r

    def report(self):
        print("\n" + table(self.results))

    def to_csv(self, path="results.csv"):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(Result.__annotations__))
            w.writeheader()
            for r in self.results:
                w.writerow(r.__dict__)
        print(f"wrote {path}")


if __name__ == "__main__":
    # self-test with a toy set: 3 erroneous + 1 already-correct sentence
    srcs = [
        "আমি বাজারে যাব না কারণ আমি অসুস্থ আছি ।",
        "সে গতকাল স্কুলে যায় ।",
        "ছেলেটি বই পড়ে ।",
        "তারা কাল আসিবে ।",
    ]
    refs = [
        "আমি বাজারে যাব না কারণ আমি অসুস্থ ।",
        "সে গতকাল স্কুলে গিয়েছিল ।",
        "ছেলেটি বই পড়ে ।",
        "তারা কাল আসবে ।",
    ]

    class Perfect:
        name = "oracle"
        def correct(self, s): return list(refs)

    class Overeager:
        name = "over-corrector"
        def correct(self, s):
            return ["আমি বাজারে যাব না কারণ আমি অসুস্থ ।",
                    "সে গতকাল স্কুলে গিয়েছিল ।",
                    "ছেলেটি একটি বই পড়ে ।",      # edits an already-correct sentence
                    "তারা আগামীকাল আসবে ।"]

    b = Benchmark(srcs, refs)
    for s in (Identity(), Perfect(), Overeager()):
        b.run(s)
    b.run(EditVoteEnsemble([Perfect(), Overeager(), Identity()], threshold=2))
    b.report()
