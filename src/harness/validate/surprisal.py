"""Local token surprisal / predictability: how guessable each sentence is to a small language model (distilgpt2, int8 ONNX).

Machine prose is predictable; human prose is not. Zero API cost. onnxruntime + tokenizers + numpy only (no torch), imported lazily.

What is measured (per sentence, over the UNMASKED prose tokens only):
  mean            masked mean surprisal, nats/token. Tokens that say nothing about style are left out: citations ("(Smith, 2021)",
                  "Smith (2021)"), digits and punctuation, capitalised mid-sentence words and acronyms (surnames, SHRM), and any token the
                  model finds shocking (> RARE nats: rare jargon). Without this, two surnames lift a sentence of pure glue to 4.8 nats.
  min_window      the lowest mean over any 6 consecutive unmasked tokens: catches a predictable island ("in the sense that why employees")
                  inside a technical sentence. predictable_span is the exact text of that window.
  lstr            share of unmasked tokens the model ranks in its top 5 guesses.
A sentence is flagged when it has >= 4 unmasked tokens and (mean < mean threshold or min_window < window threshold).

Cost (measured, 2 threads, distilgpt2 int8): ~130MB session + ~35MB working, ~2ms per token, so a 1,200-word section takes ~3.5s on 2 threads
(~2s on 8). That is compute, not call overhead: one forward pass per 96-token window, no past_key_values. Paragraph results are cached, so
re-scoring a section after an edit only pays for the paragraphs that changed.

`get_scorer()` is None when HARNESS_SURPRISAL=off, the model files are missing or onnxruntime is absent; every caller treats None as "skip".
Files in HARNESS_SURPRISAL_DIR (default data/models): lm-q.onnx (Xenova/distilgpt2 onnx/decoder_model_quantized.onnx), tokenizer.json."""
import bisect
import json
import math
import re
import statistics
import threading
from collections import OrderedDict
from pathlib import Path

from .. import config, limits
from ..text.citations import NARRATIVE_RE
from ..text.sentences import split_sentences

WINDOW = 96          # tokens per forward pass: bounds the logits array (96 x 50257 floats = 19MB)
BOS = 50256
SPAN = 6             # tokens in the rolling window
RARE = 9.5           # nats: a token this surprising is jargon or a name, not style
MIN_TOKENS = 4       # unmasked tokens a sentence needs before it can be flagged
LSTR_RANK = 5
_SCORER = None
_LOCK = threading.Lock()
_CACHE: OrderedDict = OrderedDict()
CACHE_MAX = 512
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_PAREN_CITE = re.compile(r"\([^()]*(?:19|20)\d\d[a-z]?[^()]*\)")
_BLOCK = re.compile(r"\[(?:FIGURE|TABLE)_START\]|^\s*[|!<]", re.M)


def model_dir() -> Path:
    return Path(config.get("HARNESS_SURPRISAL_DIR", "") or Path(__file__).resolve().parents[3] / "data" / "models")


class OnnxScorer:
    def __init__(self, onnx_path: Path, tok_path: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(config.get("HARNESS_SURPRISAL_THREADS", "2") or 2)
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False        # the arena keeps ~200MB of logits buffers alive
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(tok_path))
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def token_logprobs(self, text: str) -> list[tuple[int, int, float, int]]:
        """[(char_start, char_end, ln p(token | previous tokens), rank)] for every token of `text`; rank 1 = the model's first guess.
        One forward pass per window of WINDOW tokens; log-softmax and rank are vectorised over the whole window."""
        import numpy as np
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offs, out = enc.ids, enc.offsets, []
        for a in range(0, len(ids), WINDOW):
            part = [BOS] + ids[a:a + WINDOW]
            feed = {"input_ids": np.array([part], dtype=np.int64)}
            if "attention_mask" in self.inputs:
                feed["attention_mask"] = np.ones((1, len(part)), dtype=np.int64)
            logits = self.sess.run(None, feed)[0][0, :-1].astype(np.float32, copy=False)      # position t predicts token t+1
            tgt = np.array(part[1:])
            at = logits[np.arange(len(tgt)), tgt]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            lse = np.log(np.exp(shifted).sum(axis=-1)) + logits.max(axis=-1)
            lp = at - lse
            rank = (logits > at[:, None]).sum(axis=-1) + 1
            out += [(offs[a + k][0], offs[a + k][1], float(lp[k]), int(rank[k])) for k in range(len(tgt))]
            del logits, shifted
        return out


def get_scorer():
    """The shared scorer, or None. Never raises."""
    global _SCORER
    if config.get("HARNESS_SURPRISAL", "on").lower() == "off":
        return None
    if _SCORER is None:
        with _LOCK:
            if _SCORER is None:
                d = model_dir()
                try:
                    if not (d / "lm-q.onnx").exists() or not (d / "tokenizer.json").exists():
                        return None
                    _SCORER = OnnxScorer(d / "lm-q.onnx", d / "tokenizer.json")
                except Exception:  # noqa: BLE001 - optional signal; a missing runtime must not stop a section
                    return None
    return _SCORER


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def thresholds() -> tuple[float, float]:
    """(mean, window) in nats/token. data/surprisal_baseline.json (scripts/calibrate_surprisal.py) overrides the defaults."""
    try:
        p = Path(__file__).resolve().parents[3] / "data" / "surprisal_baseline.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        return float(d["mean"]), float(d["window"])
    except Exception:  # noqa: BLE001
        return limits.SURPRISAL_MEAN, limits.SURPRISAL_WINDOW


def low_threshold() -> float:
    return thresholds()[0]


def _tokens(para: str, scorer) -> list:
    key = (id(scorer), para)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    toks = scorer.token_logprobs(para)
    with _LOCK:
        _CACHE[key] = toks
        while len(_CACHE) > CACHE_MAX:
            _CACHE.popitem(last=False)
    return toks


def _citation_spans(para: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PAREN_CITE.finditer(para)] + [(m.start(), m.end()) for m in NARRATIVE_RE.finditer(para)]


def _masked(para: str, toks: list, a: int, cites: list, words: list, starts: list) -> list[bool]:
    """True for each token that says nothing about style (see the module doc)."""
    first = next((w for w in words if w[0] >= a), (a, a))[0]
    out = []
    for (x, y, lp, _r) in toks:
        t = para[x:y]
        w = words[i] if (i := bisect.bisect_right(starts, x) - 1) >= 0 and words[i][0] <= x < words[i][1] else None
        m = (not any(ch.isalnum() for ch in t) or any(ch.isdigit() for ch in t) or -lp > RARE
             or any(x < e and y > s for s, e in cites))
        if w and not m:
            word = para[w[0]:w[1]]
            m = (word[:1].isupper() and w[0] != first) or (len(word) >= 2 and word.isupper() and any(c.isalpha() for c in word))
        out.append(m)
    return out


def _metrics(para: str, toks: list, a: int, b: int, cites: list, words: list, starts: list) -> dict | None:
    mine = [t for t in toks if t[0] >= a and t[1] <= b]
    if not mine:
        return None
    mask = _masked(para, mine, a, cites, words, starts)
    keep = [t for t, m in zip(mine, mask) if not m]
    raw = sum(-t[2] for t in mine) / len(mine)
    if not keep:
        return {"n": 0, "mean": raw, "raw_mean": raw, "min_window": raw, "lstr": 0.0, "predictable_span": ""}
    s = [-t[2] for t in keep]
    size = min(SPAN, len(s))
    means = [sum(s[i:i + size]) / size for i in range(len(s) - size + 1)]
    i = min(range(len(means)), key=means.__getitem__)
    return {"n": len(keep), "mean": sum(s) / len(s), "raw_mean": raw, "min_window": means[i],
            "lstr": sum(1 for t in keep if t[3] <= LSTR_RANK) / len(keep),
            "predictable_span": para[keep[i][0]:keep[i + size - 1][1]]}


def score_section_surprisal(text: str, scorer=None, low: float | None = None, window: float | None = None) -> dict | None:
    """-> {mean, perplexity, burstiness, n_tokens, sentences:[{text, para, n, mean, raw_mean, min_window, lstr, predictable_span, flagged}],
    low:[flagged sentences], threshold:{mean, window}} or None (no scorer / scorer failed / nothing to score).
    mean/perplexity are over unmasked tokens; burstiness = std of the sentence means (human text varies more)."""
    scorer = scorer if scorer is not None else get_scorer()
    if scorer is None or not (text or "").strip():
        return None
    tm, tw = thresholds()
    tm = tm if low is None else low
    tw = (tw if low is None else low) if window is None else window
    sents, tot, n = [], 0.0, 0
    try:
        for i, para in enumerate(q.strip() for q in re.split(r"\n\s*\n", text) if q.strip()):
            if para.startswith("#") or _BLOCK.search(para):
                continue
            toks = _tokens(para, scorer)
            cites = _citation_spans(para)
            words = [(m.start(), m.end()) for m in _WORD.finditer(para)]
            starts = [w[0] for w in words]
            pos = 0
            for s in split_sentences(para):
                a = para.find(s, pos)
                if a < 0:
                    continue
                pos = a + len(s)
                m = _metrics(para, toks, a, pos, cites, words, starts)
                if m is None or m["n"] == 0:
                    continue
                m.update(text=s, para=i, flagged=m["n"] >= MIN_TOKENS and (m["mean"] < tm or m["min_window"] < tw))
                sents.append(m)
                tot, n = tot + m["mean"] * m["n"], n + m["n"]
    except Exception:  # noqa: BLE001
        return None
    if not sents:
        return None
    mean = tot / n
    return {"mean": mean, "perplexity": math.exp(mean), "n_tokens": n,
            "burstiness": statistics.pstdev([s["mean"] for s in sents]) if len(sents) > 1 else 0.0,
            "sentences": sents, "low": [s for s in sents if s["flagged"]], "threshold": {"mean": tm, "window": tw}}


def summary(r: dict | None) -> dict | None:
    return None if r is None else {k: round(r[k], 3) for k in ("mean", "perplexity", "burstiness")} | {"low": len(r["low"]), "n_tokens": r["n_tokens"]}
