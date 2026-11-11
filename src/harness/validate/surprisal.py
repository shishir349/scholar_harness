"""Local token surprisal / perplexity: how predictable each sentence is to a small language model (distilgpt2, int8 ONNX).

Machine prose is predictable (low mean surprisal, little spread between sentences); human prose is not. Zero API cost.
No torch/transformers: onnxruntime + tokenizers + numpy only, imported lazily, so the server starts without any of them.

Resources (measured): ~130MB for the session + ~35MB working at WINDOW=96, about 2ms per token on 2 threads (a 1200-word section
takes ~3-4s, so call it off the request's critical path and never in a loop). Text is scored per paragraph in windows of <= WINDOW tokens. `get_scorer()` is None when HARNESS_SURPRISAL=off, the model files are missing or onnxruntime is absent;
every caller treats None as "skip this step".

Files in HARNESS_SURPRISAL_DIR (default data/models): lm-q.onnx (Xenova/distilgpt2 onnx/decoder_model_quantized.onnx), tokenizer.json."""
import math
import os
import re
import statistics
import threading
from pathlib import Path

from .. import config, limits
from ..text.sentences import split_sentences

WINDOW = 96
BOS = 50256
_SCORER = None
_LOCK = threading.Lock()


def model_dir() -> Path:
    return Path(config.get("HARNESS_SURPRISAL_DIR", "") or Path(__file__).resolve().parents[3] / "data" / "models")


class OnnxScorer:
    def __init__(self, onnx_path: Path, tok_path: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False        # the arena keeps ~200MB of logits buffers alive
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(tok_path))
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def token_logprobs(self, text: str) -> list[tuple[int, int, float]]:
        """[(char_start, char_end, ln p(token | previous tokens))] for every token of `text`, in windows."""
        import numpy as np
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offs, out = enc.ids, enc.offsets, []
        for a in range(0, len(ids), WINDOW):
            part = [BOS] + ids[a:a + WINDOW]
            feed = {"input_ids": np.array([part], dtype=np.int64)}
            if "attention_mask" in self.inputs:
                feed["attention_mask"] = np.ones((1, len(part)), dtype=np.int64)
            logits = self.sess.run(None, feed)[0][0, :-1].astype(np.float32, copy=False)      # position t predicts token t+1
            logits = logits - logits.max(axis=-1, keepdims=True)
            lse = np.log(np.exp(logits).sum(axis=-1))
            tgt = np.array(part[1:])
            lp = logits[np.arange(len(tgt)), tgt] - lse
            out += [(offs[a + k][0], offs[a + k][1], float(lp[k])) for k in range(len(tgt))]
            del logits
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


def low_threshold() -> float:
    """Mean surprisal (nats/token) under which a sentence reads as machine-predictable. data/surprisal_baseline.json (made by
    scripts/calibrate_surprisal.py from the human corpus) overrides the default."""
    try:
        import json
        p = Path(__file__).resolve().parents[3] / "data" / "surprisal_baseline.json"
        return float(json.loads(p.read_text(encoding="utf-8"))["low"])
    except Exception:  # noqa: BLE001
        return limits.SURPRISAL_LOW


def score_section_surprisal(text: str, scorer=None, low: float | None = None) -> dict | None:
    """-> {mean, perplexity, burstiness, n_tokens, sentences:[{text, mean, para}], low:[...]} or None (no scorer / scorer failed).
    mean = average surprisal in nats per token; burstiness = std of the sentence means (human text varies more)."""
    scorer = scorer if scorer is not None else get_scorer()
    if scorer is None or not (text or "").strip():
        return None
    low = low_threshold() if low is None else low
    sents, tot, n = [], 0.0, 0
    try:
        for i, para in enumerate(q.strip() for q in re.split(r"\n\s*\n", text) if q.strip()):
            if para.startswith("#") or re.search(r"\[(?:FIGURE|TABLE)_START\]|^\s*[|!<]", para, re.M):
                continue
            toks = scorer.token_logprobs(para)
            pos = 0
            for s in split_sentences(para):
                a = para.find(s, pos)
                if a < 0:
                    continue
                pos = a + len(s)
                lps = [-lp for (x, y, lp) in toks if x >= a and y <= pos]
                if lps:
                    sents.append({"text": s, "mean": sum(lps) / len(lps), "para": i, "n": len(lps)})
                    tot, n = tot + sum(lps), n + len(lps)
    except Exception:  # noqa: BLE001
        return None
    if not sents:
        return None
    mean = tot / n
    return {"mean": mean, "perplexity": math.exp(mean), "n_tokens": n,
            "burstiness": statistics.pstdev([s["mean"] for s in sents]) if len(sents) > 1 else 0.0,
            "sentences": sents, "low": [s for s in sents if s["n"] >= 4 and s["mean"] < low], "threshold": low}


def summary(r: dict | None) -> dict | None:
    return None if r is None else {k: round(r[k], 3) for k in ("mean", "perplexity", "burstiness")} | {"low": len(r["low"]), "n_tokens": r["n_tokens"]}
