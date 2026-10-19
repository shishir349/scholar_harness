"""Per-run cost tracking with a hard budget. Rates mirror acdyon-scholar aiLogger.service.js (USD per 1K tokens)."""
from collections import defaultdict
from langchain_core.callbacks import BaseCallbackHandler

from . import config

# (input, output) per 1K tokens; unknown models fall back to the Sonnet rate
RATES = {
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.00025, 0.00125),
}
_DEFAULT_RATE = (0.003, 0.015)


class BudgetExceeded(RuntimeError):
    pass


class Spend(BaseCallbackHandler):
    """Attach as a LangChain callback; every LLM call in every node is counted."""

    # LangChain swallows callback exceptions unless told otherwise; the budget stop must propagate.
    raise_error = True

    def __init__(self, budget_usd: float | None = None):
        self.budget = config.budget_usd() if budget_usd is None else budget_usd
        self.by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "cache_read": 0, "cache_write": 0})

    @property
    def total(self) -> float:
        return sum(m["usd"] for m in self.by_model.values())

    def add(self, model: str, tokens_in: int, tokens_out: int, cache_read: int = 0, cache_write: int = 0) -> None:
        """tokens_in includes cached tokens (LangChain sums them). Cache reads bill at 0.1x input, writes at 1.25x."""
        rin, rout = RATES.get(model, _DEFAULT_RATE)
        m = self.by_model[model]
        m["calls"] += 1
        m["in"] += tokens_in
        m["out"] += tokens_out
        m["cache_read"] += cache_read
        m["cache_write"] += cache_write
        plain = tokens_in - cache_read - cache_write
        m["usd"] += (plain + 0.1 * cache_read + 1.25 * cache_write) / 1000 * rin + tokens_out / 1000 * rout
        if self.total > self.budget:
            raise BudgetExceeded(f"run spend ${self.total:.3f} exceeded budget ${self.budget:.2f}")

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                model = (getattr(msg, "response_metadata", None) or {}).get("model_name") or config.sonnet_model()
                det = usage.get("input_token_details") or {}
                # the API reports writes under cache_creation or, per TTL, ephemeral_5m/1h_input_tokens
                write = det.get("cache_creation", 0) or det.get("ephemeral_5m_input_tokens", 0) + det.get("ephemeral_1h_input_tokens", 0)
                self.add(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), det.get("cache_read", 0), write)

    def summary(self) -> dict:
        return {"total_usd": round(self.total, 5), "by_model": {k: dict(v) for k, v in self.by_model.items()}}
