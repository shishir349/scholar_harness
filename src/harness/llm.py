"""Model factory. All framework model construction lives here."""
from . import config

_fake_factory = None  # tests / HARNESS_FAKE_LLM swap in scripted models here


def set_fake(factory) -> None:
    """factory(tier: str) -> chat model, or None to restore the real one."""
    global _fake_factory
    _fake_factory = factory


def text_of(msg) -> str:
    """Plain text of a model message (content may be a string or a list of blocks)."""
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c if not isinstance(b, dict) or b.get("type") == "text")


def ask(tier: str, system: str, user: str, *, max_tokens: int = 1024, spend=None, temperature: float | None = None) -> str:
    """One system+user round trip, returns text. Instructions belong in `system`."""
    from langchain_core.messages import HumanMessage, SystemMessage

    model = chat(tier, max_tokens=max_tokens, spend=spend, temperature=temperature)
    return text_of(model.invoke([SystemMessage(content=system), HumanMessage(content=user)]))


def chat(tier: str, *, max_tokens: int = 1024, spend=None, temperature: float | None = None):
    """tier is 'sonnet' or 'haiku'. `spend` (a harness.spend.Spend) is attached as a callback."""
    if _fake_factory is not None:
        return _fake_factory(tier)
    from langchain_anthropic import ChatAnthropic

    model = config.sonnet_model() if tier == "sonnet" else config.haiku_model()
    extra = {} if temperature is None else {"temperature": temperature}
    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        **extra,
        api_key=config.get("ANTHROPIC_API_KEY") or None,
        callbacks=[spend] if spend is not None else None,
    )
