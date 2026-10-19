import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness.spend import Spend, BudgetExceeded


def _msg(text, tin, tout, model="claude-haiku-4-5-20251001"):
    return AIMessage(content=text, usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout},
                     response_metadata={"model_name": model})


def test_cost_maths_haiku_and_sonnet():
    s = Spend(budget_usd=10)
    s.add("claude-haiku-4-5-20251001", 1000, 1000)   # 0.00025 + 0.00125
    s.add("claude-sonnet-4-6", 1000, 1000)           # 0.003 + 0.015
    assert s.total == pytest.approx(0.0195)
    assert s.summary()["by_model"]["claude-sonnet-4-6"]["calls"] == 1


def test_unknown_model_uses_sonnet_rate():
    s = Spend(budget_usd=10)
    s.add("mystery", 1000, 0)
    assert s.total == pytest.approx(0.003)


def test_budget_exceeded_raises():
    s = Spend(budget_usd=0.001)
    with pytest.raises(BudgetExceeded):
        s.add("claude-sonnet-4-6", 1000, 1000)


def test_callback_counts_every_call_and_budget_propagates():
    s = Spend(budget_usd=10)
    m = GenericFakeChatModel(messages=iter([_msg("a", 100, 10), _msg("b", 200, 20)]), callbacks=[s])
    m.invoke("x"); m.invoke("y")
    assert s.by_model["claude-haiku-4-5-20251001"]["calls"] == 2
    assert s.by_model["claude-haiku-4-5-20251001"]["in"] == 300

    tight = Spend(budget_usd=0.0000001)
    m2 = GenericFakeChatModel(messages=iter([_msg("a", 1000, 1000)]), callbacks=[tight])
    with pytest.raises(BudgetExceeded):
        m2.invoke("x")


def test_cache_tokens_bill_at_read_and_write_rates():
    s = Spend(budget_usd=10)
    s.add("claude-sonnet-4-6", 3000, 0, cache_read=1000, cache_write=1000)   # 1000 plain + 100 + 1250 = 2350 token-equivalents
    assert s.total == pytest.approx(2.35 * 0.003)
