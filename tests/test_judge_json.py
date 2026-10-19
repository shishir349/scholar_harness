from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import pytest

from harness import llm
from harness.text.jsonparse import parse_json
from harness.validate.judge import judge_pair


@pytest.fixture(autouse=True)
def restore_llm():
    yield
    llm.set_fake(None)


def fake(*texts):
    m = GenericFakeChatModel(messages=iter([AIMessage(content=t) for t in texts]))
    llm.set_fake(lambda tier: m)
    return m


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here: [1, 2, 3] hope that helps') == [1, 2, 3]
    assert parse_json('{"a": {"b": [1]}} trailing }') == {"a": {"b": [1]}}
    assert parse_json("no json") is None and parse_json(None) is None and parse_json("") is None


def test_text_of_handles_blocks():
    assert llm.text_of(AIMessage(content=[{"type": "text", "text": "hi "}, {"type": "tool_use", "id": "x"},
                                          {"type": "text", "text": "there"}])) == "hi there"


def test_judge_win_lose_tie_and_order_swap():
    # draft first: picks A (draft) ; draft second: picks B (draft) -> win
    fake('{"human":"A"}', '{"human":"B"}')
    assert judge_pair("draft", "ref", "introduction")["result"] == "win"
    # rejects draft both times
    fake('{"human":"B"}', '{"human":"A"}')
    assert judge_pair("draft", "ref", "introduction")["result"] == "lose"
    # position bias (always A) cancels to a tie
    fake('{"human":"A"}', '{"human":"A"}')
    assert judge_pair("draft", "ref", "introduction")["result"] == "tie"
    # unparseable -> tie
    fake("nonsense", '{"human":"B"}')
    assert judge_pair("draft", "ref", "introduction")["result"] == "tie"


def test_judge_puts_instructions_in_system_message_and_swaps_passages():
    seen = []

    class Rec(GenericFakeChatModel):
        def _generate(self, messages, *a, **k):
            seen.append(messages)
            return super()._generate(messages, *a, **k)

    m = Rec(messages=iter([AIMessage(content='{"human":"A"}'), AIMessage(content='{"human":"B"}')]))
    llm.set_fake(lambda tier: m)
    judge_pair("DRAFT-TEXT", "REF-TEXT", "methodology")
    assert seen[0][0].type == "system" and "human researcher" in seen[0][0].content
    first, second = seen[0][1].content, seen[1][1].content
    assert first.index("DRAFT-TEXT") < first.index("REF-TEXT")
    assert second.index("REF-TEXT") < second.index("DRAFT-TEXT")


def test_h2h_averages_repeats_and_keeps_a_wide_tie_zone():
    from harness.validate.judge import judge_h2h
    fake(*['{"human":"A"}', '{"human":"B"}'] * 3)                    # draft preferred in every order
    assert judge_h2h("d", "o", "introduction")["result"] == "win"
    fake(*['{"human":"B"}', '{"human":"A"}'] * 3)                    # other preferred in every order
    assert judge_h2h("d", "o", "introduction")["result"] == "lose"
    fake(*['{"human":"A"}', '{"human":"A"}'] * 3)                    # pure position bias cancels
    r = judge_h2h("d", "o", "introduction")
    assert r["result"] == "tie" and r["score"] == 0.5
    fake(*['{"human":"TIE"}', "garbage"] * 3)                        # ties and unusable replies count half
    assert judge_h2h("d", "o", "introduction")["score"] == 0.5


def test_forced_mode_has_no_tie_option():
    from harness.validate.judge import SYSTEM_H2H, SYSTEM_H2H_FORCED
    assert "TIE" in SYSTEM_H2H and "TIE" not in SYSTEM_H2H_FORCED and "must pick one" in SYSTEM_H2H_FORCED
