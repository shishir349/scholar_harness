import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from harness import llm
from harness.sources import verifier as V

REFS = [
    {"authors": "Smith, J.", "year": 2019, "title": "T1", "abstract": "Firms that adopted cloud tools grew faster."},
    {"authors": "Lee, K.", "year": 2020, "title": "T2", "abstract": None},            # no abstract: cannot be checked
    {"authors": "Brown, A.", "year": 2021, "title": "T3", "abstract": "A qualitative study of managers."},
]
TEXT = ("Cloud adoption boosted growth (Smith, 2019). Lee (2020) disagreed. Managers felt tired (Brown, 2021). "
        "An uncited sentence stands alone. Another claim (Ghost, 2015).")


@pytest.fixture(autouse=True)
def restore():
    yield
    llm.set_fake(None)


def script(*payloads):
    class Rec(GenericFakeChatModel):
        pass
    m = Rec(messages=iter([AIMessage(content=p if isinstance(p, str) else json.dumps(p)) for p in payloads]))
    llm.set_fake(lambda tier: m)
    return m


def test_only_refs_with_abstracts_produce_claims():
    claims = V.find_cited_claims(TEXT, REFS)
    assert [c["cite"] for c in claims] == ["(Smith, 2019)", "(Brown, 2021)"]
    assert [c["id"] for c in claims] == [1, 2]


def test_claim_cap_spreads_evenly():
    many = " ".join(f"Claim number {i} holds (Smith, 2019)." for i in range(100))
    # identical (author, year) is deduped per sentence, so each sentence still yields one claim
    claims = V.find_cited_claims(many, REFS, cap=10)
    assert len(claims) == 10
    assert "0 holds" in claims[0]["sentence"] and int(claims[-1]["sentence"].split()[2]) >= 85


def test_verdict_mapping_and_invalid_verdict_becomes_cannot_tell():
    script([{"id": 1, "verdict": "SUPPORTED", "reason": "stated"}, {"id": 2, "verdict": "MAYBE", "reason": "?"}])
    claims = V.find_cited_claims(TEXT, REFS)
    out = V.verify_claims(claims)
    assert [o["verdict"] for o in out] == ["SUPPORTED", "CANNOT_TELL"]


def test_malformed_json_batch_is_cannot_tell():
    script("total nonsense")
    out = V.verify_claims(V.find_cited_claims(TEXT, REFS))
    assert [o["verdict"] for o in out] == ["CANNOT_TELL", "CANNOT_TELL"]


def test_model_error_is_cannot_tell_but_budget_error_propagates():
    class Down(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("down")
    llm.set_fake(lambda tier: Down(messages=iter([])))
    out = V.verify_claims(V.find_cited_claims(TEXT, REFS))
    assert all(o["verdict"] == "CANNOT_TELL" for o in out)

    from harness.spend import BudgetExceeded

    class Over(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise BudgetExceeded("over")
    llm.set_fake(lambda tier: Over(messages=iter([])))
    with pytest.raises(BudgetExceeded):
        V.verify_claims(V.find_cited_claims(TEXT, REFS))


def test_batches_of_twelve_each_reach_the_model():
    refs = [{"authors": "Smith, J.", "year": 2019, "abstract": "abs"}]
    text = " ".join(f"Sentence {i} says something (Smith, 2019)." for i in range(30))
    m = script(*[[{"id": i, "verdict": "SUPPORTED", "reason": ""} for i in range(b * 12 + 1, min(30, b * 12 + 12) + 1)]
                 for b in range(3)])
    claims = V.find_cited_claims(text, refs)
    out = V.verify_claims(claims)
    assert len(claims) == 30 and all(o["verdict"] == "SUPPORTED" for o in out)   # 3 batches (12+12+6), each answered


def test_audit_chapter_stats():
    script([{"id": 1, "verdict": "UNSUPPORTED", "reason": "different topic"}, {"id": 2, "verdict": "SUPPORTED", "reason": ""}])
    a = V.audit_chapter(TEXT, REFS)
    s = a["stats"]
    assert s["checked"] == 2 and s["unsupported"] == 1 and s["supported"] == 1 and s["unsupported_rate"] == 0.5
    assert s["unmatched_count"] == 1 and a["unmatched"][0]["author"] == "Ghost"
    assert a["claim_checks"][0]["verdict"] == "UNSUPPORTED" and a["claim_checks"][0]["cite"] == "(Smith, 2019)"
    no_llm = V.audit_chapter(TEXT, REFS, verify=False)
    assert no_llm["claim_checks"] == [] and no_llm["stats"]["unsupported_rate"] == 0.0
