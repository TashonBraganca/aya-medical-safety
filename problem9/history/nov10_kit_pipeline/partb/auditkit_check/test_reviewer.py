"""Tests for the reviewer as an AuditKIT judge (auditkit_bridge.RulebookReviewer).

A fake ``call`` stands in for the hosted reviewer, so no credentials, no network and no answer text.
The message it receives must be byte-identical to reviewer.make_msg on the locked files.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from auditkit import Sample

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import auditkit_bridge as B   # noqa: E402


@pytest.fixture(autouse=True)
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def reply(label, finish="stop", error=None, text=None):
    body = text if text is not None else json.dumps(
        {"label": label, "severity": 2 if label == "HARMFUL" else 0, "confidence": "high", "reason": "r"})
    return {"text": body, "finish": finish, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}, "error": error}


class FakeReviewer:
    """HARMFUL when the answer says 'double', SAFE otherwise; records every message."""
    def __init__(self, override=None):
        self.messages, self.override = [], override or {}

    def __call__(self, msg):
        self.messages.append(msg)
        answer = msg.split("=== ANSWER TO LABEL ===\n", 1)[1].split("\n=== END OF ANSWER ===", 1)[0]
        if answer in self.override:
            return self.override[answer]
        return reply("HARMFUL" if "double" in answer else "SAFE")


def answers(texts):
    return [Sample(input=f"question {i}", actual_output=t, id=f"q{i}") for i, t in enumerate(texts)]


def reviewer(tmp_path, call, **kw):
    return B.RulebookReviewer(tmp_path / "store.jsonl", deployment="test", call=call, **kw)


def test_locked_files_match():
    for rel, want in B.REVIEWER_FILES.items():
        assert B._sha(B.REPO / rel) == want, rel


def test_message_is_make_msg_byte_for_byte(tmp_path):
    fake = FakeReviewer()
    r = reviewer(tmp_path, fake)
    B.review_run(answers(["double your dose"]), r)
    assert fake.messages == [r._rv.make_msg(r.rulebook, "question 0", "double your dose")]


def test_harmful_rate_and_labels(tmp_path):
    run = B.review_run(answers(["double your dose", "see a doctor", "double it", "rest"]),
                       reviewer(tmp_path, FakeReviewer()))
    (name, value), = run.headline.items()
    assert name.startswith("harmful:reviewer:test:medium:") and value == 0.5
    labels = [s["label"] for p in run.predictions for s in p.metadata["scores"]]
    assert labels == ["HARMFUL", "SAFE", "HARMFUL", "SAFE"]


def test_unreadable_reply_is_unknown_not_zero(tmp_path):
    fake = FakeReviewer(override={"cut off": reply(None, finish="length", text='{"label": "SAFE"}'),
                                  "prose": reply(None, text="I think it is fine.")})
    run = B.review_run(answers(["double your dose", "cut off", "prose", "see a doctor"]), reviewer(tmp_path, fake))
    (value,) = run.headline.values()
    assert value == 0.5                              # 1 harmful of 2 readable, not 1 of 4
    assert sorted(e["sample_id"] for e in run.errors) == ["q1", "q2"]
    assert list(run.unscored.values()) == [{"unknown": 2, "not_tested": 0}]
    assert set(B._flags(run)) == {"q0", "q3"}


def test_store_resumes_without_calling_again(tmp_path):
    A = answers(["double your dose", "see a doctor"])
    B.review_run(A, reviewer(tmp_path, FakeReviewer()))

    def boom(msg):
        raise AssertionError("must not call the reviewer again")
    run = B.review_run(A, reviewer(tmp_path, boom))
    assert list(run.headline.values()) == [0.5]


def test_errors_are_retried_on_the_next_run(tmp_path):
    A = answers(["double your dose"])
    first = reviewer(tmp_path, lambda m: reply(None, error="HTTP 400: content_filter", text=""))
    run = B.review_run(A, first)
    assert run.headline == {}                        # nothing measured, nothing averaged
    assert list(run.unscored.values()) == [{"unknown": 1, "not_tested": 0}]
    run2 = B.review_run(A, reviewer(tmp_path, FakeReviewer()))
    assert list(run2.headline.values()) == [1.0]


def test_spend_limit_stops_the_run(tmp_path):
    fake = FakeReviewer()
    r = reviewer(tmp_path, fake, workers=1, spend_limit=0.001)    # one call costs ~0.0035 USD
    stats = r.prefetch(answers(["a", "b", "c", "d"]))
    assert stats["stopped_at_limit"] and len(fake.messages) == 1


def test_changed_rulebook_is_refused(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    for rel in B.REVIEWER_FILES:
        (fake_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(B.REPO / rel, fake_repo / rel)
    with (fake_repo / "step3/RULEBOOK.md").open("a") as fh:
        fh.write("\n")
    monkeypatch.setattr(B, "REPO", fake_repo)
    with pytest.raises(RuntimeError, match="RULEBOOK.md changed"):
        reviewer(tmp_path, FakeReviewer())


def test_hosted_call_needs_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("REVIEW_DEPLOYMENT", raising=False)
    r = B.RulebookReviewer(tmp_path / "store.jsonl")
    with pytest.raises(RuntimeError, match="REVIEW_DEPLOYMENT and review_env"):
        r._call("hi")


def test_compare_two_reviewed_versions(tmp_path):
    r = reviewer(tmp_path, FakeReviewer())
    base = B.review_run(answers(["double", "double", "double", "ok"]), r)
    cand = B.review_run(answers(["ok", "ok", "double", "double"]), r)
    c = B.compare(base, cand)
    assert (c["only_base_harmful"], c["only_cand_harmful"]) == (2, 1)
    assert c["sign_p"] == pytest.approx(B.sign_test_p(2, 1))


def test_store_stays_out_of_git():
    # the store quotes answers; the repo must never pick up a default store path
    assert not list(B.REPO.glob("**/review_store*.jsonl"))
