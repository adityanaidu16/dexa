"""Fast offline gate for the accuracy-vs-KV-memory dataset plumbing.

Covers:
  * :mod:`dexa.bench.qa_metrics` -- known SQuAD/LongBench cases (perfect / partial
    / invariance / empty), all offline.
  * :func:`dexa.bench.datasets.load_ruler` -- every RULER task yields well-shaped
    QAExamples, and NIAH answers actually appear in the context.
  * :func:`dexa.bench.datasets.load_longbench` -- guarded by importorskip and a
    graceful offline skip, so CI never hangs on the network.

Run: ``.venv/bin/python -m pytest tests/test_datasets_metrics.py -v``
"""

from __future__ import annotations

import pytest

from dexa.bench.datasets import (
    QAExample,
    list_datasets,
    load_ruler,
)
from dexa.bench.qa_metrics import normalize, score, substring_em, token_f1

_RULER_TASKS = ("niah_single", "niah_multikey", "multihop", "variable_tracking", "aggregation")


# --------------------------------------------------------------------------- #
# qa_metrics
# --------------------------------------------------------------------------- #
def test_metrics_perfect_match():
    s = score("the answer is 42", ["the answer is 42"])
    assert s["em"] == 1.0
    assert s["f1"] == pytest.approx(1.0)


def test_metrics_partial_overlap():
    f1 = token_f1("the quick brown fox", ["the slow brown dog"])
    assert 0.0 < f1 < 1.0
    # partial overlap should not count as a substring EM here.
    assert substring_em("the quick brown fox", ["the slow brown dog"]) == 0.0


def test_metrics_invariance_articles_case_punct():
    # articles, case and punctuation must not matter.
    assert normalize("The Answer.") == "answer"
    assert score("The Answer!", ["answer"]) == {"em": 1.0, "f1": pytest.approx(1.0)}
    assert token_f1("A red CAR.", ["red car"]) == pytest.approx(1.0)


def test_metrics_substring_em():
    # LongBench-style: gold is a substring of a longer prediction.
    assert substring_em("the capital of france is paris today", ["paris"]) == 1.0
    assert substring_em("berlin", ["paris"]) == 0.0


def test_metrics_max_over_golds():
    # takes the best of several acceptable golds.
    assert token_f1("paris", ["london", "paris"]) == pytest.approx(1.0)
    assert substring_em("paris", ["london", "paris"]) == 1.0


def test_metrics_empty_pred():
    assert score("", ["something"]) == {"em": 0.0, "f1": 0.0}
    assert token_f1("", ["something"]) == 0.0
    assert substring_em("", ["something"]) == 0.0


def test_metrics_empty_gold_never_matches():
    # an empty gold must not vacuously satisfy substring EM.
    assert substring_em("anything at all", [""]) == 0.0


# --------------------------------------------------------------------------- #
# load_ruler (offline, deterministic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", _RULER_TASKS)
def test_load_ruler_shapes(task):
    exs = load_ruler(task, length=200, n=5, seed=0)
    assert len(exs) == 5
    for ex in exs:
        assert isinstance(ex, QAExample)
        assert isinstance(ex.context, str) and ex.context.strip()
        assert isinstance(ex.question, str) and ex.question.strip()
        assert isinstance(ex.answers, list) and ex.answers
        assert all(isinstance(a, str) and a for a in ex.answers)
        assert ex.meta.get("task") == task


@pytest.mark.parametrize("task", ("niah_single", "niah_multikey"))
def test_load_ruler_niah_answer_in_context(task):
    # the planted answer must actually be findable in the context.
    for ex in load_ruler(task, length=300, n=5, seed=1):
        assert any(a in ex.context for a in ex.answers)


def test_load_ruler_deterministic():
    a = load_ruler("niah_single", length=200, n=3, seed=7)
    b = load_ruler("niah_single", length=200, n=3, seed=7)
    assert [x.context for x in a] == [x.context for x in b]
    assert [x.answers for x in a] == [x.answers for x in b]


def test_load_ruler_unknown_task():
    with pytest.raises(ValueError):
        load_ruler("does_not_exist")


def test_list_datasets_catalogue():
    cat = list_datasets()
    sources = {c["source"] for c in cat}
    assert {"longbench", "ruler"} <= sources
    subsets = {c["subset"] for c in cat}
    assert set(_RULER_TASKS) <= subsets


# --------------------------------------------------------------------------- #
# load_longbench (guarded: needs `datasets` + network)
# --------------------------------------------------------------------------- #
def test_load_longbench_shapes():
    pytest.importorskip("datasets")
    from dexa.bench.datasets import load_longbench

    try:
        exs = load_longbench("multifieldqa_en", n=3, max_context_chars=2000, seed=0)
    except (OSError, ConnectionError) as e:  # offline / HF hub unreachable
        pytest.skip(f"LongBench unavailable (offline?): {e}")
    except Exception as e:  # datasets wraps network errors in its own types
        if any(w in type(e).__name__.lower() for w in ("connection", "http", "offline", "timeout")):
            pytest.skip(f"LongBench unavailable (offline?): {e}")
        raise

    assert 1 <= len(exs) <= 3
    for ex in exs:
        assert isinstance(ex, QAExample)
        assert isinstance(ex.context, str) and ex.context
        assert len(ex.context) <= 2000
        assert isinstance(ex.question, str) and ex.question
        assert isinstance(ex.answers, list) and ex.answers
        assert ex.meta["source"] == "longbench"
