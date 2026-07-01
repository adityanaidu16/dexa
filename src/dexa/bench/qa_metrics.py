"""Standard extractive-QA metrics (SQuAD / LongBench style).

These operate on *strings* (not token ids), so they score real long-context QA
where the gold is a set of acceptable answer texts. They mirror the official
SQuAD normalization (lowercase, drop punctuation + articles + extra whitespace)
and the LongBench convention of substring exact-match for extractive answers.

  * :func:`normalize`     -- SQuAD answer normalization.
  * :func:`substring_em`  -- 1.0 if any gold is a substring of the prediction.
  * :func:`token_f1`      -- max over golds of SQuAD token-level F1.
  * :func:`score`         -- ``{"em": ..., "f1": ...}`` for a (pred, golds) pair.

Token-id F1 for the toy backend lives in :func:`dexa.bench.tasks.token_match`;
this module is its text-space analogue for the real datasets.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, drop articles (a/an/the), collapse spaces."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def substring_em(pred: str, golds: list[str]) -> float:
    """LongBench-style EM: 1.0 if any normalized gold is a substring of the
    normalized prediction, else 0.0. A non-empty gold is required to match."""
    npred = normalize(pred)
    for g in golds:
        ng = normalize(g)
        if ng and ng in npred:
            return 1.0
    return 0.0


def _f1_single(pred: str, gold: str) -> float:
    """SQuAD token-F1 between a single prediction and a single gold."""
    ptoks = normalize(pred).split()
    gtoks = normalize(gold).split()
    if not ptoks or not gtoks:
        # by SQuAD convention, F1 is 1 only if both are empty, else 0.
        return 1.0 if ptoks == gtoks else 0.0
    common = Counter(ptoks) & Counter(gtoks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(ptoks)
    rec = overlap / len(gtoks)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, golds: list[str]) -> float:
    """Max SQuAD token-F1 of the prediction over the acceptable golds."""
    if not golds:
        return 0.0
    return max(_f1_single(pred, g) for g in golds)


def score(pred: str, golds: list[str]) -> dict[str, float]:
    """Return ``{"em": substring_em, "f1": token_f1}`` for one (pred, golds)."""
    return {"em": substring_em(pred, golds), "f1": token_f1(pred, golds)}
