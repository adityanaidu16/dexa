"""Unified dataset loaders for the accuracy-vs-KV-memory benchmark.

``docs/BENCHMARK.md`` needs two families of long-context QA on one axis:

  * **LongBench** single-doc QA (NarrativeQA / Qasper / MultiFieldQA) -- real,
    standard, F1-scored, ~10-20k-token contexts. The credibility anchor. Loaded
    from HuggingFace ``datasets`` (network + the ``[bench]`` extra).
  * **RULER**-style controlled synthetic (NIAH variants, variable tracking,
    aggregation) -- a clean compression sweep with no network, reusing the
    text generators behind :mod:`dexa.bench.tasks`.

Both are mapped onto one :class:`QAExample` (text context + question + a list of
acceptable gold strings), scored by :mod:`dexa.bench.qa_metrics`. This is the
text-space sibling of :class:`dexa.bench.tasks.Task` (which carries token ids for
the toy backend); here everything stays as strings so the real datasets and the
string QA metrics line up.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Reuse the bench vocabulary / splicing so RULER contexts read like the rest of
# the suite and share the same token-frequency structure.
from dexa.bench.tasks import _FILLER_VOCAB, _KEYS, _filler, _splice  # noqa: F401


@dataclass
class QAExample:
    """One long-context QA instance.

    ``context`` is the document to be cached/compacted, ``question`` is asked
    against it, and ``answers`` is the list of acceptable gold strings (any one
    counts as correct under the substring-EM / max-F1 metrics).
    """

    context: str
    question: str
    answers: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# LongBench (real, HuggingFace `datasets`)
# --------------------------------------------------------------------------- #
# Single-doc QA subsets that map cleanly onto (context, input, answers).
_LONGBENCH_SUBSETS = ("narrativeqa", "qasper", "multifieldqa_en")


def load_longbench(
    subset: str,
    n: int = 50,
    max_context_chars: int | None = None,
    seed: int = 0,
) -> list[QAExample]:
    """Load ``n`` deterministically-subsampled examples from ``THUDM/LongBench``.

    ``subset`` is a LongBench config name such as ``"narrativeqa"``, ``"qasper"``
    or ``"multifieldqa_en"``. Each row maps ``context=row["context"]``,
    ``question=row["input"]``, ``answers=row["answers"]``. ``datasets`` is
    imported lazily so the synthetic path (and the rest of the suite) never
    depends on it; ``max_context_chars`` optionally truncates the context.
    """
    try:
        import datasets  # noqa: F401  (presence check; sub-imports used below)
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "load_longbench requires the 'datasets' package. "
            "Install it with: pip install -e '.[bench]'  (or: pip install datasets)"
        ) from e

    rows = _load_longbench_rows(subset)

    # deterministic subsample of row indices.
    n = min(n, len(rows))
    rng = random.Random(hash(("longbench", subset, n, seed)) & 0xFFFFFFFF)
    idxs = sorted(rng.sample(range(len(rows)), k=n))

    out: list[QAExample] = []
    for i in idxs:
        row = rows[i]
        context = row["context"]
        if max_context_chars is not None and len(context) > max_context_chars:
            context = context[:max_context_chars]
        answers = [str(a) for a in row["answers"]]
        out.append(
            QAExample(
                context=context,
                question=row["input"],
                answers=answers,
                meta={
                    "source": "longbench",
                    "subset": subset,
                    "row": i,
                    "id": row.get("_id"),
                    "length": row.get("length"),
                },
            )
        )
    return out


def _load_longbench_rows(subset: str) -> list[dict[str, Any]]:
    """Fetch the raw LongBench rows for ``subset`` as a list of dicts.

    Prefers the standard ``datasets.load_dataset`` path; recent ``datasets``
    releases dropped support for hub *loading scripts* (LongBench ships one), so
    on that specific ``RuntimeError`` we fall back to reading the repo's
    ``data.zip`` (``data/{subset}.jsonl``) directly via ``huggingface_hub``.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("THUDM/LongBench", subset, split="test")
        return [dict(r) for r in ds]
    except RuntimeError as e:
        if "script" not in str(e).lower():
            raise
    # script-free fallback: read the jsonl straight out of data.zip.
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        with z.open(f"data/{subset}.jsonl") as f:
            return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# RULER-style synthetic (no network) -- text generators returning QAExample
# --------------------------------------------------------------------------- #
def _niah_single(length: int, rng: random.Random, depth: float = 0.5) -> QAExample:
    """One needle in a haystack of filler. Ask for its magic number."""
    key = rng.choice(_KEYS)
    value = str(rng.randint(1_000_000, 9_999_999))
    needle = f"The magic number for {key} is {value} ."
    context = " ".join(_splice(_filler(length, rng), [(depth, needle)]))
    return QAExample(
        context=context,
        question=f"What is the magic number for {key} ?",
        answers=[value],
        meta={"task": "niah_single", "length": length, "key": key, "depth": depth},
    )


def _niah_multikey(length: int, rng: random.Random, n_keys: int = 4) -> QAExample:
    """Several needles with distinct keys; ask for one. Tests selectivity."""
    keys = rng.sample(_KEYS, k=min(n_keys, len(_KEYS)))
    pairs = {k: str(rng.randint(1_000_000, 9_999_999)) for k in keys}
    inserts = [
        ((i + 0.5) / len(pairs), f"The magic number for {k} is {v} .")
        for i, (k, v) in enumerate(pairs.items())
    ]
    context = " ".join(_splice(_filler(length, rng), inserts))
    target = rng.choice(keys)
    return QAExample(
        context=context,
        question=f"What is the magic number for {target} ?",
        answers=[pairs[target]],
        meta={"task": "niah_multikey", "length": length, "n_keys": len(keys), "key": target},
    )


def _multihop(length: int, rng: random.Random, hops: int = 4) -> QAExample:
    """Chained assignment X1=v; X2=X1; ...; ask the final value (RULER-style)."""
    value = str(rng.randint(10, 99))
    chain = [f"X1 = {value} ."] + [f"X{i} = X{i - 1} ." for i in range(2, hops + 1)]
    inserts = [((i + 0.5) / len(chain), s) for i, s in enumerate(chain)]
    context = " ".join(_splice(_filler(length, rng), inserts))
    return QAExample(
        context=context,
        question=f"What is the value of X{hops} ?",
        answers=[value],
        meta={"task": "multihop", "length": length, "hops": hops},
    )


def _variable_tracking(length: int, rng: random.Random, hops: int = 4) -> QAExample:
    """RULER variable tracking: a value flows through a chain of aliases; find
    every variable that ends up holding it. Answers are the alias names, all of
    which appear verbatim in the context (so any one counts under substring-EM)."""
    value = str(rng.randint(10_000, 99_999))
    names = [f"VAR_{k.upper()}" for k in rng.sample(_KEYS, k=min(hops, len(_KEYS)))]
    chain = [f"{names[0]} = {value} ."]
    chain += [f"{names[i]} = {names[i - 1]} ." for i in range(1, len(names))]
    inserts = [((i + 0.5) / len(chain), s) for i, s in enumerate(chain)]
    context = " ".join(_splice(_filler(length, rng), inserts))
    return QAExample(
        context=context,
        question=f"Which variables are assigned the value {value} ?",
        answers=list(names),
        meta={"task": "variable_tracking", "length": length, "value": value, "names": names},
    )


def _aggregation(length: int, rng: random.Random, n_targets: int = 3) -> QAExample:
    """RULER frequent-words aggregation: a few rare marker words are repeated
    more often than any filler word; ask for the most frequent words. Answers
    are the markers (which are, by construction, the top-``n_targets``)."""
    pool = ["aardvark", "zeppelin", "quokka", "xylophone", "wombat", "kumquat", "narwhal"]
    targets = rng.sample(pool, k=min(n_targets, len(pool)))

    filler = _filler(length, rng)
    max_filler = max(Counter(filler).values()) if filler else 0
    rep = max_filler + 5  # each marker strictly beats every filler word.

    words = list(filler)
    for t in targets:
        words += [t] * rep
    rng.shuffle(words)  # spread markers throughout the context.
    context = " ".join(words)
    return QAExample(
        context=context,
        question=f"What are the {len(targets)} most frequently repeated words in the text above ?",
        answers=list(targets),
        meta={"task": "aggregation", "length": length, "targets": targets, "count": rep},
    )


_RULER_TASKS = {
    "niah_single": _niah_single,
    "niah_multikey": _niah_multikey,
    "multihop": _multihop,
    "variable_tracking": _variable_tracking,
    "aggregation": _aggregation,
}


def load_ruler(
    task: str = "niah_single",
    length: int = 4000,
    n: int = 20,
    seed: int = 0,
) -> list[QAExample]:
    """Build ``n`` controlled synthetic RULER-style :class:`QAExample` instances.

    ``task`` is one of ``niah_single``, ``niah_multikey``, ``multihop``,
    ``variable_tracking``, ``aggregation``. ``length`` is the filler length (the
    needle/fact sentences are spliced on top, so the realized context is a bit
    longer). No network is needed; every example is deterministic in ``seed``.
    """
    if task not in _RULER_TASKS:
        raise ValueError(f"unknown RULER task {task!r}; choose from {sorted(_RULER_TASKS)}")
    gen = _RULER_TASKS[task]
    out: list[QAExample] = []
    for s in range(n):
        rng = random.Random(hash((task, length, seed, s)) & 0xFFFFFFFF)
        out.append(gen(length, rng))
    return out


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def list_datasets() -> list[dict[str, str]]:
    """Describe the available ``(source, subset)`` pairs for the benchmark."""
    catalogue: list[dict[str, str]] = []
    for sub in _LONGBENCH_SUBSETS:
        catalogue.append(
            {
                "source": "longbench",
                "subset": sub,
                "loader": "load_longbench",
                "network": "yes",
                "metric": "substring_em / token_f1",
            }
        )
    for task in _RULER_TASKS:
        catalogue.append(
            {
                "source": "ruler",
                "subset": task,
                "loader": "load_ruler",
                "network": "no",
                "metric": "substring_em / token_f1",
            }
        )
    return catalogue
