"""Workload generators for the Dexa benchmark.

Each generator takes a :class:`~dexa.engine.base.ModelBackend` (used only as a
tokenizer here) and returns a :class:`Task`: a context to be cached/compacted, a
prompt/question, the gold answer token ids, and a scorer. The same generators
produce valid token-id structures for the toy :class:`FakeBackend` (so plumbing
and attention-reconstruction run) and for a real HF backend (so answer accuracy
is meaningful).

Tasks are RULER-style long-context probes:
    * ``niah_single``   needle-in-a-haystack: one planted magic number.
    * ``niah_multikey`` several needles, ask one (tests selectivity).
    * ``multihop``      variable tracking X1=7; X2=X1; ... ask the last.
    * ``synthetic_qa``  a few facts + a question.

``context length`` is expressed in *filler* tokens; the needle/fact sentences
are spliced in on top, so the realized context is a bit longer (recorded in
``meta``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

# A small, repetitive filler vocabulary. Repetition gives the toy backend a real
# token-frequency structure, so a selective compactor (attention matching) has
# something to be selective *about* relative to random selection.
_FILLER_VOCAB = (
    "the quick brown fox jumps over a lazy dog while the calm river flows past "
    "green hills under a wide blue sky as birds sing and the warm wind moves "
    "softly through the tall old trees near the quiet little town"
).split()

_KEYS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
]


# --- scoring ---------------------------------------------------------------
def token_match(pred_ids: list[int], gold_ids: list[int]) -> dict[str, float]:
    """Exact-match (is gold a contiguous run inside pred) plus token-set F1."""
    if not gold_ids:
        return {"exact_match": 0.0, "token_f1": 0.0}

    # exact match: gold appears as a contiguous subsequence of pred
    exact = 0.0
    g = gold_ids
    for i in range(0, max(0, len(pred_ids) - len(g) + 1)):
        if pred_ids[i : i + len(g)] == g:
            exact = 1.0
            break

    # token-set F1 (multiset overlap)
    from collections import Counter

    pc, gc = Counter(pred_ids), Counter(gold_ids)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        f1 = 0.0
    else:
        prec = overlap / max(1, len(pred_ids))
        rec = overlap / max(1, len(gold_ids))
        f1 = 2 * prec * rec / (prec + rec)
    return {"exact_match": exact, "token_f1": float(f1)}


@dataclass
class Task:
    """A single benchmark instance.

    ``context_ids`` is prefilled + compacted; ``prompt_ids`` is the question
    decoded against the (possibly compact) context; ``gold_ids`` is the answer.
    """

    name: str
    context_ids: list[int]
    prompt_ids: list[int]
    gold_ids: list[int]
    meta: dict[str, Any] = field(default_factory=dict)
    scorer: Callable[[list[int], list[int]], dict[str, float]] = token_match


def _filler(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(_FILLER_VOCAB) for _ in range(max(0, n))]


def _splice(words: list[str], inserts: list[tuple[int, str]]) -> list[str]:
    """Insert ``(fraction, sentence)`` items into ``words`` at fractional spots."""
    out = list(words)
    # insert from the back so earlier indices stay valid
    for frac, sentence in sorted(inserts, key=lambda x: -x[0]):
        pos = min(len(out), max(0, int(frac * len(out))))
        out[pos:pos] = sentence.split()
    return out


# --- generators ------------------------------------------------------------
def niah_single(backend, length: int = 256, *, seed: int = 0, depth: float = 0.5) -> Task:
    """One needle in a haystack of filler. Ask for its magic number."""
    rng = random.Random(hash(("niah_single", length, seed)))
    key = rng.choice(_KEYS)
    value = str(rng.randint(1_000_000, 9_999_999))
    needle = f"The magic number for {key} is {value} ."
    words = _splice(_filler(length, rng), [(depth, needle)])
    context = " ".join(words)
    question = f"What is the magic number for {key} ?"
    return Task(
        name="niah_single",
        context_ids=backend.tokenize(context),
        prompt_ids=backend.tokenize(question),
        gold_ids=backend.tokenize(value),
        meta={"length": length, "seed": seed, "key": key, "value": value, "depth": depth},
    )


def niah_multikey(
    backend, length: int = 256, *, seed: int = 0, n_keys: int = 4
) -> Task:
    """Several needles with distinct keys; ask for one. Tests selectivity."""
    rng = random.Random(hash(("niah_multikey", length, seed)))
    keys = rng.sample(_KEYS, k=min(n_keys, len(_KEYS)))
    pairs = {k: str(rng.randint(1_000_000, 9_999_999)) for k in keys}
    inserts = []
    for i, (k, v) in enumerate(pairs.items()):
        depth = (i + 0.5) / len(pairs)
        inserts.append((depth, f"The magic number for {k} is {v} ."))
    words = _splice(_filler(length, rng), inserts)
    context = " ".join(words)
    target_key = rng.choice(keys)
    question = f"What is the magic number for {target_key} ?"
    return Task(
        name="niah_multikey",
        context_ids=backend.tokenize(context),
        prompt_ids=backend.tokenize(question),
        gold_ids=backend.tokenize(pairs[target_key]),
        meta={
            "length": length,
            "seed": seed,
            "n_keys": len(keys),
            "target_key": target_key,
            "value": pairs[target_key],
        },
    )


def multihop(backend, length: int = 256, *, seed: int = 0, hops: int = 4) -> Task:
    """Variable tracking: X1=v; X2=X1; ...; ask the final value (RULER-style)."""
    rng = random.Random(hash(("multihop", length, seed)))
    value = str(rng.randint(10, 99))
    chain = [f"X1 = {value} ."]
    for i in range(2, hops + 1):
        chain.append(f"X{i} = X{i - 1} .")
    inserts = [((i + 0.5) / len(chain), s) for i, s in enumerate(chain)]
    words = _splice(_filler(length, rng), inserts)
    context = " ".join(words)
    question = f"What is the value of X{hops} ?"
    return Task(
        name="multihop",
        context_ids=backend.tokenize(context),
        prompt_ids=backend.tokenize(question),
        gold_ids=backend.tokenize(value),
        meta={"length": length, "seed": seed, "hops": hops, "value": value},
    )


def synthetic_qa(backend, length: int = 256, *, seed: int = 0) -> Task:
    """A handful of facts buried in filler, plus a question over one of them."""
    rng = random.Random(hash(("synthetic_qa", length, seed)))
    people = ["alice", "bob", "carol", "dave"]
    cities = ["paris", "tokyo", "cairo", "oslo", "lima", "delhi"]
    rng.shuffle(cities)
    facts = {p: cities[i] for i, p in enumerate(people)}
    inserts = [
        ((i + 0.5) / len(facts), f"{p} lives in {c} .")
        for i, (p, c) in enumerate(facts.items())
    ]
    words = _splice(_filler(length, rng), inserts)
    context = " ".join(words)
    who = rng.choice(people)
    question = f"Where does {who} live ?"
    return Task(
        name="synthetic_qa",
        context_ids=backend.tokenize(context),
        prompt_ids=backend.tokenize(question),
        gold_ids=backend.tokenize(facts[who]),
        meta={"length": length, "seed": seed, "who": who, "answer": facts[who]},
    )


_GENERATORS = {
    "niah_single": niah_single,
    "niah_multikey": niah_multikey,
    "multihop": multihop,
    "synthetic_qa": synthetic_qa,
}


def make_tasks(
    backend,
    lengths: list[int] = (256, 1024),
    n_per: int = 2,
    names: list[str] | None = None,
) -> list[Task]:
    """Build a flat list of tasks across generators × lengths × ``n_per`` seeds."""
    names = list(names) if names is not None else list(_GENERATORS)
    tasks: list[Task] = []
    for name in names:
        gen = _GENERATORS[name]
        for length in lengths:
            for s in range(n_per):
                tasks.append(gen(backend, length=length, seed=s))
    return tasks
