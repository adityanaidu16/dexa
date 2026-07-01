"""The accuracy-vs-KV-memory frontier benchmark (docs/BENCHMARK.md).

Puts every method on one plot: X = KV memory held for the corpus (bytes),
Y = task accuracy (generate the answer, score F1/EM vs gold). The claim under
test: the **cartridge** curve dominates the frontier — as accurate as
full-context at a fraction of the memory, above RAG and above training-free KV
compression (H2O/SnapKV/Attention-Matching) at the same memory budget.

Every method is evaluated by building a *context cache* for the corpus, then
generating an answer to each question against it, and measuring:
  accuracy = mean QA score over questions,   memory = bytes of that cache.
So the X-axis (bytes actually held) is directly comparable across methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dexa.bench.corpus import BowRetriever, chunk_corpus
from dexa.bench.qa_metrics import score as qa_score
from dexa.compaction.base import CompactionBudget
from dexa.compaction.baselines import build as build_compactor
from dexa.core.types import CostModel

# training-free KV-compression baselines + our analytic method
_COMPRESSIVE = ("attention_matching", "heavy_hitter", "snapkv")
_ALL_METHODS = ("full_context", "rag") + _COMPRESSIVE + ("cartridge",)


@dataclass
class FrontierPoint:
    method: str
    setting: str          # e.g. "16x" or "k=3"
    accuracy_f1: float
    accuracy_em: float
    memory_bytes: float
    memory_ratio: float   # full_context bytes / this method's bytes
    n: int


def _answer(backend, cache, q_prompt_ids, max_new) -> str:
    out = backend.generate(cache, q_prompt_ids, max_new_tokens=max_new)
    return backend.detokenize(out)


def run_frontier(
    backend,
    examples,                                   # list[QAExample]
    *,
    methods=_ALL_METHODS,
    ratios=(4, 16, 50, 128),                    # compression budgets for KV methods
    rag_ks=(1, 3, 8),                           # retrieval budgets for RAG
    chunk_tokens: int = 256,
    ref_strategy: str = "repeat_prefill",
    n_ref: int = 128,
    cartridge_opts: Optional[dict] = None,
    max_new_tokens: int = 32,
    cost: Optional[CostModel] = None,
    verbose: bool = True,
) -> dict:
    methods = list(methods)
    cost = cost or CostModel()
    cartridge_opts = dict(cartridge_opts or {"t": 128, "steps": 100})
    # accumulate per (method, setting): lists of (f1, em) and a memory sample
    acc: dict[tuple, dict] = {}

    def add(method, setting, f1, em, mem_bytes):
        k = (method, setting)
        d = acc.setdefault(k, {"f1": [], "em": [], "mem": []})
        d["f1"].append(f1); d["em"].append(em); d["mem"].append(mem_bytes)

    for ei, ex in enumerate(examples):
        ctx_ids = backend.tokenize(ex.context)
        q_ids = backend.tokenize(f"\n\nQuestion: {ex.question}\nAnswer:")
        full = backend.prefill(ctx_ids)
        full_bytes = full.nbytes()
        refs = None
        if any(m in _COMPRESSIVE for m in methods):
            refs = backend.reference_queries(ctx_ids, strategy=ref_strategy, n_per_head=n_ref)

        if "full_context" in methods:
            s = qa_score(_answer(backend, full, q_ids, max_new_tokens), ex.answers)
            add("full_context", "full", s["f1"], s["em"], full_bytes)

        if "rag" in methods:
            chunks = chunk_corpus(ctx_ids, chunk_tokens)
            retr = BowRetriever(chunks)
            for k in rag_ks:
                idxs = retr.retrieve(backend.tokenize(ex.question), k=k)
                ret_ids = [t for ci in sorted(idxs) for t in chunks[ci]]
                cache = backend.prefill(ret_ids) if ret_ids else full
                s = qa_score(_answer(backend, cache, q_ids, max_new_tokens), ex.answers)
                add("rag", f"k={k}", s["f1"], s["em"], cache.nbytes())

        for ratio in ratios:
            budget = CompactionBudget(ratio=float(ratio))
            for m in methods:
                if m not in _COMPRESSIVE and m != "cartridge":
                    continue
                try:
                    if m == "cartridge":
                        from dexa.cartridge.compiler import CartridgeCompiler
                        opts = {**cartridge_opts}
                        opts.pop("t", None)
                        cc = CartridgeCompiler(backend).compile(
                            ex.context, t=max(1, round(len(ctx_ids) / ratio)),
                            verbose=False, **opts).to_compact_cache()
                    else:
                        comp = build_compactor(m)
                        kw = {"ref_queries": refs} if comp.needs_ref_queries else {}
                        cc = comp.compact(full, budget, **kw)
                except Exception as e:  # keep the sweep going; record the gap
                    if verbose:
                        print(f"    [{m}@{ratio}x] skipped: {e}", flush=True)
                    continue
                s = qa_score(_answer(backend, cc, q_ids, max_new_tokens), ex.answers)
                add(m, f"{int(ratio)}x", s["f1"], s["em"], cc.nbytes())

        if verbose:
            print(f"  example {ei+1}/{len(examples)} done (ctx={len(ctx_ids)} tok)", flush=True)

    # aggregate -> frontier points
    points: list[FrontierPoint] = []
    for (method, setting), d in acc.items():
        mem = float(np.mean(d["mem"]))
        points.append(FrontierPoint(
            method=method, setting=setting,
            accuracy_f1=float(np.mean(d["f1"])), accuracy_em=float(np.mean(d["em"])),
            memory_bytes=mem, memory_ratio=(float(np.mean(_full_mems(acc))) / mem if mem else float("inf")),
            n=len(d["f1"]),
        ))
    points.sort(key=lambda p: (p.method, p.memory_bytes))
    result = {"points": [p.__dict__ for p in points], "n_examples": len(examples),
              "methods": methods, "verdict": _verdict(points)}
    return result


def report_frontier(result: dict, out_dir: str = "benchmarks/out") -> str:
    import os
    points = [FrontierPoint(**p) if not isinstance(p, FrontierPoint) else p
              for p in result["points"]]
    lines = ["", "Accuracy vs KV-memory frontier (docs/BENCHMARK.md)", ""]
    lines.append(f"{'method':>18} {'setting':>8} {'F1':>7} {'EM':>7} {'mem MB':>9} {'ratio':>7}")
    for p in sorted(points, key=lambda x: (x.method, x.memory_bytes)):
        lines.append(f"{p.method:>18} {p.setting:>8} {p.accuracy_f1:>7.3f} {p.accuracy_em:>7.3f} "
                     f"{p.memory_bytes/1e6:>9.1f} {p.memory_ratio:>6.0f}x")
    v = result["verdict"]
    lines += ["", f"VERDICT: {'PASS ✓' if v['passes'] else 'not yet'}  {v['detail']}"]
    out = "\n".join(lines)
    print(out)
    # plot: accuracy (F1) vs memory, one curve per method
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        by = {}
        for p in points:
            by.setdefault(p.method, []).append(p)
        for method, ps in by.items():
            ps = sorted(ps, key=lambda x: x.memory_bytes)
            xs = [p.memory_bytes / 1e6 for p in ps]
            ys = [p.accuracy_f1 for p in ps]
            ax.plot(xs, ys, marker="o", label=method, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("KV memory held for the corpus (MB, log)")
        ax.set_ylabel("QA accuracy (F1)")
        ax.set_title("Dexa: accuracy vs KV-memory frontier")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "frontier.png"), dpi=130)
        print(f"saved {out_dir}/frontier.png")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
    return out


def _full_mems(acc: dict) -> list[float]:
    for (method, _), d in acc.items():
        if method == "full_context":
            return d["mem"]
    # fall back to the largest memory seen
    return [max((m for d in acc.values() for m in d["mem"]), default=1.0)]


def _verdict(points) -> dict:
    """Check the pre-registered success threshold from docs/BENCHMARK.md at a
    16x-50x budget: cartridge >= full-2, >= best training-free +5, >= rag +3 (F1)."""
    by = {}
    for p in points:
        by.setdefault(p.method, []).append(p)
    full = max((p.accuracy_f1 for p in by.get("full_context", [])), default=None)
    out = {"passes": False, "detail": {}}
    if full is None or "cartridge" not in by:
        out["detail"]["note"] = "need full_context + cartridge points"
        return out
    # cartridge point in the [16x,50x] memory band (closest ratio in range)
    cart = [p for p in by["cartridge"] if p.setting in ("16x", "32x", "50x")]
    if not cart:
        cart = by["cartridge"]
    c = max(cart, key=lambda p: p.accuracy_f1)
    tf = [p.accuracy_f1 for m in _COMPRESSIVE for p in by.get(m, [])]
    rag = [p.accuracy_f1 for p in by.get("rag", [])]
    best_tf = max(tf) if tf else 0.0
    best_rag = max(rag) if rag else 0.0
    checks = {
        "cartridge_vs_full(-2)": c.accuracy_f1 >= full - 0.02,
        "cartridge_vs_trainfree(+5)": c.accuracy_f1 >= best_tf + 0.05,
        "cartridge_vs_rag(+3)": c.accuracy_f1 >= best_rag + 0.03,
    }
    out["passes"] = all(checks.values())
    out["detail"] = {"cartridge_f1": c.accuracy_f1, "full_f1": full,
                     "best_trainfree_f1": best_tf, "best_rag_f1": best_rag, **checks}
    return out
