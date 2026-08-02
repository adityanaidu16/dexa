"""Contention v2: does early-stop serving pay under LOAD (many concurrent agents)?

v1 (dedicated GPU, one agent) showed early-stop saves decode tokens but ~0 wall-clock,
because an idle GPU batches extra branches nearly free. The real cost axis for a
self-hosting buyer is throughput under contention: with many concurrent agents saturating
the GPU, each agent's extra branches DO compete for GPU slots, so generating fewer
(early-stop) should let more agents through.

We sweep concurrency C. Each agent does a turn = B branches over its OWN W-token context
(distinct per agent -> no cross-agent prefix sharing, the realistic multi-tenant case).
Two policies, same engine:
  generic : every agent generates all B branches.
  agentic : every agent generates only k = ceil(1/p) branches (verifier-guided early-stop).
We submit all C agents' work at once (SGLang continuous-batches/queues it) and time it.
throughput = C / makespan (agent-turns/sec); the number that matters is
throughput_agentic / throughput_generic as C rises.

Expectation: at low C (idle) the ratio ~1 (v1); as C saturates the GPU the ratio should
climb toward B/k (fewer branches = proportionally more agents served). If it does,
early-stop serving has a real throughput/$ advantage under load; if it stays ~1, it does
not and the platform cost story is dead.

    modal run evals/modal_contention.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("sglang[all]", "transformers")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-contention")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

CONCURRENCY = [1, 8, 32, 96]
PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, W: int, B: int, g: int, p: float) -> None:
    from time import perf_counter

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    base_ids = [int(x) for x in tok(PARA, add_special_tokens=False)["input_ids"]]
    k = min(B, max(1, round(1.0 / p)))     # branches generated with early-stop

    def ids_for(L):
        reps = L // len(base_ids) + 1
        return (base_ids * reps)[:L]

    engine = sgl.Engine(model_path=model, mem_fraction_static=0.85,
                        disable_cuda_graph=True, context_length=W + g + 8)

    uid = [2000]  # ever-incrementing valid unique token so agent prefixes never cache-hit

    def workload(C, branches_per_agent):
        batch = []
        for _a in range(C):
            uid[0] += 1
            pfx = ([uid[0]] + ids_for(W))[:W - 1]      # distinct per agent (no sharing)
            for b in range(branches_per_agent):
                batch.append(pfx + [100 + b])           # agent's branches share its prefix
        return batch

    def make_span(batch):
        sp = [{"temperature": 0.8, "top_p": 0.95, "max_new_tokens": g} for _ in batch]
        t = perf_counter()
        engine.generate(input_ids=batch, sampling_params=sp)
        return perf_counter() - t

    make_span(workload(2, 2))  # warmup

    rows = []
    for C in CONCURRENCY:
        dt_gen = make_span(workload(C, B))
        dt_ag = make_span(workload(C, k))
        thr_gen, thr_ag = C / dt_gen, C / dt_ag
        rows.append((C, dt_gen, dt_ag, thr_gen, thr_ag, thr_ag / thr_gen))
        print(f"[C={C:>3}] generic {thr_gen:6.2f} turns/s ({dt_gen:5.2f}s)  "
              f"agentic {thr_ag:6.2f} turns/s ({dt_ag:5.2f}s)  ratio {thr_ag/thr_gen:4.2f}x",
              flush=True)

    print("\n" + "=" * 84)
    print(f"CONTENTION — {model.split('/')[-1]} ({GPU}), W={W}, B={B}, early-stop k={k}, gen={g}")
    print("=" * 84)
    print(f"  {'concurrency':>11} {'generic turns/s':>16} {'agentic turns/s':>16} "
          f"{'agentic advantage':>18}")
    for C, dg, da, tg, ta, r in rows:
        print(f"  {C:>11} {tg:>16.2f} {ta:>16.2f} {r:>16.2f}x")
    print("=" * 84)
    lo, hi = rows[0][5], rows[-1][5]
    verdict = ("early-stop serving DOES pay under load — advantage grows with concurrency"
               if hi > 1.5 * lo and hi > 1.5 else
               "early-stop advantage does NOT grow with load — the platform cost story is weak")
    print(f"  HEADLINE: agentic advantage goes {lo:.2f}x (C={CONCURRENCY[0]}) -> {hi:.2f}x "
          f"(C={CONCURRENCY[-1]}), toward the B/k ceiling of {B/k:.1f}x. {verdict}.")
    print("=" * 84)
    engine.shutdown()
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", w: int = 4096, b: int = 8,
         g: int = 128, p: float = 0.5) -> None:
    print(f"contention benchmark on {GPU}: {model}, W={w}, B={b}")
    run.remote(model, w, b, g, p)
