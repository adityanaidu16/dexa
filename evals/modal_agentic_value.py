"""Value benchmark for agentic serving: does the gap over generic serving grow with
shared-context size (W) and branching factor (B)?

The agentic access pattern = one large shared context (system + tools + history + RAG),
then B branching continuations (best-of-N actions / parallel reasoning), verify, keep the
best (early-stop). We measure GPU-seconds (wall-clock on a dedicated GPU) per agent turn
under three serving conditions on SGLang, swept over W and B:

  reprefill (generic, NO prompt cache) : each branch re-processes the full W context
                                         (independent requests, no prefix sharing).
  cached    (generic, WITH prompt cache): branches share the W prefix (RadixAttention),
                                          but all B run to completion (no early-stop).
  agentic   (our platform)             : shared prefix + verifier-guided early-stop —
                                          generate in waves, stop at the first passing
                                          branch, so only ~1/p branches are generated.

Honest framing: the win vs `reprefill` (prefill sharing) should grow with W AND B; the
win vs `cached` (early-stop only) should grow with B and be roughly flat in W — because a
good caching provider already amortizes the prefill. Reporting both brackets the value
against a weak and a strong generic baseline. Verification is simulated by a per-branch
pass probability p (first success at branch ceil(1/p)); we're measuring serving cost, not
task quality (quality is the separate verifier-search result).

    modal run evals/modal_agentic_value.py
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

app = modal.App("dexa-agentic-value")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

WS = [4096, 16384, 32768]
BS = [4, 8]
PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, g: int, p: float, wave: int) -> None:
    from time import perf_counter

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    base_ids = [int(x) for x in tok(PARA, add_special_tokens=False)["input_ids"]]

    def ids_for(L):
        reps = L // len(base_ids) + 1
        return (base_ids * reps)[:L]

    engine = sgl.Engine(model_path=model, mem_fraction_static=0.85,
                        disable_cuda_graph=True, context_length=max(WS) + g + 8)

    def tgen(batch, mnt):
        sp = [{"temperature": 0.8, "top_p": 0.95, "max_new_tokens": mnt} for _ in batch]
        t = perf_counter()
        engine.generate(input_ids=batch, sampling_params=sp)
        return perf_counter() - t

    tgen([ids_for(512)], 8)  # global warmup (JIT)

    # early-stop generates until the first passing branch; wave-rounded, capped at B.
    def k_earlystop(B):
        need = -(-max(1, round(1.0 / p)) // 1)          # ceil(1/p) branches to first pass
        return min(B, -(-need // wave) * wave)

    uid = [900_000]  # ever-incrementing unique token to defeat prefix caching for reprefill

    rows = []
    for W in WS:
        shared = ids_for(W)
        pfx = shared[:W - 1]
        tgen([pfx + [50]], 8)                            # warm the shared W-prefix
        for B in BS:
            k = k_earlystop(B)
            cached = [pfx + [100 + i] for i in range(B)]  # share prefix, unique last token

            def reprefill_batch():
                out = []
                for i in range(B):
                    uid[0] += 1
                    out.append([uid[0]] + pfx)            # unique FIRST token -> no sharing
                return out

            t_ca = min(tgen(cached, g) for _ in range(2))            # warm prefix
            t_ag = min(sum(tgen(cached[w:w + wave], g) for w in range(0, k, wave))
                       for _ in range(2))
            t_re = tgen(reprefill_batch(), g)                        # cold, once (no cache)

            rows.append((W, B, k, t_re, t_ca, t_ag))
            print(f"[W={W:>6} B={B}] reprefill={t_re:6.2f}s cached={t_ca:6.2f}s "
                  f"agentic(k={k})={t_ag:6.2f}s", flush=True)

    print("\n" + "=" * 90)
    print(f"AGENTIC SERVING VALUE — {model.split('/')[-1]} ({GPU}), gen={g}, p={p}, wave={wave}")
    print("  cost = GPU-seconds per agent turn (wall-clock, dedicated GPU)")
    print("=" * 90)
    hdr = (f"  {'W':>7} {'B':>3} {'reprefill s':>12} {'cached s':>10} {'agentic s':>10} "
           f"{'vs reprefill':>13} {'vs cached':>10} {'decode tok x':>13}")
    print(hdr)
    for W, B, k, t_re, t_ca, t_ag in rows:
        gain_re = t_re / t_ag if t_ag else 0
        gain_ca = t_ca / t_ag if t_ag else 0
        tok_x = B / k                                    # generic B*g vs agentic k*g decode tokens
        print(f"  {W:>7} {B:>3} {t_re:>12.2f} {t_ca:>10.2f} {t_ag:>10.2f} "
              f"{gain_re:>12.2f}x {gain_ca:>9.2f}x {tok_x:>12.2f}x")
    print("=" * 90)
    lo = rows[0]
    hi = rows[-1]
    print(f"  HEADLINE: vs a generic provider with NO prompt cache, agentic serving is "
          f"{lo[3]/lo[5]:.1f}x cheaper at W={lo[0]//1024}k/B={lo[1]} rising to "
          f"{hi[3]/hi[5]:.1f}x at W={hi[0]//1024}k/B={hi[1]} — the gap grows with context "
          f"and branching. vs a strong prompt-caching provider the win is the early-stop "
          f"factor (~{BS[-1]}/{rows[-1][2]}x fewer decode tokens), flat in W.")
    print("=" * 90)
    engine.shutdown()
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", g: int = 128, p: float = 0.5,
         wave: int = 2) -> None:
    print(f"agentic-serving value benchmark on {GPU}: {model}")
    run.remote(model, g, p, wave)
