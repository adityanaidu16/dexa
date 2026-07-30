"""Can a learned bridge MAKE the KV cache interchangeable across model weights?

Raw KV injection across a fine-tune failed (base<->instruct diverged in 2-5 tokens) even
though the caches were numerically close (cosine ~0.98). Hypothesis: model B's attention
expects KV in *its own* representation, but A's is a near-linear transform away. So fit a
cheap per-(layer, head) linear map  B_KV ~= A_KV @ W + b  by closed-form ridge regression
on a calibration passage, then at inference translate A's cached prefix KV into B's space
before B generates. No training loop.

Protocol: fit W on calibration text; test on a DIFFERENT held-out passage. Prefill A
(base) on the test prefix -> A_KV; bridge -> B-space KV; B greedily continues. Compare
token agreement vs B's own reference against the raw (un-bridged) baseline. Also report
KV reconstruction cosine to B's true KV, raw vs bridged, so we see if the map closes the
numeric gap and whether that translates into generation agreement.

  raw    : B generates from A's KV directly            (baseline from the interchange run)
  bridged: B generates from  A_KV @ W + b  (per head)  (the learned interchange)
  ceiling: B generates from its OWN KV                 (100% by construction)

    modal run evals/modal_kv_bridge.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "transformers==4.46.3", "accelerate", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-kv-bridge")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

CALIB = (
    "The eruption of Krakatoa in 1883 was heard nearly three thousand miles away. Coral "
    "polyps build reefs by secreting calcium carbonate, a process disrupted when ocean pH "
    "falls. In 1969 the Apollo guidance computer ran at about forty kilohertz with four "
    "kilobytes of memory, yet it landed two people on the Moon. The Fibonacci sequence "
    "appears in sunflower seed spirals because packing at the golden angle maximizes light. "
    "Venetian glassblowers on Murano were forbidden to leave the republic under penalty of "
    "death. Enzymes lower activation energy by stabilizing the transition state, sometimes "
    "accelerating reactions a billionfold. The Antikythera mechanism modeled the motions of "
    "the Sun and Moon with bronze gears. Sourdough rises because wild yeast and lactobacilli "
    "ferment the flour, producing carbon dioxide and lactic acid. The Treaty of Tordesillas "
    "in 1494 divided newly discovered lands between Spain and Portugal along a meridian. "
    "Neutron stars can spin hundreds of times per second. The printing press used movable "
    "metal type and an oil-based ink. Monarch butterflies migrate to a few forests in "
    "central Mexico, navigating by a time-compensated sun compass. Comparative advantage "
    "explains why two parties can both gain from trade. The Rosetta Stone carried the same "
    "decree in hieroglyphic, Demotic, and Greek. Hydrothermal vents support ecosystems that "
    "draw energy from chemistry rather than sunlight. Photosynthesis converts carbon dioxide "
    "and water into glucose using the energy of sunlight captured by chlorophyll. "
)
# Held-out test passage — different content so the bridge can't memorize positions.
TEST = (
    "Bees communicate the direction and distance of flowers through a waggle dance. The "
    "Silk Road carried not only goods but ideas, religions, and diseases across Eurasia. "
    "Superconductors expel magnetic fields and carry current without resistance below a "
    "critical temperature. The Library of Alexandria collected scrolls from across the "
    "ancient world before its gradual decline. Tectonic plates drift a few centimeters a "
    "year, reshaping continents over millions of years. The invention of the stirrup "
    "transformed mounted warfare in medieval Europe. Photorespiration reduces the efficiency "
    "of many plants on hot dry days when stomata close. The abacus enabled rapid arithmetic "
    "millennia before electronic calculators existed. Ocean currents redistribute heat from "
    "the equator toward the poles, moderating coastal climates. "
)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def run(base_id: str, instruct_id: str, k_gen: int, lam: float) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(instruct_id)

    def load(mid):
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to(dev).eval()
        return m

    def to_legacy(c):
        return c.to_legacy_cache() if hasattr(c, "to_legacy_cache") else c

    @torch.no_grad()
    def prefill(model, ids):
        return to_legacy(model(ids, use_cache=True).past_key_values)

    @torch.no_grad()
    def continue_greedy(model, legacy, last, k):
        cache = DynamicCache.from_legacy_cache(tuple(legacy))
        toks, cur = [], last
        for _ in range(k):
            out = model(cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(cur))
        return toks

    def agree(a, ref):
        m = sum(1 for x, y in zip(a, ref) if x == y)
        div = next((i for i, (x, y) in enumerate(zip(a, ref)) if x != y), len(ref))
        return m / len(ref), div

    A = load(base_id)        # source of the KV
    B = load(instruct_id)    # the model that must generate from it

    calib = tok(CALIB, return_tensors="pt").input_ids.to(dev)
    testids = tok(TEST, return_tensors="pt").input_ids.to(dev)
    print(f"[info] calib {calib.shape[1]} tok, test {testids.shape[1]} tok", flush=True)

    # ---- fit per-(layer, head) ridge maps K and V on calibration KV ----
    a_cal, b_cal = prefill(A, calib), prefill(B, calib)
    L = len(a_cal)
    n_heads = a_cal[0][0].shape[1]

    def fit(X, Y):  # X,Y: [N, d] float32 -> W: [d+1, d] (last row = bias)
        Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
        G = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=X.device, dtype=X.dtype)
        return torch.linalg.solve(G, Xa.T @ Y)

    def apply_map(X, W):
        Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)
        return Xa @ W

    WK, WV = [], []
    for l in range(L):
        ak, av = a_cal[l][0][0].float(), a_cal[l][1][0].float()  # [H,S,d]
        bk, bv = b_cal[l][0][0].float(), b_cal[l][1][0].float()
        WK.append([fit(ak[h], bk[h]) for h in range(n_heads)])
        WV.append([fit(av[h], bv[h]) for h in range(n_heads)])

    # ---- apply on held-out test KV ----
    prefix, last = testids[:, :-1], testids[:, -1:]
    a_test, b_test = prefill(A, prefix), prefill(B, prefix)

    def cos(x, y):
        return torch.nn.functional.cosine_similarity(x.flatten().float(),
                                                      y.flatten().float(), dim=0).item()

    bridged = []
    raw_cos, br_cos = [], []
    for l in range(L):
        ak, av = a_test[l][0][0], a_test[l][1][0]
        bk, bv = b_test[l][0][0], b_test[l][1][0]
        nk = torch.stack([apply_map(ak[h].float(), WK[l][h]) for h in range(n_heads)])
        nv = torch.stack([apply_map(av[h].float(), WV[l][h]) for h in range(n_heads)])
        raw_cos.append((cos(ak, bk) + cos(av, bv)) / 2)
        br_cos.append((cos(nk, bk) + cos(nv, bv)) / 2)
        bridged.append((nk.unsqueeze(0).to(bk.dtype), nv.unsqueeze(0).to(bv.dtype)))

    ref = continue_greedy(B, b_test, last, k_gen)               # B's own KV (ceiling)
    raw = continue_greedy(B, a_test, last, k_gen)               # B from raw base KV
    brd = continue_greedy(B, bridged, last, k_gen)              # B from bridged base KV
    f_raw, d_raw = agree(raw, ref)
    f_brd, d_brd = agree(brd, ref)

    print("\n" + "=" * 74)
    print(f"KV BRIDGE — {base_id.split('/')[-1]} KV -> {instruct_id.split('/')[-1]} gen")
    print("=" * 74)
    print(f"  mean KV recon cosine to B's true KV:  raw={sum(raw_cos)/L:.3f}  "
          f"bridged={sum(br_cos)/L:.3f}")
    print(f"  token agreement vs B's own reference ({k_gen} greedy tokens):")
    print(f"     raw base KV     : {f_raw*100:5.1f}%   (diverges at {d_raw}/{k_gen})")
    print(f"     BRIDGED base KV : {f_brd*100:5.1f}%   (diverges at {d_brd}/{k_gen})")
    print("=" * 74)
    print(f"  [B reference ] {tok.decode(ref)!r}")
    print(f"  [raw base KV ] {tok.decode(raw)!r}")
    print(f"  [bridged     ] {tok.decode(brd)!r}")
    print("=" * 74)
    better = f_brd - f_raw
    verdict = ("BRIDGE WORKS — a cheap linear map makes cross-fine-tune KV largely reusable"
               if f_brd > 0.6 else
               "PARTIAL — bridge helps but errors still compound" if better > 0.15 else
               "NO — even a fitted linear map can't make KV interchangeable; recompute")
    print(f"  VERDICT: raw {f_raw*100:.0f}% -> bridged {f_brd*100:.0f}%. {verdict}.")
    print("=" * 74)
    hf_cache.commit()


@app.local_entrypoint()
def main(base: str = "unsloth/Meta-Llama-3.1-8B",
         instruct: str = "unsloth/Llama-3.1-8B-Instruct",
         k_gen: int = 48, lam: float = 0.1) -> None:
    print(f"KV bridge experiment on {GPU}")
    run.remote(base, instruct, k_gen, lam)
