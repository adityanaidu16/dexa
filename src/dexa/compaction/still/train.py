"""KL-distillation training for the STILL perceivers.

STILL learns its per-layer perceivers *once*, against a frozen base model, so
that compaction is later a single forward pass. The objective is straight
knowledge distillation on a held-out answer region:

    teacher logits  = base model attending over the **full** KV cache
    student logits  = base model attending over the **compact** KV cache
                      (compact keys/values + per-key bias beta produced by the
                      perceivers in one pass)
    loss            = KL(teacher || student) averaged over the answer tokens

Only the perceiver parameters receive gradients; the base model is frozen. Both
teacher and student decode the *same* answer tokens at positions starting at the
context's logical length -- the only difference is full vs. compact context --
so the loss isolates compaction error.

This module is CPU-runnable end-to-end with the tiny test model for a handful of
steps (see ``main``). Full-scale training (long contexts, real models, many
documents) needs the cluster; the loop, optimizer and loss are identical there.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Importing the HF backend installs the bias-aware eager attention monkeypatch
# (``module._dexa_beta``) that the compact-cache student forward relies on.
from dexa.engine import hf_backend as _hfb  # noqa: F401
from dexa.engine.hf_backend import HFBackend
from dexa.compaction.still.perceiver import StillPerceiver

try:  # transformers is only present in the torch extra
    from transformers import DynamicCache
except Exception:  # pragma: no cover - import guard
    DynamicCache = None  # type: ignore

_NEG = torch.finfo(torch.float32).min

Sample = tuple[list[int], list[int]]  # (context_token_ids, answer_token_ids)


# --- perceiver construction ----------------------------------------------
def build_perceivers(
    backend: HFBackend,
    n_latents: int,
    *,
    internal_rope_theta: float = 10000.0,
    **kwargs,
) -> list[StillPerceiver]:
    """One identity-initialized perceiver per layer, RoPE-matched to ``backend``."""
    spec = backend.spec
    theta = float(getattr(backend.model.config, "rope_theta", None) or 10000.0)
    return [
        StillPerceiver(
            head_dim=spec.head_dim,
            n_latents=n_latents,
            model_rope_theta=theta,
            internal_rope_theta=internal_rope_theta,
            **kwargs,
        )
        for _ in range(spec.n_layers)
    ]


# --- differentiable decode helpers ---------------------------------------
def _full_kv(backend: HFBackend, context_ids: list[int]):
    """Run prefill and return per-layer (K, V) torch tensors [n_kv, T, d] plus
    the integer positions [T]. Detached -- the perceivers consume them as input
    and we never backprop into the frozen base for the cache itself."""
    s = backend.spec
    ids = torch.tensor([context_ids], dtype=torch.long, device=backend.device)
    positions = torch.arange(len(context_ids), device=backend.device).unsqueeze(0)
    with torch.no_grad():
        out = backend.model(input_ids=ids, position_ids=positions, use_cache=True)
    pkv = out.past_key_values
    layers = []
    for li in range(s.n_layers):
        k = pkv.layers[li].keys[0].to(torch.float32).detach()    # [n_kv, T, d]
        v = pkv.layers[li].values[0].to(torch.float32).detach()
        layers.append((k, v))
    pos = torch.arange(len(context_ids), device=backend.device, dtype=torch.float32)
    return layers, pos


def _decode_logits(
    backend: HFBackend,
    kv_layers: Sequence[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    logical_len: int,
    input_ids: list[int],
) -> torch.Tensor:
    """Decode ``input_ids`` over a (full or compact) torch KV cache and return
    logits ``[len(input_ids), vocab]``.

    ``kv_layers[li]`` is ``(K, V, beta)`` with K/V shaped ``[n_kv, t, d]`` and
    ``beta`` shaped ``[n_kv, t]`` (``None`` => zeros). Gradients flow through K/V
    and beta (the perceiver outputs); the frozen model contributes none.
    """
    s = backend.spec
    model = backend.model
    device = backend.device
    phys = kv_layers[0][0].shape[1]
    q_len = len(input_ids)
    kv_len = phys + q_len

    kv_pairs = []
    for (k, v, _beta) in kv_layers:
        kv_pairs.append((k.unsqueeze(0), v.unsqueeze(0)))  # [1, n_kv, t, d]
    cache = DynamicCache(ddp_cache_data=kv_pairs, config=model.config)

    # Shared additive 4D mask: attend-to-all over cached columns, causal among
    # the new tokens.
    mask = torch.zeros(1, 1, q_len, kv_len, dtype=torch.float32, device=device)
    causal = torch.triu(
        torch.full((q_len, q_len), _NEG, dtype=torch.float32, device=device), diagonal=1
    )
    mask[0, 0, :, phys:] = causal

    # Per-layer beta (expanded kv-head -> q-head), zero on new-token columns.
    for li, layer in enumerate(model.model.layers):
        beta = kv_layers[li][2]
        beta_full = torch.zeros(1, s.n_q_heads, 1, kv_len, dtype=torch.float32, device=device)
        if beta is not None:
            beta_q = beta.repeat_interleave(s.group_size, dim=0)  # [n_q, t]
            beta_full[0, :, 0, :phys] = beta_q
        layer.self_attn._dexa_beta = beta_full

    position_ids = torch.arange(
        logical_len, logical_len + q_len, device=device
    ).unsqueeze(0)
    try:
        out = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long, device=device),
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    finally:
        for layer in model.model.layers:
            layer.self_attn._dexa_beta = None
    return out.logits[0]  # [q_len, vocab]


def _sample_loss(
    backend: HFBackend,
    perceivers: list[StillPerceiver],
    sample: Sample,
) -> torch.Tensor:
    """KL(teacher || student) over the answer tokens for one sample."""
    s = backend.spec
    context_ids, answer_ids = sample
    T = len(context_ids)
    full_kv, positions = _full_kv(backend, context_ids)

    # Teacher: full cache (beta = 0), no grad.
    teacher_layers = [(k, v, None) for (k, v) in full_kv]
    with torch.no_grad():
        teacher_logits = _decode_logits(backend, teacher_layers, T, answer_ids)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)

    # Student: compact cache from the perceivers (one forward pass per layer).
    student_layers = []
    for li in range(s.n_layers):
        k, v = full_kv[li]
        Ck, Cv, beta, _pos = perceivers[li](k, v, positions)
        student_layers.append((Ck, Cv, beta))
    student_logits = _decode_logits(backend, student_layers, T, answer_ids)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)

    # KL(teacher || student) = sum P_t (log P_t - log P_s), mean over tokens.
    teacher_p = teacher_logp.exp()
    kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean()
    return kl


# --- training loop --------------------------------------------------------
def train(
    backend: HFBackend,
    samples: Sequence[Sample],
    perceivers: list[StillPerceiver],
    *,
    steps: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> list[float]:
    """Distill the perceivers against ``backend``'s frozen model.

    Returns the per-step mean KL loss. The base model is frozen; only the
    perceiver parameters are optimized.
    """
    if DynamicCache is None:  # pragma: no cover
        raise RuntimeError("transformers is required for STILL training")

    for p in backend.model.parameters():
        p.requires_grad_(False)
    backend.model.eval()

    params = [pp for perc in perceivers for pp in perc.parameters()]
    for perc in perceivers:
        perc.train()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history: list[float] = []
    for step in range(steps):
        opt.zero_grad()
        losses = [_sample_loss(backend, perceivers, sm) for sm in samples]
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if verbose:
            print(f"step {step:4d}  KL={history[-1]:.6f}")
    return history


# --- synthetic data + CLI -------------------------------------------------
def random_samples(
    backend: HFBackend,
    n_samples: int,
    *,
    context_len: int,
    answer_len: int,
    seed: int = 0,
) -> list[Sample]:
    """Random in-vocab token sequences -- a synthetic distillation task that
    exercises the full loop on CPU in seconds."""
    rng = np.random.default_rng(seed)
    vocab = int(backend.model.config.vocab_size)
    samples: list[Sample] = []
    for _ in range(n_samples):
        ctx = rng.integers(0, vocab, size=context_len).tolist()
        ans = rng.integers(0, vocab, size=answer_len).tolist()
        samples.append(([int(x) for x in ctx], [int(x) for x in ans]))
    return samples


# Config keys -> CLI dest names (only these are read from a --config YAML/JSON).
_CONFIG_KEYS = {
    "model": "model",
    "device": "device",
    "dtype": "dtype",
    "n_latents": "n_latents",
    "context_len": "context_len",
    "answer_len": "answer_len",
    "n_samples": "n_samples",
    "steps": "steps",
    "lr": "lr",
    "seed": "seed",
}


def _resolve_device(device: str) -> str:
    """Honor the requested device, falling back to CPU when CUDA is absent."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"NOTE: '{device}' requested but CUDA is unavailable; using CPU.")
        return "cpu"
    return device


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train STILL perceivers. Defaults run a tiny-model CPU smoke "
        "test; pass --config configs/still-train.yaml for a real run."
    )
    # --device defaults to cuda and falls back to cpu automatically, which keeps
    # the no-config invocation a CPU smoke run on a machine without a GPU.
    ap.add_argument("--config", default=None, help="YAML/JSON of hyperparameters (configs/still-train.yaml)")
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (cuda falls back to cpu if unavailable)")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16 ('auto' => f32 on cpu, bf16 on gpu)")
    ap.add_argument("--n-latents", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=12)
    ap.add_argument("--answer-len", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    # Two-pass parse so config values override the built-in defaults while an
    # explicit CLI flag still overrides the config.
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        from dexa.bench.config import _load_raw

        raw = _load_raw(pre.config) or {}
        unknown = set(raw) - set(_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown still-train config keys: {sorted(unknown)}")
        ap.set_defaults(**{_CONFIG_KEYS[k]: v for k, v in raw.items()})
    args = ap.parse_args(argv)

    device = _resolve_device(str(args.device))
    dtype = args.dtype
    if dtype == "auto":
        dtype = "float32" if device == "cpu" else "bfloat16"

    if args.config is None:
        print(
            "NOTE: no --config given -> tiny-model smoke defaults. Full STILL "
            "training (real models, long contexts, many documents) needs the "
            "cluster; see configs/still-train.yaml and docs/CLUSTER.md."
        )
    print(f"==> model={args.model} device={device} dtype={dtype} "
          f"n_latents={args.n_latents} steps={args.steps}", flush=True)

    backend = HFBackend(model_name=args.model, device=device, dtype=dtype)
    perceivers = build_perceivers(backend, args.n_latents)
    samples = random_samples(
        backend,
        args.n_samples,
        context_len=args.context_len,
        answer_len=args.answer_len,
        seed=args.seed,
    )
    history = train(backend, samples, perceivers, steps=args.steps, lr=args.lr, verbose=True)
    print(f"KL: {history[0]:.6f} -> {history[-1]:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
