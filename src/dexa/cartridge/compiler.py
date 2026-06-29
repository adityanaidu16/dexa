"""CartridgeCompiler — train a compact KV cache for a corpus.

The method (Cartridges, Eyuboglu et al. 2025, productized on open models):

1. Tokenize the corpus (length T) and prefill it once.
2. Warm-start a compact cache of ``t`` tokens from a downsample of the corpus KV
   at positions ``linspace(0, T-1, t)`` (RoPE phases preserved).
3. Self-study: synthesize questions about the corpus; the *teacher* answers each
   with the FULL corpus in context.
4. Train: make the compact K/V ``requires_grad``, freeze the model, and minimize
   ``KL(teacher || student)`` on the answer span — teacher = full corpus context,
   student = the compact cartridge — backpropagating into the K/V only.

Torch-heavy; works with :class:`~dexa.engine.hf_backend.HFBackend` (uses its
model/tokenizer/spec). Validatable on CPU with a small model; the real win is a
real model + large corpus on GPU.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from transformers import DynamicCache

from dexa.cartridge.artifact import Cartridge

_DEFAULT_SELFSTUDY = (
    "Summarize the key facts in the text.",
    "List the specific names, numbers, and definitions mentioned.",
    "What questions could be asked about this text, and what are the answers?",
    "Explain the most important details in the text.",
    "Repeat the important entities and their relationships.",
    "What would someone need to remember from this text?",
)


class CartridgeCompiler:
    def __init__(self, backend) -> None:
        # backend: dexa.engine.hf_backend.HFBackend (duck-typed)
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.spec = backend.spec
        self.device = backend.device
        self.dtype = backend._torch_dtype

    # ------------------------------------------------------------------ API
    def compile(
        self,
        corpus: str,
        *,
        t: int = 64,
        steps: int = 120,
        lr: float = 0.02,
        n_selfstudy: int = 6,
        answer_len: int = 24,
        selfstudy_prompts: Optional[list[str]] = None,
        max_corpus_tokens: Optional[int] = None,
        verbose: bool = True,
    ) -> Cartridge:
        s = self.spec
        corpus_ids = self.backend.tokenize(corpus)
        if max_corpus_tokens:
            corpus_ids = corpus_ids[:max_corpus_tokens]
        T = len(corpus_ids)
        t = min(t, T)
        if verbose:
            print(f"[cartridge] corpus={T} tok -> t={t} ({T/t:.0f}x)  "
                  f"steps={steps} self-study={n_selfstudy}", flush=True)

        # 1+2. warm start from a downsample of the corpus KV.
        full = self.backend.prefill(corpus_ids)
        idx = np.linspace(0, T - 1, t).astype(np.int64)
        k0 = np.stack([full.layers[li].key[:, idx] for li in range(s.n_layers)])    # [L,n_kv,t,d]
        v0 = np.stack([full.layers[li].value[:, idx] for li in range(s.n_layers)])
        positions = full.positions[idx].astype(np.int64)

        # trainable params (one K and V tensor per layer, [1, n_kv, t, d]).
        self._k = [torch.tensor(k0[li], dtype=self.dtype, device=self.device,
                                requires_grad=True).unsqueeze(0) for li in range(s.n_layers)]
        self._v = [torch.tensor(v0[li], dtype=self.dtype, device=self.device,
                                requires_grad=True).unsqueeze(0) for li in range(s.n_layers)]
        # leaf tensors after unsqueeze are non-leaf; re-make leaves:
        self._k = [x.detach().clone().requires_grad_(True) for x in self._k]
        self._v = [x.detach().clone().requires_grad_(True) for x in self._v]
        self._t = t
        self._T = T

        # 3. self-study: distill the corpus's OWN content. Sample spans spread
        #    across the corpus; with the full corpus in context the teacher
        #    "repeats" each span with high confidence, and we train the cartridge
        #    to reproduce that — so every fact in the corpus enters the training
        #    signal (generic prompts don't surface specific facts and overfit).
        span_len = max(8, answer_len)
        n_items = max(1, n_selfstudy)
        starts = np.linspace(0, max(0, T - span_len), n_items).astype(int)
        items = []  # (q_ids, teacher_logprobs [span, vocab])
        seen = set()
        for st in starts:
            st = int(st)
            if st in seen:
                continue
            seen.add(st)
            q_ids = list(corpus_ids[st:st + span_len])
            if len(q_ids) < 2:
                continue
            with torch.no_grad():
                t_logits = self.backend._decode_logits(full, q_ids)  # teacher: full corpus, repeat span
            t_lp = torch.log_softmax(t_logits.float(), dim=-1).detach()  # [span, vocab]
            items.append((q_ids, t_lp))
        if not items:
            raise RuntimeError("self-study produced no training items")

        # 4. train.
        opt = torch.optim.Adam(self._k + self._v, lr=lr)
        t0 = time.time()
        for step in range(steps):
            opt.zero_grad()
            total = 0.0
            for q_ids, t_lp in items:
                s_logits = self._student_logits(q_ids)          # [span, vocab] (grad)
                s_lp = torch.log_softmax(s_logits.float(), dim=-1)
                # KL(teacher || student) averaged over span positions + vocab.
                kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
                kl.backward()
                total += float(kl.detach())
            opt.step()
            if verbose and (step % max(1, steps // 8) == 0 or step == steps - 1):
                print(f"  step {step:4d}  KL={total/len(items):.4f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)

        keys = np.stack([self._k[li].detach().float().cpu().numpy()[0] for li in range(s.n_layers)])
        values = np.stack([self._v[li].detach().float().cpu().numpy()[0] for li in range(s.n_layers)])
        return Cartridge(
            spec=s, keys=keys, values=values, positions=positions, logical_length=T,
            meta={"t": t, "T": T, "steps": steps, "lr": lr, "n_selfstudy": len(items),
                  "method": "cartridge", "model": s.name},
        )

    # ---------------------------------------------------------- grad forward
    def _student_logits(self, q_ids: list[int]) -> torch.Tensor:
        """Forward the query over the (trainable) cartridge; grad flows to K/V."""
        s = self.spec
        q_len = len(q_ids)
        t = self._t
        cache = DynamicCache(
            ddp_cache_data=[(self._k[li], self._v[li]) for li in range(s.n_layers)],
            config=self.model.config,
        )
        neg = self.backend._neg
        kv_len = t + q_len
        mask = torch.zeros(1, 1, q_len, kv_len, dtype=self.dtype, device=self.device)
        causal = torch.triu(
            torch.full((q_len, q_len), neg, dtype=self.dtype, device=self.device), diagonal=1
        )
        mask[0, 0, :, t:] = causal
        pos = torch.arange(self._T, self._T + q_len, device=self.device).unsqueeze(0)
        ids = torch.tensor([q_ids], device=self.device)
        with self.backend._attn_impl("sdpa"):
            out = self.model(
                input_ids=ids, attention_mask=mask, position_ids=pos,
                past_key_values=cache, use_cache=True,
            )
        return out.logits[0]
