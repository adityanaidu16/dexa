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

_QGEN_LEADS = (
    "Write one specific factual question about this part of the document.",
    "Ask a question about a name, number, or detail mentioned in this part.",
    "What is an important factual question someone might ask about this part?",
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
        n_selfstudy: int = 16,
        answer_len: int = 24,
        selfstudy: str = "qa",
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

        # 3. self-study: build (query, teacher answer-distribution) items.
        if selfstudy == "qa":
            items = self._selfstudy_qa(full, corpus_ids, n_selfstudy, answer_len, verbose)
        elif selfstudy == "corpus_span":
            items = self._selfstudy_corpus_span(full, corpus_ids, n_selfstudy, answer_len)
        else:
            raise ValueError(f"unknown selfstudy {selfstudy!r}")
        if not items:
            raise RuntimeError("self-study produced no training items")
        if verbose:
            print(f"[cartridge] self-study: {len(items)} items ({selfstudy})", flush=True)

        # 4. train.
        opt = torch.optim.Adam(self._k + self._v, lr=lr)
        t0 = time.time()
        for step in range(steps):
            opt.zero_grad()
            total = 0.0
            for q_ids, t_lp, span in items:
                s_logits = self._student_logits(q_ids)          # [q_len, vocab] (grad)
                s_lp = torch.log_softmax(s_logits[-span:].float(), dim=-1)
                # KL(teacher || student) over the answer span, averaged.
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

    # --------------------------------------------------------- self-study
    def _newline_id(self) -> Optional[int]:
        ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
        return ids[-1] if ids else None

    @staticmethod
    def _truncate_at(ids: list[int], stop: Optional[int]) -> list[int]:
        if stop is not None and stop in ids:
            return ids[: ids.index(stop)]
        return list(ids)

    def _selfstudy_qa(self, full, corpus_ids, n, answer_len, verbose) -> list:
        """Corpus-conditioned synthetic Q&A self-study (the Cartridges method).

        For chunks spread across the corpus, the teacher (full corpus in context)
        writes a question about that chunk and answers it; we distill the student
        (cartridge) to match the teacher's ANSWER distribution given the question.
        Chunk-conditioning yields diverse, fact-bearing questions covering the
        whole corpus even with greedy decoding. (This is the distribution that
        matters — corpus-LM distillation reproduces the corpus but hurts QA.)"""
        T = len(corpus_ids)
        n = max(1, n)
        chunk = max(16, T // n)
        starts = np.linspace(0, max(0, T - chunk), n).astype(int)
        nl = self._newline_id()
        alead = self.tokenizer("\nAnswer:", add_special_tokens=False)["input_ids"]
        items = []
        for i, st in enumerate(starts):
            st = int(st)
            snippet = self.backend.detokenize(list(corpus_ids[st:st + 16]))
            lead_txt = (f"\n\n{_QGEN_LEADS[i % len(_QGEN_LEADS)]} "
                        f"Part: \"{snippet}\"\nQuestion:")
            lead = self.tokenizer(lead_txt, add_special_tokens=False)["input_ids"]
            q = self._truncate_at(self.backend.generate(full, lead, max_new_tokens=20), nl)
            if len(q) < 2:
                continue
            prefix = list(q) + list(alead)
            a = self._truncate_at(
                self.backend.generate(full, prefix, max_new_tokens=answer_len), nl)
            if len(a) < 1:
                continue
            query = prefix + list(a)
            span = len(a)
            with torch.no_grad():
                t_logits = self.backend._decode_logits(full, query)
            t_lp = torch.log_softmax(t_logits[-span:].float(), dim=-1).detach()
            items.append((query, t_lp, span))
            if verbose and i < 3:
                print(f"    Q={self.backend.detokenize(q)[:50]!r} "
                      f"A={self.backend.detokenize(a)[:30]!r}", flush=True)
        return items

    def _selfstudy_corpus_span(self, full, corpus_ids, n, span_len) -> list:
        """Ablation: distill the corpus's own spans (encodes content but HURTS QA
        — kept for comparison; see docs/CARTRIDGES.md)."""
        T = len(corpus_ids)
        span_len = max(8, span_len)
        starts = np.linspace(0, max(0, T - span_len), max(1, n)).astype(int)
        items, seen = [], set()
        for st in starts:
            st = int(st)
            if st in seen:
                continue
            seen.add(st)
            q_ids = list(corpus_ids[st:st + span_len])
            if len(q_ids) < 2:
                continue
            with torch.no_grad():
                t_logits = self.backend._decode_logits(full, q_ids)
            t_lp = torch.log_softmax(t_logits.float(), dim=-1).detach()
            items.append((q_ids, t_lp, len(q_ids)))
        return items

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
