"""A real transformer model backend for Dexa, built on Hugging Face transformers.

This implements :class:`~dexa.engine.base.ModelBackend` against a live Llama-family
model so that RoPE and grouped-query attention (GQA) actually apply. It is the
faithful counterpart to :class:`~dexa.engine.fake.FakeBackend`: same interface,
same tensor convention (**numpy float32** at every boundary), but the numbers come
from a genuine forward pass.

torch lives *only* in this file; we convert at the edges. ``engine/__init__`` does
not import this module so that torch stays an optional dependency.

Key design points
-----------------
* **prefill** returns the model's own ``past_key_values`` (keys are post-RoPE,
  which is exactly what HF caches), converted to numpy ``[n_kv_heads, T, head_dim]``.
* **reference_queries** recomputes post-RoPE queries from the per-layer hidden
  states (``q = RoPE(q_proj(input_layernorm(h)))``) so they match what attention
  actually sees, including the RoPE phase at the chosen positions.
* **decode against a (compact) cache**: we rebuild a ``DynamicCache`` from numpy
  K/V, feed new tokens with ``position_ids`` starting at the cache's *logical*
  length, and inject the per-key additive bias ``beta`` directly into the
  pre-softmax attention scores. Because ``beta`` varies per layer (and per head),
  it cannot ride on the single attention mask the model shares across layers; we
  therefore monkeypatch the eager attention function to add a per-module bias
  tensor (``module._dexa_beta``). The layer-independent structure (causal among
  new tokens, "attend-to-all" over the cached region, padding of ragged compact
  budgets) rides on a standard additive 4D mask we pass as ``attention_mask``.
"""

from __future__ import annotations

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama import modeling_llama as _ml
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from dexa.core.types import (
    CompactCache,
    CompactLayer,
    KVCache,
    LayerKV,
    ModelSpec,
    RefQueries,
)
from dexa.engine.base import ContextCache, ModelBackend

_NEG = torch.finfo(torch.float32).min


def _dexa_eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """Eager attention that additionally adds ``module._dexa_beta`` (per-layer,
    per-head, per-key additive bias) to the pre-softmax scores.

    A faithful copy of ``transformers``' ``eager_attention_forward`` plus the
    beta term. Installed by monkeypatch so the per-layer bias can be injected
    even though the model shares one attention mask across all layers.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]

    beta = getattr(module, "_dexa_beta", None)
    if beta is not None:
        # beta: [1, n_q_heads, 1, kv_len] -> broadcast over the query axis.
        attn_weights = attn_weights + beta

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# Install the bias-aware eager attention once. ``LlamaAttention.forward`` resolves
# ``eager_attention_forward`` from the module globals at call time and uses it as
# the fallback for ``_attn_implementation == "eager"`` (which is not in the global
# attention registry), so patching the module global is sufficient and scoped.
_ml.eager_attention_forward = _dexa_eager_attention_forward


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_name: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self._torch_dtype = getattr(torch, dtype)
        # dtype-aware "-inf" sentinel for masks/padded biases. float32's min
        # (-3.40e38) overflows bf16 (max ~3.39e38), so derive it from the active
        # dtype; exp(this) == 0 after softmax for any float dtype.
        self._neg = float(torch.finfo(self._torch_dtype).min)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self._torch_dtype, attn_implementation="eager"
        )
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        n_q = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
        self._spec = ModelSpec(
            name=model_name,
            n_layers=cfg.num_hidden_layers,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_size=cfg.hidden_size,
            dtype=dtype,
        )

    # --- properties -------------------------------------------------------
    @property
    def spec(self) -> ModelSpec:
        return self._spec

    # --- tokenization -----------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # --- internals --------------------------------------------------------
    def _ids(self, token_ids: list[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    def _rope(self, seq_len: int, position_ids: torch.Tensor):
        # cos/sin shaped [1, seq_len, head_dim].
        dummy = torch.zeros(1, seq_len, self._spec.hidden_size, device=self.device,
                            dtype=self._torch_dtype)
        return self.model.model.rotary_emb(dummy, position_ids)

    # --- prefill ----------------------------------------------------------
    def prefill(self, token_ids: list[int], *, position_offset: int = 0) -> KVCache:
        token_ids = list(token_ids)
        T = len(token_ids)
        s = self._spec
        position_ids = torch.arange(position_offset, position_offset + T,
                                    device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=self._ids(token_ids),
                position_ids=position_ids,
                use_cache=True,
            )
        pkv = out.past_key_values
        layers: list[LayerKV] = []
        for li in range(s.n_layers):
            k = pkv.layers[li].keys[0].to(torch.float32).cpu().numpy()    # [n_kv, T, d]
            v = pkv.layers[li].values[0].to(torch.float32).cpu().numpy()  # [n_kv, T, d]
            layers.append(LayerKV(key=np.ascontiguousarray(k), value=np.ascontiguousarray(v)))
        positions = np.arange(position_offset, position_offset + T, dtype=np.int64)
        return KVCache(spec=s, layers=layers, positions=positions, token_ids=token_ids)

    # --- reference queries ------------------------------------------------
    #: fixed self-study prompts: synthetic "questions" whose post-RoPE queries
    #: better cover how a real question attends back over the context than the
    #: repeat-prefill heuristic (Attention Matching paper, self-study).
    _SELF_STUDY_PROMPTS = (
        "Summarize the key facts.",
        "List every specific detail mentioned.",
        "What questions could be asked about this?",
        "Repeat the important numbers and names.",
    )

    def reference_queries(
        self,
        token_ids: list[int],
        *,
        strategy: str = "repeat_prefill",
        n_per_head: int = 512,
    ) -> RefQueries:
        token_ids = list(token_ids)
        T = len(token_ids)
        s = self._spec

        if strategy == "self_study":
            return self._self_study_queries(token_ids, n_per_head=n_per_head)

        if strategy == "repeat_prefill":
            sep = [self._sep_token()]
            full = token_ids + sep + token_ids
            # positions of the SECOND occurrence of the context.
            ref_start = T + len(sep)
            ref_slice = slice(ref_start, ref_start + T)
        elif strategy == "self":
            full = token_ids
            ref_slice = slice(0, T)
        else:
            raise ValueError(f"unknown reference-query strategy: {strategy!r}")

        layers = self._post_rope_queries(full, ref_slice)
        layers = [self._subsample(q, n_per_head) for q in layers]
        return RefQueries(spec=s, layers=layers)

    @staticmethod
    def _subsample(q: np.ndarray, n_per_head: int) -> np.ndarray:
        """Evenly subsample the query axis (axis=1) to at most ``n_per_head``."""
        if q.shape[1] > n_per_head:
            idx = np.linspace(0, q.shape[1] - 1, n_per_head).astype(int)
            q = q[:, idx]
        return np.ascontiguousarray(q)

    def _post_rope_queries(self, full: list[int], ref_slice: slice) -> list[np.ndarray]:
        """Recompute post-RoPE queries ``q = RoPE(q_proj(input_layernorm(h)))``
        over ``full`` and slice the query axis to ``ref_slice``. Returns a list
        per layer of ``[n_q_heads, n_sliced, head_dim]`` numpy arrays."""
        s = self._spec
        L = len(full)
        position_ids = torch.arange(L, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=self._ids(full),
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            cos, sin = self._rope(L, position_ids)

        layers: list[np.ndarray] = []
        with torch.no_grad():
            for li in range(s.n_layers):
                h = out.hidden_states[li]                # input to layer li: [1, L, hidden]
                layer = self.model.model.layers[li]
                normed = layer.input_layernorm(h)
                q = layer.self_attn.q_proj(normed)       # [1, L, n_q*d]
                q = q.view(1, L, s.n_q_heads, s.head_dim).transpose(1, 2)  # [1, n_q, L, d]
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)  # post-RoPE queries
                q = q[0, :, ref_slice, :].to(torch.float32).cpu().numpy()  # [n_q, n, d]
                layers.append(np.ascontiguousarray(q))
        return layers

    def _self_study_queries(self, token_ids: list[int], *, n_per_head: int) -> RefQueries:
        """Generate short synthetic continuations of the context under several
        fixed prompts, then collect the post-RoPE queries at the prompt+generated
        positions. These positions sit just after the context (logical positions
        ``T..``), so their RoPE phase relative to the cached keys matches a real
        follow-up question. Aggregate across prompts and subsample per head."""
        s = self._spec
        T = len(token_ids)
        sep = [self._sep_token()]
        n_gen = 16

        chunks: list[list[np.ndarray]] = [[] for _ in range(s.n_layers)]
        for prompt in self._SELF_STUDY_PROMPTS:
            p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            primer = token_ids + sep + list(p_ids)
            gen = self._greedy_continuation(primer, n_gen)
            full = primer + gen
            # reference positions: the prompt + generated tokens (after context).
            ref_slice = slice(T + len(sep), len(full))
            per_layer = self._post_rope_queries(full, ref_slice)
            for li in range(s.n_layers):
                chunks[li].append(per_layer[li])

        layers: list[np.ndarray] = []
        for li in range(s.n_layers):
            q = np.concatenate(chunks[li], axis=1)        # [n_q, sum_n, d]
            layers.append(self._subsample(q, n_per_head))
        return RefQueries(spec=s, layers=layers)

    def _greedy_continuation(self, prefix: list[int], n_new: int) -> list[int]:
        """Greedily decode ``n_new`` tokens after ``prefix`` (no dexa bias)."""
        L = len(prefix)
        position_ids = torch.arange(L, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=self._ids(prefix),
                position_ids=position_ids,
                use_cache=True,
            )
            cache = out.past_key_values
            last = out.logits[0, -1]
            generated: list[int] = []
            for step in range(n_new):
                nxt = int(torch.argmax(last).item())
                generated.append(nxt)
                pos = torch.tensor([[L + step]], device=self.device)
                out = self.model(
                    input_ids=self._ids([nxt]),
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                last = out.logits[0, -1]
        return generated

    def _sep_token(self) -> int:
        for attr in ("bos_token_id", "eos_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                return int(tid)
        return 0

    # --- cache reconstruction ---------------------------------------------
    def _physical_len(self, context: ContextCache) -> int:
        """A single physical KV length used for every layer (we pad ragged
        compact budgets up to this so one shared attention mask suffices)."""
        if isinstance(context, KVCache):
            return context.seq_len
        return max(
            (k.shape[0] for layer in context.layers for k in layer.keys),
            default=0,
        )

    def _build_cache(self, context: ContextCache, phys: int):
        """Return a ``DynamicCache`` (all layers padded to ``phys``) and a list of
        per-layer beta tensors ``[n_q_heads, phys]`` (real bias on kept keys,
        ``_NEG`` on padded slots, ``0`` everywhere for a full cache)."""
        s = self._spec
        kv_pairs = []
        betas: list[torch.Tensor] = []
        for li in range(s.n_layers):
            k_full = np.zeros((s.n_kv_heads, phys, s.head_dim), dtype=np.float32)
            v_full = np.zeros((s.n_kv_heads, phys, s.head_dim), dtype=np.float32)
            beta_kv = np.full((s.n_kv_heads, phys), self._neg, dtype=np.float32)
            if isinstance(context, KVCache):
                lk = context.layers[li]
                t = lk.key.shape[1]
                k_full[:, :t] = lk.key
                v_full[:, :t] = lk.value
                beta_kv[:, :t] = 0.0
            else:
                cl = context.layers[li]
                for h in range(s.n_kv_heads):
                    t = cl.keys[h].shape[0]
                    k_full[h, :t] = cl.keys[h]
                    v_full[h, :t] = cl.values[h]
                    beta_kv[h, :t] = cl.biases[h]
            k_t = torch.from_numpy(k_full).to(self.device, self._torch_dtype).unsqueeze(0)
            v_t = torch.from_numpy(v_full).to(self.device, self._torch_dtype).unsqueeze(0)
            kv_pairs.append((k_t, v_t))
            # expand kv-head beta to q-heads.
            beta_q = np.repeat(beta_kv, s.group_size, axis=0)  # [n_q, phys]
            betas.append(torch.from_numpy(beta_q).to(self.device, self._torch_dtype))
        cache = DynamicCache(ddp_cache_data=kv_pairs, config=self.model.config)
        return cache, betas

    def _logical_len(self, context: ContextCache) -> int:
        return context.seq_len if isinstance(context, KVCache) else context.logical_length

    def _decode_logits(self, context: ContextCache, input_ids: list[int]) -> torch.Tensor:
        """Forward ``input_ids`` over ``context`` and return logits
        ``[len(input_ids), vocab]``. Injects beta and logical-length positions."""
        s = self._spec
        phys = self._physical_len(context)
        q_len = len(input_ids)
        kv_len = phys + q_len
        cache, betas = self._build_cache(context, phys)

        start = self._logical_len(context)
        position_ids = torch.arange(start, start + q_len, device=self.device).unsqueeze(0)

        # Shared, layer-independent additive 4D mask [1, 1, q_len, kv_len]:
        #   * compact/cached columns (0..phys-1): 0  (attend-to-all; padding is
        #     handled per-layer by beta = _NEG)
        #   * new-token columns: causal.
        mask = torch.zeros(1, 1, q_len, kv_len, dtype=self._torch_dtype, device=self.device)
        causal = torch.triu(
            torch.full((q_len, q_len), self._neg, dtype=self._torch_dtype, device=self.device),
            diagonal=1,
        )
        mask[0, 0, :, phys:] = causal

        # Per-layer beta tensor over the full kv axis (0 on new-token columns).
        for li, layer in enumerate(self.model.model.layers):
            beta_full = torch.zeros(1, s.n_q_heads, 1, kv_len, dtype=self._torch_dtype,
                                    device=self.device)
            beta_full[0, :, 0, :phys] = betas[li]
            layer.self_attn._dexa_beta = beta_full

        try:
            with torch.no_grad():
                out = self.model(
                    input_ids=self._ids(input_ids),
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
        finally:
            for layer in self.model.model.layers:
                layer.self_attn._dexa_beta = None
        return out.logits[0]  # [q_len, vocab]

    def _drop_last(self, context: ContextCache):
        """Return (context_without_last_position, last_token_id, last_position).

        Used to recover the boundary logit (prediction right after the context)
        when scoring/generating with an empty prompt.
        """
        if isinstance(context, KVCache):
            if context.token_ids is None:
                raise ValueError("KVCache.token_ids required to score with an empty prompt")
            last_tok = int(context.token_ids[-1])
            last_pos = int(context.positions[-1])
            layers = [LayerKV(key=l.key[:, :-1], value=l.value[:, :-1]) for l in context.layers]
            ctx2 = KVCache(
                spec=context.spec,
                layers=layers,
                positions=context.positions[:-1],
                token_ids=list(context.token_ids[:-1]),
            )
            return ctx2, last_tok, last_pos
        # CompactCache: drop the last physical key per head. Exact only when the
        # last compact slot corresponds to the last logical position (true for the
        # keep-all / identity case); the last token id is read from meta.
        toks = context.meta.get("token_ids")
        if not toks:
            raise ValueError("CompactCache.meta['token_ids'] required to score with empty prompt")
        last_tok = int(toks[-1])
        last_pos = context.logical_length - 1
        new_layers = []
        for cl in context.layers:
            new_layers.append(
                CompactLayer(
                    keys=[k[:-1] for k in cl.keys],
                    values=[v[:-1] for v in cl.values],
                    biases=[b[:-1] for b in cl.biases],
                    positions=[p[:-1] for p in cl.positions],
                )
            )
        ctx2 = CompactCache(
            spec=context.spec,
            layers=new_layers,
            logical_length=context.logical_length - 1,
            method=context.method,
            meta=dict(context.meta, token_ids=list(toks[:-1])),
        )
        return ctx2, last_tok, last_pos

    # --- score ------------------------------------------------------------
    def score(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        prompt_token_ids = list(prompt_token_ids)
        target_token_ids = list(target_token_ids)
        n_t = len(target_token_ids)
        if n_t == 0:
            return np.zeros(0, dtype=np.float32)

        if prompt_token_ids:
            # Feed prompt+target; target[k] is predicted by the logit at the
            # preceding position.
            seq = prompt_token_ids + target_token_ids
            logits = self._decode_logits(context, seq)
            pred = logits[len(prompt_token_ids) - 1 : len(prompt_token_ids) - 1 + n_t]
        else:
            # Empty prompt: recover the boundary logit by re-feeding the last
            # context token over the context minus its last position.
            ctx2, last_tok, _ = self._drop_last(context)
            seq = [last_tok] + target_token_ids
            logits = self._decode_logits(ctx2, seq)
            pred = logits[0:n_t]

        logprobs = torch.log_softmax(pred.float(), dim=-1)
        tgt = torch.tensor(target_token_ids, dtype=torch.long, device=self.device)
        out = logprobs[torch.arange(n_t), tgt]
        return out.to(torch.float32).cpu().numpy()

    # --- generate ---------------------------------------------------------
    def generate(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 64,
        greedy: bool = True,
    ) -> list[int]:
        prompt_token_ids = list(prompt_token_ids)

        if prompt_token_ids:
            logits = self._decode_logits(context, prompt_token_ids)
        else:
            ctx2, last_tok, _ = self._drop_last(context)
            logits = self._decode_logits(ctx2, [last_tok])
            context = context  # decode the rest against the original full context

        generated: list[int] = []
        last_logit = logits[-1]
        for _ in range(max_new_tokens):
            if greedy:
                nxt = int(torch.argmax(last_logit).item())
            else:
                probs = torch.softmax(last_logit.float(), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            generated.append(nxt)
            if len(generated) >= max_new_tokens:
                break
            # Re-decode the full (prompt + generated) prefix over the context. This
            # is simple and exact (no incremental-cache bookkeeping needed for the
            # tiny models used in tests); the prefix grows by one each step.
            prefix = prompt_token_ids + generated
            last_logit = self._decode_logits(context, prefix)[-1]
        return generated

    # --- optional attention outputs (numpy, model-free) -------------------
    def attention_outputs(self, cache: ContextCache, queries: RefQueries) -> list[np.ndarray]:
        s = self._spec
        scale = 1.0 / np.sqrt(s.head_dim)
        out: list[np.ndarray] = []
        for li in range(s.n_layers):
            Q = queries.layers[li]            # [n_q, n_ref, d]
            n_ref = Q.shape[1]
            res = np.zeros((s.n_q_heads, n_ref, s.head_dim), dtype=np.float32)
            for qh in range(s.n_q_heads):
                h = s.kv_head_of(qh)
                if isinstance(cache, KVCache):
                    K = cache.layers[li].key[h]
                    V = cache.layers[li].value[h]
                    beta = None
                else:
                    K = cache.layers[li].keys[h]
                    V = cache.layers[li].values[h]
                    beta = cache.layers[li].biases[h]
                logits = (Q[qh] @ K.T) * scale
                if beta is not None:
                    logits = logits + beta[None, :]
                logits -= logits.max(axis=-1, keepdims=True)
                w = np.exp(logits)
                w /= np.clip(w.sum(axis=-1, keepdims=True), 1e-8, None)
                res[qh] = w @ V
            out.append(res)
        return out
