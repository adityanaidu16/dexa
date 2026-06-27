"""The STILL per-layer amortized compaction Perceiver.

Implements the network from "STILL: Amortized KV Cache Compaction in a Single
Forward Pass". One small :class:`StillPerceiver` per transformer layer learns to
map a layer's *full* keys/values ``K, V`` [n_kv, T, d] to a *compact*
``Ck, Cv`` [n_kv, t, d] plus per-key attention biases ``beta`` [n_kv, t] in a
single forward pass -- the amortized counterpart to Attention Matching's
per-context numerical fit.

Pipeline (per layer; the kv-heads are the batch axis, parameters are shared)
---------------------------------------------------------------------------
1. **Un-rotate** the cached keys with the *model's* inverse RoPE at their
   original token positions, so the perceiver reasons over rotation-free keys
   (``K`` is stored post-RoPE).
2. Concatenate ``[K_unrot ; V]`` per position into a width-``2d`` input token.
3. ``t`` learnable **latent queries** cross-attend into those input tokens.
   Routing uses the perceiver's *own* internal RoPE: input keys sit at their
   original (relative) positions, the latents are spread across the sequence via
   ``linspace(0, T-1, t)``. This makes routing positional/relative.
4. A latent **self-attention** block lets the latents coordinate.
5. Output heads project each latent to a compact key, a compact value and a
   scalar bias ``beta`` -- crucially **without a final RMSNorm**, so the natural
   per-key norm variation survives into the compact cache.
6. **Re-rotate** the compact keys with the model's RoPE at the latents'
   evenly-spaced positions, so they drop straight back into the model's
   attention at decode time.

Identity initialization
-----------------------
The module is initialized so that *at init* each latent simply copies its
positionally-nearest input key/value (and ``beta = 0``):

* **value pathway is an identity chain** -- the cross-attention value projection
  is the identity, and the key/value output heads are coordinate selectors that
  read the ``K_unrot`` / ``V`` halves straight through;
* **biased q/k projections** -- the query/key projections have zero weight and a
  constant bias, so the only thing distinguishing positions at init is the
  internal RoPE phase. A large routing temperature makes the cross-attention
  essentially one-hot to the nearest position;
* **zero-init self-attention / MLP output projections** -- the coordination
  blocks are residual and contribute nothing at init.

Consequently, with ``t == T`` (no compression) the compact cache reproduces the
input KV up to floating point: un-rotate then re-rotate at the same positions is
the identity, and routing is one-hot on the diagonal.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE half-split rotation: [x1, x2] -> [-x2, x1] (transformers layout)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor,
          sign: float = 1.0) -> torch.Tensor:
    """Apply RoPE to ``x`` (``sign=+1``) or its inverse (``sign=-1``).

    ``x`` is ``[..., L, d]``; ``positions`` is ``[L]`` (may be fractional);
    ``inv_freq`` is ``[d/2]``. cos/sin broadcast over any leading axes.
    """
    freqs = positions[:, None].to(inv_freq.dtype) * inv_freq[None, :]  # [L, d/2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [L, d]
    cos = emb.cos()
    sin = emb.sin()
    return x * cos + sign * _rotate_half(x) * sin


class StillPerceiver(nn.Module):
    """Amortized per-layer KV compactor (one forward pass).

    Parameters
    ----------
    head_dim:
        Per-head key/value dimension ``d`` (must be even for RoPE).
    n_latents:
        Number of compact tokens ``t`` produced per kv-head.
    model_rope_theta:
        RoPE base of the *frozen base model* -- used to un-rotate the cached keys
        and re-rotate the compact keys (must match the model that produced ``K``).
    internal_rope_theta:
        RoPE base of the perceiver's *own* routing attention.
    self_attn_heads:
        Number of heads in the latent self-attention coordination block.
    mlp_ratio:
        Hidden-width multiplier of the latent MLP.
    init_routing_temp:
        Initial routing temperature; large => near one-hot positional routing at
        init (it is a learnable scalar, so training can soften it).
    """

    def __init__(
        self,
        head_dim: int,
        n_latents: int,
        *,
        model_rope_theta: float = 10000.0,
        internal_rope_theta: float = 10000.0,
        self_attn_heads: int = 1,
        mlp_ratio: float = 2.0,
        init_routing_temp: float = 30.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if n_latents < 1:
            raise ValueError("n_latents must be >= 1")
        self.head_dim = int(head_dim)
        self.n_latents = int(n_latents)
        d = self.head_dim
        d_model = 2 * d  # width of a [K_unrot ; V] input token / latent

        # --- learnable latent content (routing-only at init via biased q) -----
        self.latents = nn.Parameter(torch.zeros(self.n_latents, d_model))

        # --- cross-attention routing (latent q, input k) ----------------------
        # q/k project the width-2d tokens down to the d-dim routing head.
        self.q_proj = nn.Linear(d_model, d, bias=True)
        self.k_proj = nn.Linear(d_model, d, bias=True)
        # value projection of the cross-attention -- identity (the carrier of the
        # value pathway): copies [K_unrot ; V] of the routed-to position.
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        # learnable routing temperature (peaked at init).
        self.routing_temp = nn.Parameter(torch.tensor(float(init_routing_temp)))

        # --- latent self-attention coordination block (residual) --------------
        if d_model % self_attn_heads != 0:
            self_attn_heads = 1
        self.sa_heads = self_attn_heads
        self.sa_norm = nn.RMSNorm(d_model)
        self.sa_q = nn.Linear(d_model, d_model, bias=False)
        self.sa_k = nn.Linear(d_model, d_model, bias=False)
        self.sa_v = nn.Linear(d_model, d_model, bias=False)
        self.sa_o = nn.Linear(d_model, d_model, bias=False)

        # --- latent MLP (residual) -------------------------------------------
        hidden = max(d_model, int(round(mlp_ratio * d_model)))
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp_in = nn.Linear(d_model, hidden, bias=True)
        self.mlp_out = nn.Linear(hidden, d_model, bias=True)
        self.mlp_act = nn.GELU()

        # --- output heads (NO final RMSNorm -- preserve norm variation) -------
        self.key_head = nn.Linear(d_model, d, bias=True)
        self.value_head = nn.Linear(d_model, d, bias=True)
        self.beta_head = nn.Linear(d_model, 1, bias=True)

        # --- RoPE frequency tables -------------------------------------------
        self.register_buffer(
            "model_inv_freq",
            1.0 / (model_rope_theta ** (torch.arange(0, d, 2).float() / d)),
            persistent=False,
        )
        self.register_buffer(
            "internal_inv_freq",
            1.0 / (internal_rope_theta ** (torch.arange(0, d, 2).float() / d)),
            persistent=False,
        )

        self._identity_init()

    # --- initialization ---------------------------------------------------
    def _identity_init(self) -> None:
        """Set parameters so each latent copies its nearest input at init."""
        d = self.head_dim
        d_model = 2 * d

        # Biased q/k: zero weight + constant bias => routing driven purely by the
        # internal RoPE phase (peaked on the matching position).
        nn.init.zeros_(self.q_proj.weight)
        nn.init.ones_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.weight)
        nn.init.ones_(self.k_proj.bias)

        # Cross-attention value = identity on [K_unrot ; V].
        with torch.no_grad():
            self.cross_v.weight.copy_(torch.eye(d_model))

        # Self-attention output projection zero => block is a no-op at init.
        nn.init.zeros_(self.sa_o.weight)

        # MLP output zero => block is a no-op at init.
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

        # Output heads: coordinate selectors. key_head reads the first d (the
        # un-rotated key half); value_head reads the last d (the value half).
        with torch.no_grad():
            kw = torch.zeros(d, d_model)
            kw[:, :d] = torch.eye(d)
            self.key_head.weight.copy_(kw)
            nn.init.zeros_(self.key_head.bias)

            vw = torch.zeros(d, d_model)
            vw[:, d:] = torch.eye(d)
            self.value_head.weight.copy_(vw)
            nn.init.zeros_(self.value_head.bias)

        # beta = 0 at init.
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    # --- forward ----------------------------------------------------------
    def latent_positions(self, T: int, device, dtype) -> torch.Tensor:
        """Relative positions of the ``t`` latents: ``linspace(0, T-1, t)``."""
        if self.n_latents == 1 or T <= 1:
            return torch.zeros(self.n_latents, device=device, dtype=dtype)
        return torch.linspace(0.0, float(T - 1), self.n_latents, device=device, dtype=dtype)

    def _self_attend(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention over the latents (residual, no RoPE)."""
        B, t, dm = x.shape
        h = self.sa_heads
        hd = dm // h
        n = self.sa_norm(x)
        q = self.sa_q(n).view(B, t, h, hd).transpose(1, 2)  # [B, h, t, hd]
        k = self.sa_k(n).view(B, t, h, hd).transpose(1, 2)
        v = self.sa_v(n).view(B, t, h, hd).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, t, dm)
        return x + self.sa_o(out)

    def forward(
        self,
        key: torch.Tensor,        # [B, T, d] post-RoPE keys (B = n_kv heads)
        value: torch.Tensor,      # [B, T, d]
        positions: torch.Tensor,  # [T] absolute token positions (float ok)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact one layer in a single pass.

        Returns ``(compact_key, compact_value, beta, compact_positions)`` with
        shapes ``[B, t, d]``, ``[B, t, d]``, ``[B, t]`` and ``[t]`` (absolute).
        Compact keys are post-RoPE (re-rotated at the compact positions).
        """
        B, T, d = key.shape
        device = key.device
        dtype = key.dtype
        positions = positions.to(device=device, dtype=dtype)

        # Relative frame so a non-zero position offset does not desync routing.
        pos0 = positions[0]
        rel_key_pos = positions - pos0                       # [T]
        rel_lat_pos = self.latent_positions(T, device, dtype)  # [t]
        comp_abs_pos = pos0 + rel_lat_pos                    # absolute compact pos

        # 1. un-rotate cached keys with the model's inverse RoPE.
        key_unrot = _rope(key, positions, self.model_inv_freq, sign=-1.0)  # [B,T,d]

        # 2. input tokens = [K_unrot ; V].
        x_in = torch.cat((key_unrot, value), dim=-1)         # [B, T, 2d]

        # 3. cross-attention routing (internal RoPE on q/k).
        q = self.q_proj(self.latents)                        # [t, d]
        q = _rope(q, rel_lat_pos, self.internal_inv_freq, sign=1.0)
        q = q.unsqueeze(0).expand(B, -1, -1)                 # [B, t, d]
        k = self.k_proj(x_in)                                # [B, T, d]
        k = _rope(k, rel_key_pos, self.internal_inv_freq, sign=1.0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)    # [B, t, T]
        scores = scores * self.routing_temp
        attn = torch.softmax(scores, dim=-1)
        v_in = self.cross_v(x_in)                            # [B, T, 2d]
        latents = attn @ v_in                                # [B, t, 2d]

        # 4. coordinate the latents.
        latents = self._self_attend(latents)
        latents = latents + self.mlp_out(self.mlp_act(self.mlp_in(self.mlp_norm(latents))))

        # 5. output heads (no final norm).
        key_unrot_c = self.key_head(latents)                 # [B, t, d]
        comp_value = self.value_head(latents)                # [B, t, d]
        beta = self.beta_head(latents).squeeze(-1)           # [B, t]

        # 6. re-rotate compact keys at their evenly-spaced positions.
        comp_key = _rope(key_unrot_c, comp_abs_pos, self.model_inv_freq, sign=1.0)

        return comp_key, comp_value, beta, comp_abs_pos
