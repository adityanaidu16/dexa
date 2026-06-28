"""Config-driven benchmark runs (the OSS-standard entrypoint).

A single YAML (or JSON) file describes a run: which backend/model, which suites
(needle-recall, agentic), and their parameters. This is what makes Dexa usable
as a tool — `dexa run --config configs/llama32-1b.yaml` on any box/scheduler,
no code edits. See `configs/` for examples and `docs/CLUSTER.md` for the runbook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class NiahSuite:
    enabled: bool = True
    task: str = "niah_single"
    lengths: list[int] = field(default_factory=lambda: [800])
    ratios: list[float] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    seeds: int = 4
    ref_strategy: str = "self_study"
    n_ref: int = 256
    compactors: list[str] = field(default_factory=lambda: [
        "attention_matching", "heavy_hitter", "snapkv", "recent_window", "random_subset",
    ])
    am: dict = field(default_factory=lambda: {
        "alloc": "sensitivity", "ridge": 0.05, "mass_frac": 0.5, "recent_frac": 0.1,
    })


@dataclass
class AgenticSuite:
    enabled: bool = True
    turns: int = 16
    turn_tokens: int = 300
    budget: int = 600
    seeds: int = 2
    n_facts: int = 6
    ref_strategy: str = "repeat_prefill"
    strategies: list[str] = field(default_factory=lambda: [
        "full_kv", "truncate_recent", "dexa:attention_matching", "dexa:heavy_hitter",
    ])
    am: dict = field(default_factory=lambda: {
        "alloc": "sensitivity", "ridge": 0.05, "mass_frac": 1.0, "recent_frac": 0.0,
    })


@dataclass
class RunConfig:
    name: str = "dexa-run"
    backend: str = "hf"            # hf | vllm | fake
    model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    out_dir: str = "benchmarks/out/run"
    plots: bool = True
    niah: NiahSuite = field(default_factory=NiahSuite)
    agentic: AgenticSuite = field(default_factory=AgenticSuite)
    # passthrough cost-model overrides (prefill_tok_per_s, gpu_dollars_per_hour, ...)
    cost: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _load_raw(path: str) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML required for YAML configs: pip install 'dexa[bench]' (or use a .json config)"
            ) from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a YAML/JSON file, applying defaults for omitted keys."""
    raw = _load_raw(path)
    niah = NiahSuite(**{**vars(NiahSuite()), **(raw.pop("niah", {}) or {})})
    agentic = AgenticSuite(**{**vars(AgenticSuite()), **(raw.pop("agentic", {}) or {})})
    top = {k: v for k, v in raw.items() if k in RunConfig.__dataclass_fields__}
    unknown = set(raw) - set(top)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return RunConfig(niah=niah, agentic=agentic, **top)


def build_backend(cfg: RunConfig):
    """Instantiate the backend named by the config."""
    if cfg.backend == "fake":
        from dexa.engine.fake import FakeBackend
        return FakeBackend()
    if cfg.backend == "hf":
        from dexa.engine.hf_backend import HFBackend
        return HFBackend(cfg.model, device=cfg.device, dtype=cfg.dtype)
    if cfg.backend == "vllm":
        from dexa.engine.vllm_backend import VLLMBackend
        return VLLMBackend(cfg.model, dtype=cfg.dtype)
    raise ValueError(f"unknown backend {cfg.backend!r}")
