// Tiering + savings — TS port of dexa_platform/sessions/tiering.py.
// Break-even boundaries derive from the measured cost model (evals/stateful_cost_model.py),
// not hand-picked constants.

// context tokens -> [kv_gb, prefill_ms, restore_cpu_ms, restore_nvme_ms] (A100, Qwen2.5-7B)
const PROFILE: Array<[number, [number, number, number, number]]> = [
  [4096, [0.23, 296, 25, 246]],
  [16384, [0.94, 1321, 115, 390]],
  [32768, [1.88, 3167, 182, 646]],
  [65536, [3.76, 8567, 297, 1228]],
];

const GPU_USD_HR = 1.8;
const RAM_USD_GB_HR = 0.006;
const NVME_USD_GB_HR = 0.0002;
const GPU_USABLE_GB = 60;
const GPU_USD_S = GPU_USD_HR / 3600;

export type Profile = { kvGb: number; prefillMs: number; restoreCpuMs: number; restoreNvmeMs: number };

export function profileFor(tokens: number): Profile {
  const keys = PROFILE.map((p) => p[0]);
  const lerp = (a: [number, number, number, number], b: [number, number, number, number], f: number) =>
    a.map((v, i) => v + f * (b[i] - v)) as [number, number, number, number];
  let v: [number, number, number, number];
  if (tokens <= keys[0]) {
    v = PROFILE[0][1].map((x) => (x * tokens) / keys[0]) as [number, number, number, number];
  } else if (tokens >= keys[keys.length - 1]) {
    const base = PROFILE[PROFILE.length - 1][1];
    v = base.map((x) => (x * tokens) / keys[keys.length - 1]) as [number, number, number, number];
  } else {
    let i = 0;
    while (tokens > PROFILE[i + 1][0]) i++;
    const f = (tokens - PROFILE[i][0]) / (PROFILE[i + 1][0] - PROFILE[i][0]);
    v = lerp(PROFILE[i][1], PROFILE[i + 1][1], f);
  }
  return { kvGb: v[0], prefillMs: v[1], restoreCpuMs: v[2], restoreNvmeMs: v[3] };
}

export type Breakevens = { warmS: number; ramS: number; nvmeS: number };

export function breakevens(tokens: number): Breakevens {
  const p = profileFor(tokens);
  const prefill = (p.prefillMs / 1000) * GPU_USD_S;
  const rCpu = (p.restoreCpuMs / 1000) * GPU_USD_S;
  const rNvme = (p.restoreNvmeMs / 1000) * GPU_USD_S;
  const warmHr = (p.kvGb / GPU_USABLE_GB) * GPU_USD_HR;
  const ramHr = p.kvGb * RAM_USD_GB_HR;
  const nvmeHr = p.kvGb * NVME_USD_GB_HR;
  return {
    warmS: (prefill / warmHr) * 3600,
    ramS: ((prefill - rCpu) / ramHr) * 3600,
    nvmeS: ((prefill - rNvme) / nvmeHr) * 3600,
  };
}

export type Tier = "warm" | "ram" | "nvme" | "drop";

export function decideTier(idleS: number, tokens: number): { tier: Tier; reason: string; breakevens: Breakevens } {
  const be = breakevens(tokens);
  let tier: Tier;
  if (idleS <= be.warmS) tier = "warm";
  else if (idleS <= be.ramS) tier = "ram";
  else if (idleS <= be.nvmeS) tier = "nvme";
  else tier = "drop";
  return { tier, reason: `idle ${idleS.toFixed(0)}s vs break-evens warm ${be.warmS.toFixed(0)}s / ram ${be.ramS.toFixed(0)}s / nvme ${be.nvmeS.toFixed(0)}s`, breakevens: be };
}

export function estimatedSavings(tokens: number, warm: boolean) {
  const p = profileFor(tokens);
  if (!warm) return { reprefillMs: p.prefillMs, restoreMs: p.restoreCpuMs, savedMs: 0, savedUsd: 0, speedup: null };
  const savedMs = Math.max(0, p.prefillMs - p.restoreCpuMs);
  return {
    reprefillMs: Math.round(p.prefillMs),
    restoreMs: Math.round(p.restoreCpuMs),
    savedMs: Math.round(savedMs),
    savedUsd: (savedMs / 1000) * GPU_USD_S,
    speedup: +(p.prefillMs / p.restoreCpuMs).toFixed(1),
  };
}
