#!/usr/bin/env python3
"""Summarize a vLLM /metrics snapshot (Prometheus text) taken at the end of a run: prefix-cache hit rate, prefill
against decode time, preemptions, request latency and prompt-size histograms. Writes metrics_summary.md next to it."""
import os, re, sys


def main(path):
    txt = open(path).read()

    def scalar(name):
        m = re.search(r'^vllm:%s\{[^}]*\} ([\d.e+]+)$' % name, txt, re.M)
        return float(m.group(1)) if m else None

    def buckets(name):
        out = []
        for m in re.finditer(r'^vllm:%s_bucket\{[^}]*le="([^"]+)"[^}]*\} ([\d.e+]+)$' % name, txt, re.M):
            out.append((float("inf") if m.group(1) == "+Inf" else float(m.group(1)), float(m.group(2))))
        return sorted(out)

    def hist(name, unit):
        b = buckets(name); rows = []; prev = 0
        for le, c in b:
            if c - prev > 0: rows.append(f"| up to {le:g}{unit} | {c - prev:.0f} |")
            prev = c
        return "| bucket | requests |\n|---|---|\n" + "\n".join(rows)

    n = scalar("request_prompt_tokens_count") or 1
    q = scalar("prefix_cache_queries_total") or 0; h = scalar("prefix_cache_hits_total") or 0
    out = [f"# vLLM metrics at the end of `{os.path.basename(os.path.dirname(os.path.abspath(path)))}`\n",
           "| metric | value |\n|---|---|",
           f"| requests | {n:.0f} |",
           f"| prompt tokens | {scalar('prompt_tokens_total'):.0f} ({scalar('prompt_tokens_total') / n:.0f} per request) |",
           f"| generation tokens | {scalar('generation_tokens_total'):.0f} ({scalar('generation_tokens_total') / n:.0f} per request) |",
           f"| prefix cache hit rate | {h / q:.4f} ({q - h:.0f} uncached prompt tokens, {(q - h) / n:.0f} per request) |",
           f"| preemptions | {scalar('num_preemptions_total'):.0f} |",
           f"| prefill time, sum (s) | {scalar('request_prefill_time_seconds_sum'):.0f} |",
           f"| decode time, sum (s) | {scalar('request_decode_time_seconds_sum'):.0f} |",
           f"| queue time, sum (s) | {scalar('request_queue_time_seconds_sum'):.2f} |",
           f"| time to first token, mean (ms) | {scalar('time_to_first_token_seconds_sum') / n * 1000:.0f} |",
           f"| time per output token, mean (ms) | {scalar('time_per_output_token_seconds_sum') / max(1, scalar('time_per_output_token_seconds_count')) * 1000:.1f} |",
           f"| end-to-end latency, mean (s) | {scalar('e2e_request_latency_seconds_sum') / n:.2f} |",
           ""]
    m = re.search(r'^vllm:cache_config_info\{([^}]*)\}', txt, re.M)
    if m:
        cfg = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        blk = int(cfg.get("num_gpu_blocks", "0") or 0); bs = int(cfg.get("block_size", "16") or 16)
        out.append(f"KV cache: {blk} blocks of {bs} tokens = {blk * bs:,} tokens; prefix caching {cfg.get('enable_prefix_caching')}; gpu_memory_utilization {cfg.get('gpu_memory_utilization')}.\n")
    out += ["## End-to-end request latency\n", hist("e2e_request_latency_seconds", " s"), "", "## Prompt tokens per request\n", hist("request_prompt_tokens", " tok"), "",
            "## Time to first token\n", hist("time_to_first_token_seconds", " s"), ""]
    text = "\n".join(out)
    open(os.path.join(os.path.dirname(os.path.abspath(path)), "metrics_summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
