"""Stateful sessions — the differentiated product.

A session is a durable, warm context. It is prefilled once; its KV state is kept warm, then
offloaded to cheaper tiers on idle and restored on wake, instead of being re-prefilled every
turn. Grounded in measured results: see docs/STATEFUL_SESSIONS.md and evals/RESULTS.md.
"""
