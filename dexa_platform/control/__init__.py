"""Dexa control plane: accounts, API keys, and durable usage metering.

Stateful and low-throughput — the source of truth for who a customer is and what they owe.
The data-plane gateway never calls into here synchronously on the hot path; it resolves keys
from a short-TTL cache (see resolver.py) and meters usage through a thin recorder.
"""
