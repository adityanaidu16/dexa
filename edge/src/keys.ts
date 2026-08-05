// Hot-path key resolution: Workers KV cache (short TTL) in front of Postgres.
// Auth resolves in a single KV read on the happy path; a revoke propagates within the TTL —
// the same fast/resilient trade as the Python resolver, now at the edge across all POPs.
import { lookupKey, sha256Hex, type Tenant } from "./db";
import type { Env } from "./index";

export function bearer(req: Request): string | null {
  const a = req.headers.get("authorization") || "";
  if (a.toLowerCase().startsWith("bearer ")) return a.slice(7).trim();
  return req.headers.get("x-api-key");
}

export async function resolveKey(env: Env, secret: string | null): Promise<Tenant | null> {
  if (!secret || !(secret.startsWith("dexa_live_") || secret.startsWith("dexa_test_"))) return null;
  const hash = await sha256Hex(secret);
  const cacheKey = `key:${hash}`;

  const cached = await env.KEYS.get(cacheKey, "json");
  if (cached) return cached === "null" ? null : (cached as Tenant);

  const tenant = await lookupKey(env, hash);
  const ttl = Math.max(30, Number(env.KEY_CACHE_TTL_SECONDS) || 30);
  await env.KEYS.put(cacheKey, JSON.stringify(tenant ?? "null"), { expirationTtl: ttl });
  return tenant;
}
