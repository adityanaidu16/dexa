// Postgres access from Workers via Hyperdrive (connection pooling + edge acceleration).
// One short-lived client per request; Hyperdrive multiplexes the underlying pool.
import postgres from "postgres";
import type { Env } from "./index";

export function sql(env: Env) {
  // env.DB.connectionString is provided by the Hyperdrive binding
  return postgres(env.DB.connectionString, { max: 1, fetch_types: false });
}

export async function sha256Hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export type Tenant = {
  orgId: string;
  keyId: string;
  name: string;
  mode: string;
  backendUrl: string; // resolved: org's own (BYOC) or the shared hosted pool
  backendModel: string;
};

// Source-of-truth key lookup (called only on a KV cache miss).
export async function lookupKey(env: Env, keyHash: string): Promise<Tenant | null> {
  const db = sql(env);
  try {
    const rows = await db`
      select k.id as key_id, o.id as org_id, o.name, o.mode, o.backend_url, o.backend_model
      from api_keys k join orgs o on o.id = k.org_id
      where k.key_hash = ${keyHash} and k.revoked_at is null
      limit 1`;
    if (rows.length === 0) return null;
    const r = rows[0];
    const backendUrl = r.backend_url || (r.mode === "hosted" ? env.GPU_BACKEND_URL : "");
    return { orgId: r.org_id, keyId: r.key_id, name: r.name, mode: r.mode, backendUrl, backendModel: r.backend_model || env.BACKEND_MODEL };
  } finally {
    await db.end({ timeout: 5 });
  }
}

export async function creditRemaining(env: Env, orgId: string): Promise<number> {
  const db = sql(env);
  try {
    const g = await db`select coalesce(sum(delta_usd),0) as v from credit_ledger where org_id=${orgId}`;
    const s = await db`select coalesce(sum(dexa_usd),0) as v from usage_daily where org_id=${orgId}`;
    return Number(g[0].v) - Number(s[0].v);
  } finally {
    await db.end({ timeout: 5 });
  }
}
