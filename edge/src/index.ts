// Dexa edge Worker — the global front door + session router.
//   Auth (KV-cached) -> route session turns to the per-session Durable Object -> forward to
//   the GPU backend -> emit usage async to a Queue. Postgres (via Hyperdrive) is the source
//   of truth for accounts/usage; the hot path never blocks on it.
import { SessionDO } from "./session";
import { bearer, resolveKey } from "./keys";
import { creditRemaining, sql, type Tenant } from "./db";

export interface Env {
  SESSION: DurableObjectNamespace;
  KEYS: KVNamespace;
  DB: Hyperdrive;
  USAGE: Queue<UsageEvent>;
  ARCHIVE: R2Bucket;
  GPU_BACKEND_URL: string;
  BACKEND_MODEL: string;
  KEY_CACHE_TTL_SECONDS: string;
  FREE_CREDIT_USD: string;
}

type UsageEvent = { orgId: string; keyId: string; warm: boolean; savedUsd: number; savedMs: number; day: string };

export { SessionDO };

const json = (b: unknown, s = 200) =>
  new Response(JSON.stringify(b), { status: s, headers: { "content-type": "application/json" } });

function doStub(env: Env, name: string) {
  return env.SESSION.get(env.SESSION.idFromName(name));
}
async function callDO(env: Env, name: string, path: string, body: unknown): Promise<Response> {
  return doStub(env, name).fetch(`https://do${path}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
}

// hot-path credit gate, KV-cached so it doesn't hit Postgres every request
async function hasCredit(env: Env, orgId: string): Promise<boolean> {
  const ck = `credit:${orgId}`;
  const cached = await env.KEYS.get(ck);
  if (cached !== null) return Number(cached) > 0;
  const remaining = await creditRemaining(env, orgId).catch(() => 1); // fail-open on DB blip
  await env.KEYS.put(ck, String(remaining), { expirationTtl: 30 });
  return remaining > 0;
}

function meterFrom(env: Env, ctx: ExecutionContext, tenant: Tenant, r: any) {
  const m = r?._meter;
  if (!m) return;
  const day = new Date().toISOString().slice(0, 10);
  ctx.waitUntil(env.USAGE.send({ orgId: tenant.orgId, keyId: tenant.keyId, warm: !!m.warm, savedUsd: m.savedUsd || 0, savedMs: m.savedMs || 0, day }));
  delete r._meter;
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const p = url.pathname;
    if (p === "/health") return json({ ok: true });

    const tenant = await resolveKey(env, bearer(req));
    if (!tenant) return json({ error: { message: "invalid or missing API key", code: "invalid_api_key" } }, 401);
    if (!(await hasCredit(env, tenant.orgId)))
      return json({ error: { message: "free credit exhausted — add a payment method", code: "insufficient_quota" } }, 402);

    const ns = (clientId: string) => `${tenant.orgId}:${clientId}`;

    // --- create a session (optionally with a big initial context) ---
    if (p === "/v1/sessions" && req.method === "POST") {
      const b = await req.json<any>().catch(() => ({}));
      const clientId = b.id || `sess_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
      const r = await (await callDO(env, ns(clientId), "/init", {
        orgId: tenant.orgId, model: b.model || tenant.backendModel, system: b.system, context: b.context,
        backendUrl: tenant.backendUrl, backendModel: tenant.backendModel,
      })).json();
      return json({ session_id: clientId, ...r as object,
                    hint: "first turn prefills the context (cold); repeats restore it (warm)" });
    }

    // --- managed-context turn ---
    const turnMatch = p.match(/^\/v1\/sessions\/([^/]+)\/turn$/);
    if (turnMatch && req.method === "POST") {
      const b = await req.json<any>().catch(() => ({}));
      const r = await (await callDO(env, ns(turnMatch[1]), "/turn", { content: b.content, orgId: tenant.orgId })).json<any>();
      meterFrom(env, ctx, tenant, r);
      return json(r);
    }

    const idMatch = p.match(/^\/v1\/sessions\/([^/]+)$/);
    if (idMatch && req.method === "GET")
      return new Response(await (await callDO(env, ns(idMatch[1]), "/info", {})).text(), { headers: { "content-type": "application/json" } });
    if (idMatch && req.method === "DELETE")
      return new Response(await (await callDO(env, ns(idMatch[1]), "/delete", {})).text(), { headers: { "content-type": "application/json" } });

    // --- OpenAI-compatible chat/completions with the one optional `session` field ---
    if (p === "/v1/chat/completions" && req.method === "POST") {
      const b = await req.json<any>().catch(() => ({}));
      if (b.session) {
        // stateful: route to the session's DO, forward the full messages (LMCache restores prefix)
        const r = await (await callDO(env, ns(String(b.session)), "/forward", {
          messages: b.messages, orgId: tenant.orgId, backendUrl: tenant.backendUrl, backendModel: tenant.backendModel,
        })).json<any>();
        meterFrom(env, ctx, tenant, r);
        return json(r);
      }
      // stateless pass-through to the GPU backend
      const resp = await fetch(`${tenant.backendUrl || env.GPU_BACKEND_URL}/v1/chat/completions`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...b, model: tenant.backendModel }),
      });
      return new Response(resp.body, { status: resp.status, headers: { "content-type": "application/json" } });
    }

    return json({ error: "not found" }, 404);
  },

  // async metering: fold usage events into per-key daily rollups in Postgres
  async queue(batch: MessageBatch<UsageEvent>, env: Env): Promise<void> {
    const db = sql(env);
    try {
      for (const msg of batch.messages) {
        const e = msg.body;
        await db`
          insert into usage_daily (id, org_id, key_id, day, requests, warm_turns, cold_turns, saved_usd, saved_ms)
          values (${"use_" + crypto.randomUUID().slice(0, 12)}, ${e.orgId}, ${e.keyId}, ${e.day},
                  1, ${e.warm ? 1 : 0}, ${e.warm ? 0 : 1}, ${e.savedUsd}, ${e.savedMs})
          on conflict (key_id, day) do update set
            requests = usage_daily.requests + 1,
            warm_turns = usage_daily.warm_turns + ${e.warm ? 1 : 0},
            cold_turns = usage_daily.cold_turns + ${e.warm ? 0 : 1},
            saved_usd = usage_daily.saved_usd + ${e.savedUsd},
            saved_ms = usage_daily.saved_ms + ${e.savedMs}`;
      }
    } finally {
      await db.end({ timeout: 5 });
    }
  },
};
