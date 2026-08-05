// SessionDO — a session IS a Durable Object.
// Why this is the right primitive: a session needs durable per-session state, single-writer
// turn ordering (no concurrent-turn races), and *backend affinity* (route a session's turns
// to the same GPU replica so its KV stays warm). Durable Objects give all three natively.
//
// The DO holds the session's coordination state and forwards turns to the pinned GPU backend
// (vLLM + LMCache), which restores the KV instead of re-prefilling. It does NOT store KV
// (that lives in the engine's memory hierarchy) or screenshots.
import { decideTier, estimatedSavings } from "./tiering";
import type { Env } from "./index";

type Msg = { role: string; content: any };
type State = {
  orgId: string;
  model: string;
  backendUrl: string;
  backendModel: string;
  messages: Msg[];
  turns: number;
  tokens: number;
  tier: string;
  savedUsd: number;
  savedMs: number;
  lastActive: number;
};

function estTokens(messages: Msg[]): number {
  let n = 0;
  for (const m of messages) {
    if (typeof m.content === "string") n += m.content.length;
    else if (Array.isArray(m.content))
      for (const p of m.content) if (p?.type === "text") n += (p.text || "").length;
  }
  return Math.max(1, Math.floor(n / 4));
}

export class SessionDO {
  state: DurableObjectState;
  env: Env;
  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private async load(): Promise<State | null> {
    return (await this.state.storage.get<State>("s")) ?? null;
  }
  private async save(s: State) {
    await this.state.storage.put("s", s);
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname;

    if (path === "/init") {
      const b = await req.json<any>();
      const messages: Msg[] = [];
      if (b.system) messages.push({ role: "system", content: b.system });
      if (b.context) {
        messages.push({ role: "user", content: b.context });
        messages.push({ role: "assistant", content: "Context loaded. Ready." });
      }
      const s: State = {
        orgId: b.orgId, model: b.model || this.env.BACKEND_MODEL,
        backendUrl: b.backendUrl || this.env.GPU_BACKEND_URL, backendModel: b.backendModel || this.env.BACKEND_MODEL,
        messages, turns: 0, tokens: estTokens(messages), tier: "warm", savedUsd: 0, savedMs: 0, lastActive: Date.now(),
      };
      await this.save(s);
      return json({ session: this.publicOf(s, this.state.id.toString()) });
    }

    if (path === "/info") {
      const s = await this.load();
      if (!s) return json({ error: "session not found" }, 404);
      return json(this.publicOf(s, this.state.id.toString()));
    }

    if (path === "/delete") {
      await this.state.storage.deleteAll();
      return json({ deleted: true });
    }

    // /turn (managed context: client sends only new content) and /forward (client sends full
    // messages, one-field mode) share one code path.
    if (path === "/turn" || path === "/forward") {
      const b = await req.json<any>();
      let s = await this.load();
      if (!s) {
        // auto-init (create-on-first-use for the one-field chat/completions path)
        s = {
          orgId: b.orgId || "anon", model: b.model || this.env.BACKEND_MODEL,
          backendUrl: b.backendUrl || this.env.GPU_BACKEND_URL, backendModel: b.backendModel || this.env.BACKEND_MODEL,
          messages: [], turns: 0, tokens: 0, tier: "warm", savedUsd: 0, savedMs: 0, lastActive: Date.now(),
        };
      }
      const accumulate = path === "/turn";
      const userMsg: Msg = { role: "user", content: b.content ?? "" };
      const full: Msg[] = accumulate ? [...s.messages, userMsg] : (b.messages as Msg[]);
      const tokens = estTokens(full);
      const firstTouch = s.turns === 0;
      const idleS = (Date.now() - s.lastActive) / 1000;
      const decision = decideTier(idleS, tokens);

      const t0 = Date.now();
      const resp = await fetch(`${s.backendUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: s.backendModel, messages: full, max_tokens: 16, temperature: 0 }),
      });
      if (!resp.ok) return json({ error: `backend ${resp.status}` }, 502);
      const data = await resp.json<any>();
      const latencyMs = Date.now() - t0;
      const content = data?.choices?.[0]?.message?.content ?? "";

      const warm = !firstTouch; // deterministic: prefix served before => LMCache restores it
      const savings = estimatedSavings(tokens, warm);

      if (accumulate) {
        s.messages.push(userMsg, { role: "assistant", content });
      }
      s.turns += 1;
      s.tokens = tokens;
      s.tier = decision.tier;
      s.savedUsd += savings.savedUsd;
      s.savedMs += savings.savedMs;
      s.lastActive = Date.now();
      await this.save(s);

      return json({
        content,
        session: this.publicOf(s, this.state.id.toString()),
        turn: { warm, state: warm ? "restored (warm)" : "prefilled (cold)", latency_ms: latencyMs, context_tokens: tokens },
        tiering: { tier: decision.tier, idle_s: +idleS.toFixed(1), reason: decision.reason,
                   breakevens_s: { warm: Math.round(decision.breakevens.warmS), ram: Math.round(decision.breakevens.ramS), nvme: Math.round(decision.breakevens.nvmeS) } },
        savings_vs_stateless: savings,
        // billing signal for the Worker to enqueue
        _meter: { orgId: s.orgId, warm, savedUsd: savings.savedUsd, savedMs: savings.savedMs },
      });
    }

    return json({ error: "not found" }, 404);
  }

  private publicOf(s: State, id: string) {
    return { id, model: s.model, turns: s.turns, tokens: s.tokens, tier: s.tier,
             idle_s: +((Date.now() - s.lastActive) / 1000).toFixed(1), saved_usd: +s.savedUsd.toFixed(6),
             saved_ms: Math.round(s.savedMs), messages: s.messages.length };
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}
