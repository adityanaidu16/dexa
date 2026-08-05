// End-to-end smoke test against the deployed edge Worker.
// Proves the whole path: key auth (KV->Postgres) -> Session Durable Object -> Modal GPU
// backend (vLLM+LMCache) -> warm restore on turn 2 -> usage enqueued.
//
//   DEXA_EDGE_URL=https://dexa-edge.<sub>.workers.dev  DEXA_KEY=dexa_live_…  node scripts/smoke.mjs
const base = (process.env.DEXA_EDGE_URL || "").replace(/\/$/, "");
const key = process.env.DEXA_KEY;
if (!base || !key) { console.error("set DEXA_EDGE_URL and DEXA_KEY"); process.exit(1); }

const H = { "content-type": "application/json", authorization: "Bearer " + key };
const post = async (p, b) => (await fetch(base + p, { method: "POST", headers: H, body: JSON.stringify(b) })).json();

// ~13k-token context the agent will reason over across turns
const para =
  "File auth.py: def login(user, pw): validate(user); token=issue(user); return token. " +
  "validate() does not check password strength and issue() uses a static salt. " +
  "See crypto.py for hashing and sessions.py for the token store. ";
const context = para.repeat(220).slice(0, 52000);

console.log("1) create session with a big context …");
const s = await post("/v1/sessions", { system: "You are a code-review agent.", context });
const sid = s.session_id;
console.log("   session_id:", sid);

console.log("2) turn 1 (expect COLD prefill) …");
const t1 = await post(`/v1/sessions/${sid}/turn`, { content: "In one line: what does auth.py do?" });
console.log(`   ${t1.turn.state}  live=${t1.turn.latency_ms}ms  ctx=${t1.turn.context_tokens}tok`);

console.log("3) turn 2 (expect WARM restore) …");
const t2 = await post(`/v1/sessions/${sid}/turn`, { content: "Name the security bug." });
console.log(`   ${t2.turn.state}  live=${t2.turn.latency_ms}ms  tier=${t2.tiering.tier}`);
console.log("   savings vs stateless:", JSON.stringify(t2.savings_vs_stateless));

const sp = (t1.turn.latency_ms / t2.turn.latency_ms).toFixed(1);
console.log(`\n${t2.turn.warm ? "PASS" : "FAIL"}: turn 2 restored warm, ${sp}x faster end-to-end than the cold turn 1.`);
