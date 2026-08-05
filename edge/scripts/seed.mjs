// Seed a working org + API key + free credits directly into Postgres.
// Bootstrap until GitHub OAuth is wired — gives you a `dexa_live_…` key the deployed edge
// will authenticate. Mirrors dexa_platform/control/accounts.signup_oauth.
//
//   DATABASE_URL=postgres://…  node scripts/seed.mjs you@example.com
import crypto from "node:crypto";
import postgres from "postgres";

const url = process.env.DATABASE_URL;
if (!url) { console.error("set DATABASE_URL"); process.exit(1); }
const email = process.argv[2] || "dev@example.com";
const free = Number(process.env.FREE_CREDIT_USD || 5);

const secret = "dexa_live_" + crypto.randomBytes(24).toString("base64url");
const hash = crypto.createHash("sha256").update(secret).digest("hex");
const orgId = "org_" + crypto.randomBytes(8).toString("hex");
const keyId = "key_" + crypto.randomBytes(8).toString("hex");

const sql = postgres(url, { max: 1 });
try {
  await sql`insert into orgs (id, name, mode, backend_model) values (${orgId}, ${email.split("@")[0]}, 'hosted', 'dexa-cua-vlm')`;
  await sql`insert into credit_ledger (id, org_id, delta_usd, reason) values (${"cr_" + crypto.randomBytes(6).toString("hex")}, ${orgId}, ${free}, 'seed_grant')`;
  await sql`insert into api_keys (id, org_id, name, prefix, key_hash) values (${keyId}, ${orgId}, 'default', ${secret.slice(0, 14)}, ${hash})`;
  console.log("org:        ", orgId);
  console.log("free credit:", `$${free}`);
  console.log("\nAPI key (shown once — save it):\n  " + secret + "\n");
} finally {
  await sql.end();
}
