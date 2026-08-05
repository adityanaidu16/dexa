-- Dexa control-plane schema (Postgres, reached from Workers via Hyperdrive).
-- Mirrors dexa_platform/control/models.py. Screenshots/completions are never stored here;
-- only accounts, keys, usage rollups, and session metadata.

create table if not exists orgs (
  id            text primary key,
  name          text not null default '',
  mode          text not null default 'hosted',       -- hosted | byoc
  backend_url   text not null default '',              -- empty => shared hosted GPU pool
  backend_model text not null default 'dexa-cua-vlm',
  cache_enabled boolean not null default true,
  created_at    timestamptz not null default now()
);

create table if not exists users (
  id         text primary key,
  org_id     text not null references orgs(id),
  provider   text not null,                            -- github | google
  subject    text not null,
  email      text not null default '',
  created_at timestamptz not null default now(),
  unique (provider, subject)
);

create table if not exists api_keys (
  id           text primary key,
  org_id       text not null references orgs(id),
  name         text not null default 'default',
  prefix       text not null,                          -- visible, e.g. dexa_live_9f2c
  key_hash     text not null unique,                   -- sha-256 of the secret
  scopes       text not null default 'inference',
  created_at   timestamptz not null default now(),
  last_used_at timestamptz,
  revoked_at   timestamptz
);
create index if not exists api_keys_org on api_keys(org_id);

-- one row per (key, day): O(1) upsert per request, written async off the hot path
create table if not exists usage_daily (
  id             text primary key,
  org_id         text not null references orgs(id),
  key_id         text not null references api_keys(id),
  day            date not null,
  requests       bigint not null default 0,
  warm_turns     bigint not null default 0,            -- KV restored (session hit)
  cold_turns     bigint not null default 0,            -- prefilled (first touch / miss)
  dexa_usd       double precision not null default 0,
  saved_usd      double precision not null default 0,  -- vs stateless re-prefill
  saved_ms       double precision not null default 0,
  unique (key_id, day)
);
create index if not exists usage_daily_org on usage_daily(org_id);

create table if not exists credit_ledger (
  id         text primary key,
  org_id     text not null references orgs(id),
  delta_usd  double precision not null,                -- +grant / +topup
  reason     text not null default '',
  created_at timestamptz not null default now()
);

-- session metadata mirror (the live coordination state lives in the Durable Object;
-- this is the queryable record for the console / analytics)
create table if not exists sessions (
  id          text primary key,
  org_id      text not null references orgs(id),
  model       text not null,
  created_at  timestamptz not null default now(),
  last_active timestamptz not null default now(),
  turns       bigint not null default 0,
  tokens      bigint not null default 0,
  tier        text not null default 'warm',
  saved_usd   double precision not null default 0
);
create index if not exists sessions_org on sessions(org_id);
