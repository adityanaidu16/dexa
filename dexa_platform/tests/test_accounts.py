"""Accounts, keys, metering, and hot-path resolution."""

import pytest

from dexa_platform.control import accounts, db, metering
from dexa_platform.control.resolver import KeyResolver


@pytest.fixture()
def s():
    db.reset_for_tests()
    sess = db.session()
    yield sess
    sess.close()


def test_signup_creates_org_key_and_credit(s):
    user, nk = accounts.signup_oauth(s, "github", "u123", "dev@acme.co")
    assert nk is not None and nk.secret.startswith("dexa_live_")
    snap = metering.usage_snapshot(s, user.org_id)
    assert snap["credit"]["granted_usd"] == pytest.approx(5.0)
    assert snap["credit"]["remaining_usd"] == pytest.approx(5.0)


def test_signup_is_idempotent(s):
    u1, nk1 = accounts.signup_oauth(s, "github", "same", "a@b.co")
    u2, nk2 = accounts.signup_oauth(s, "github", "same", "a@b.co")
    assert u1.id == u2.id and u1.org_id == u2.org_id
    assert nk1 is not None and nk2 is None            # secret shown only on first signup


def test_key_lifecycle_create_list_revoke_rotate(s):
    user, _ = accounts.signup_oauth(s, "github", "u", "e@x.co")
    org = user.org_id
    nk = accounts.create_key(s, org, name="ci")
    assert len(accounts.list_keys(s, org)) == 2
    assert accounts.revoke_key(s, org, nk.record.id) is True
    assert accounts.revoke_key(s, org, nk.record.id) is False   # already revoked
    rot = accounts.rotate_key(s, org, accounts.list_keys(s, org)[0].id)
    assert rot is not None and rot.secret != nk.secret


def test_metering_debits_credit_and_quota_flips(s):
    user, _ = accounts.signup_oauth(s, "github", "u", "e@x.co")
    org = user.org_id
    key_id = accounts.list_keys(s, org)[0].id
    assert metering.quota_ok(s, org) is True
    metering.record(s, org, key_id, dexa_usd=2.0, baseline_usd=80.0, saved_usd=78.0,
                    dexa_image_tokens=300, baseline_image_tokens=1100,
                    redundant_frac=0.9, cache_hit=False)
    assert metering.remaining_credit(s, org) == pytest.approx(3.0)
    metering.record(s, org, key_id, dexa_usd=4.0, baseline_usd=1.0, saved_usd=0.0,
                    dexa_image_tokens=300, baseline_image_tokens=1100,
                    redundant_frac=0.5, cache_hit=False)
    assert metering.remaining_credit(s, org) == pytest.approx(-1.0)
    assert metering.quota_ok(s, org) is False          # PLG gate trips


def test_metering_rollup_aggregates_same_day(s):
    user, _ = accounts.signup_oauth(s, "github", "u", "e@x.co")
    org = user.org_id
    key_id = accounts.list_keys(s, org)[0].id
    for _ in range(3):
        metering.record(s, org, key_id, dexa_usd=0.001, baseline_usd=0.04, saved_usd=0.039,
                        dexa_image_tokens=300, baseline_image_tokens=1100,
                        redundant_frac=0.8, cache_hit=False)
    snap = metering.usage_snapshot(s, org)
    assert snap["total"]["requests"] == 3
    assert snap["total"]["x_cheaper"] > 1
    assert len(snap["keys"]) == 1                      # one key, one daily row


def test_resolver_resolves_and_rejects_revoked(s):
    user, nk = accounts.signup_oauth(s, "github", "u", "e@x.co")
    r = KeyResolver(ttl=0)                              # ttl=0 -> always fresh, no cache staleness
    p = r.resolve(nk.secret)
    assert p is not None and p.org_id == user.org_id
    assert r.resolve("dexa_live_bogus") is None
    accounts.revoke_key(s, user.org_id, nk.record.id)
    assert r.resolve(nk.secret) is None                # revoked key no longer resolves


def test_resolver_ignores_non_keys(s):
    accounts.signup_oauth(s, "github", "u", "e@x.co")
    r = KeyResolver()
    assert r.resolve(None) is None
    assert r.resolve("sk-openai-style") is None        # not a dexa key
