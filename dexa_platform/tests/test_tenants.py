"""Tenant registry + BYOC routing behaviour."""

import json

from dexa_platform.gateway.tenants import DEMO_KEY, Tenant, TenantRegistry


def test_env_single_tenant_byoc(monkeypatch):
    monkeypatch.setenv("DEXA_BACKEND_URL", "http://vllm.internal:8000/")
    monkeypatch.setenv("DEXA_API_KEY", "byoc-key-1")
    monkeypatch.setenv("DEXA_MODE", "byoc")
    monkeypatch.delenv("DEXA_TENANTS", raising=False)
    reg = TenantRegistry.load()
    t = reg.resolve("byoc-key-1")
    assert t is not None and t.mode == "byoc"
    assert t.backend_url == "http://vllm.internal:8000"   # trailing slash stripped
    assert not t.uses_mock


def test_unknown_key_rejected_when_auth_required(monkeypatch):
    monkeypatch.setenv("DEXA_REQUIRE_AUTH", "1")
    monkeypatch.delenv("DEXA_TENANTS", raising=False)
    monkeypatch.delenv("DEXA_BACKEND_URL", raising=False)
    reg = TenantRegistry.load()
    assert reg.resolve("nope") is None
    assert reg.resolve(None) is None


def test_demo_fallback_when_auth_not_required(monkeypatch):
    monkeypatch.delenv("DEXA_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("DEXA_TENANTS", raising=False)
    monkeypatch.delenv("DEXA_BACKEND_URL", raising=False)
    reg = TenantRegistry.load()
    t = reg.resolve(None)
    assert t is not None and t.key == DEMO_KEY and t.uses_mock


def test_multi_tenant_file(tmp_path, monkeypatch):
    p = tmp_path / "tenants.json"
    p.write_text(json.dumps([
        {"key": "k-hosted", "name": "acme", "mode": "hosted",
         "backend_url": "https://a.modal.run", "default_baseline": "gpt-4o"},
        {"key": "k-byoc", "name": "initech", "mode": "byoc",
         "backend_url": "http://b:8000", "default_baseline": "gpt-4o-mini",
         "cache_enabled": False},
    ]))
    monkeypatch.setenv("DEXA_TENANTS", str(p))
    monkeypatch.setenv("DEXA_REQUIRE_AUTH", "1")
    reg = TenantRegistry.load()
    assert reg.resolve("k-hosted").name == "acme"
    initech = reg.resolve("k-byoc")
    assert initech.mode == "byoc" and initech.cache_enabled is False
    assert initech.default_baseline == "gpt-4o-mini"
    # a demo tenant is auto-added, but require_auth means it's only reachable by its key
    assert reg.resolve(DEMO_KEY) is not None
    assert reg.resolve("stranger") is None


def test_tenant_public_hides_key():
    t = Tenant(key="secret-key", name="acme", backend_url="http://x:8000")
    pub = t.public()
    assert "secret-key" not in json.dumps(pub)
