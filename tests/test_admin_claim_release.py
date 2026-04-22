# SPDX-License-Identifier: Apache-2.0
"""Tests for admin claim/release/unload endpoints with owner refcounting.

Covers Task 0.3 of the LocalAI oMLX backend plan: two new admin routes
(`/claim`, `/release`) plus an owner-aware modification of `/unload`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

import omlx.server  # noqa: F401 — register admin router on app
import omlx.admin.routes as admin_routes
from omlx.admin.auth import require_admin
from omlx.exceptions import ModelNotFoundError
from omlx.server import app


# ---------------------------------------------------------------------------
# Fake engine pool
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntry:
    """Minimal stand-in for EngineEntry — only what the routes touch."""

    model_id: str
    engine: Any = object()  # Truthy by default; tests flip to None on unload.
    claims: set[str] = field(default_factory=set)


class FakeEnginePool:
    """Minimal async engine pool implementing claim/release/_unload_engine."""

    def __init__(self, model_ids: list[str]) -> None:
        self._entries: dict[str, _FakeEntry] = {
            mid: _FakeEntry(model_id=mid) for mid in model_ids
        }
        self._lock = asyncio.Lock()
        self.unload_calls: list[tuple[str, bool]] = []

    # --- Same API shape the production pool exposes --------------------

    def get_entry(self, model_id: str) -> _FakeEntry | None:
        return self._entries.get(model_id)

    async def _unload_engine(self, model_id: str, force: bool = False) -> None:
        entry = self._entries.get(model_id)
        if entry is not None:
            # Mirror production semantics: skip when claimed unless forced.
            if entry.claims and not force:
                return
            entry.engine = None
            if force:
                entry.claims.clear()
        self.unload_calls.append((model_id, force))

    async def claim(self, model_id: str, owner: str) -> list[str]:
        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            entry.claims.add(owner)
            return sorted(entry.claims)

    async def release(self, model_id: str, owner: str) -> tuple[list[str], bool]:
        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            entry.claims.discard(owner)
            remaining = sorted(entry.claims)
            if not remaining and entry.engine is not None:
                await self._unload_engine(model_id)
                return remaining, True
            return remaining, False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pool_with_model():
    """Install a FakeEnginePool with a single 'test-model' into admin routes."""
    pool = FakeEnginePool(["test-model"])
    original = admin_routes._get_engine_pool
    admin_routes._get_engine_pool = lambda: pool
    # Override require_admin so tests are insulated from session auth state.
    app.dependency_overrides[require_admin] = lambda: True
    try:
        yield pool
    finally:
        admin_routes._get_engine_pool = original
        app.dependency_overrides.pop(require_admin, None)


@pytest.fixture
def client(fake_pool_with_model):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_claim_adds_owner(client):
    r = client.post(
        "/admin/api/models/test-model/claim", json={"owner": "token-A"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["claims"] == ["token-A"]


def test_claim_is_idempotent(client):
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-A"})
    r = client.post(
        "/admin/api/models/test-model/claim", json={"owner": "token-A"}
    )
    assert r.status_code == 200, r.text
    assert sorted(r.json()["claims"]) == ["token-A"]


def test_release_removes_owner(client):
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-A"})
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-B"})
    r = client.post(
        "/admin/api/models/test-model/release", json={"owner": "token-A"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["claims"] == ["token-B"]


def test_release_when_empty_triggers_unload(client, fake_pool_with_model):
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-A"})
    r = client.post(
        "/admin/api/models/test-model/release", json={"owner": "token-A"}
    )
    assert r.status_code == 200, r.text
    entry = fake_pool_with_model.get_entry("test-model")
    assert entry.engine is None
    assert r.json()["unloaded"] is True


def test_unload_with_owner_acts_as_release(client):
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-A"})
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-B"})
    r = client.post("/admin/api/models/test-model/unload?owner=token-A")
    assert r.status_code == 200, r.text
    assert r.json()["still_claimed"] is True


def test_unload_without_owner_force_unloads(client, fake_pool_with_model):
    client.post("/admin/api/models/test-model/claim", json={"owner": "token-A"})
    r = client.post("/admin/api/models/test-model/unload")
    assert r.status_code == 200, r.text
    assert fake_pool_with_model.get_entry("test-model").engine is None


def test_release_when_already_unloaded_is_noop(client, fake_pool_with_model):
    """Release on a model with no engine and no claims is a quiet no-op.

    Covers the idempotency path: returns ([], False) and does not raise,
    even when the engine was never loaded / already unloaded.
    """
    # Engine starts truthy in FakeEntry; simulate "already unloaded".
    fake_pool_with_model.get_entry("test-model").engine = None
    r = client.post(
        "/admin/api/models/test-model/release", json={"owner": "token-A"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["claims"] == []
    assert body["unloaded"] is False
    assert fake_pool_with_model.unload_calls == []
