from __future__ import annotations

import pytest
from fastapi import HTTPException

from xraymind.api_auth import auth_mode, configured_principals, require_api_key


def test_role_api_keys_are_parsed_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRAYMIND_API_KEYS", "ops=admin:token_a,reader=reviewer:token_b,audit=viewer:token_c")
    monkeypatch.delenv("XRAYMIND_API_KEY", raising=False)

    principals = configured_principals()

    assert auth_mode() == "role_keys"
    assert principals["token_a"].role == "admin"
    assert principals["token_a"].key_id == "ops"
    assert principals["token_b"].role == "reviewer"
    assert principals["token_c"].role == "viewer"


def test_legacy_api_key_defaults_to_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XRAYMIND_API_KEYS", raising=False)
    monkeypatch.setenv("XRAYMIND_API_KEY", "token_legacy")

    principal = require_api_key("token_legacy")

    assert auth_mode() == "single_key"
    assert principal.key_id == "legacy"
    assert principal.role == "admin"


def test_dev_mode_allows_requests_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XRAYMIND_API_KEYS", raising=False)
    monkeypatch.delenv("XRAYMIND_API_KEY", raising=False)

    principal = require_api_key(None)

    assert auth_mode() == "dev_open"
    assert principal.key_id == "dev"
    assert principal.role == "admin"


def test_invalid_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRAYMIND_API_KEYS", "reviewer:token_b")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key("wrong_token")

    assert exc_info.value.status_code == 401
