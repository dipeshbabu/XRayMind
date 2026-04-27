"""Small API-key authentication layer for hosted XRayMind deployments.

The default remains developer-friendly: when no key environment variable is set,
requests are allowed. Production deployments can use either the legacy single-key
mode or role-aware keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, Header, HTTPException

ROLE_ORDER = {"viewer": 0, "reviewer": 1, "admin": 2}


@dataclass(frozen=True)
class ApiPrincipal:
    """Authenticated API caller."""

    key_id: str
    role: str


def _parse_role_keys(raw: str | None) -> dict[str, ApiPrincipal]:
    """Parse XRAYMIND_API_KEYS.

    Supported formats:
    - admin:secret-admin,reviewer:secret-reviewer
    - ops=admin:secret-admin,reader=reviewer:secret-reviewer
    """

    principals: dict[str, ApiPrincipal] = {}
    if not raw:
        return principals
    for idx, item in enumerate(raw.split(",")):
        item = item.strip()
        if not item:
            continue
        key_id = f"key_{idx + 1}"
        value = item
        if "=" in item:
            key_id, value = item.split("=", 1)
            key_id = key_id.strip() or key_id
        if ":" not in value:
            raise ValueError("XRAYMIND_API_KEYS entries must look like role:secret or id=role:secret")
        role, secret = value.split(":", 1)
        role = role.strip().lower()
        secret = secret.strip()
        if role not in ROLE_ORDER:
            raise ValueError(f"Unsupported API key role: {role!r}")
        if not secret:
            raise ValueError("API key secret cannot be empty")
        principals[secret] = ApiPrincipal(key_id=key_id.strip(), role=role)
    return principals


def configured_principals() -> dict[str, ApiPrincipal]:
    """Return configured API keys from env."""

    role_keys = _parse_role_keys(os.getenv("XRAYMIND_API_KEYS"))
    legacy = os.getenv("XRAYMIND_API_KEY")
    if legacy and legacy not in role_keys:
        legacy_role = os.getenv("XRAYMIND_API_KEY_ROLE", "admin").lower()
        if legacy_role not in ROLE_ORDER:
            raise ValueError(f"Unsupported legacy API key role: {legacy_role!r}")
        role_keys[legacy] = ApiPrincipal(key_id="legacy", role=legacy_role)
    return role_keys


def auth_mode() -> str:
    """Return the currently active auth mode for health/debug output."""

    if os.getenv("XRAYMIND_API_KEYS"):
        return "role_keys"
    if os.getenv("XRAYMIND_API_KEY"):
        return "single_key"
    return "dev_open"


def require_api_key(x_api_key: str | None = Header(default=None)) -> ApiPrincipal:
    """Authenticate a caller when keys are configured.

    No configured key means local/dev mode and returns an admin principal so old
    examples and tests keep working without secrets.
    """

    principals = configured_principals()
    if not principals:
        return ApiPrincipal(key_id="dev", role="admin")
    principal = principals.get(x_api_key or "")
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return principal


def require_role(min_role: str) -> Callable[[ApiPrincipal], ApiPrincipal]:
    """Build a FastAPI dependency that requires at least min_role."""

    min_role = min_role.lower()
    if min_role not in ROLE_ORDER:
        raise ValueError(f"Unknown role: {min_role}")

    async def _dependency(principal: ApiPrincipal = Depends(require_api_key)) -> ApiPrincipal:
        if ROLE_ORDER[principal.role] < ROLE_ORDER[min_role]:
            raise HTTPException(status_code=403, detail=f"Requires {min_role} role")
        return principal

    return _dependency


require_reviewer = require_role("reviewer")
require_admin = require_role("admin")
