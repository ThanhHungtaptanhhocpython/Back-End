"""Request/response models for the local management API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidateRequest(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    ok: bool
    errors: dict[str, str] = Field(default_factory=dict)
    normalized: dict[str, str] = Field(default_factory=dict)
    unknown_keys: list[str] = Field(default_factory=list)


class ConfigUpdateRequest(BaseModel):
    # Complete desired non-secret key -> raw string map (as shown in the UI).
    values: dict[str, str] = Field(default_factory=dict)
    # Secret key -> new plaintext. Omit or send "" to keep the current value.
    secret_set: dict[str, str] = Field(default_factory=dict)
    # Secret keys to remove entirely.
    secret_clear: list[str] = Field(default_factory=list)
    note: str = ""
    # When true and a launcher is running, also drop a restart request.
    restart: bool = False


class ConfigUpdateResponse(BaseModel):
    ok: bool
    revision_id: int | None = None
    restart_required: bool = True
    restart_requested: bool = False
    errors: dict[str, str] = Field(default_factory=dict)
    detail: str = ""


class RestartRequest(BaseModel):
    reason: str = "manual"


class GenericResponse(BaseModel):
    ok: bool
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
