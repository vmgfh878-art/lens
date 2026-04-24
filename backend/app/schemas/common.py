from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class MetaResponse(BaseModel):
    request_id: str
    model_config = ConfigDict(extra="allow")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    meta: MetaResponse


class ApiResponse(BaseModel, Generic[T]):
    data: T
    meta: MetaResponse
