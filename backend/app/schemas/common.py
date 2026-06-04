from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")

# CP236c: from_attributes (=구 orm_mode) 함정 주의.
#   현재 응답 data는 plain dict에서 validate되며 from_attributes는 꺼져 있다 (유지).
#   장차 ORM 객체를 직렬화하려고 from_attributes=True 를 켜는 모델을 만들면,
#   Pydantic이 unloaded 관계 속성에 접근하는 순간 SQLAlchemy lazy load가 트리거된다.
#   async 세션에서는 이게 금지된 동기 I/O (MissingGreenlet)로 터진다.
#   => 그런 모델을 쓰려면 쿼리에서 해당 관계를 반드시 eager load 할 것.
#   docs/db_repository_guide.md.


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
