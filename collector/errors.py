from __future__ import annotations


class SourceLimitReachedError(RuntimeError):
    """외부 데이터 소스의 일일 한도 또는 속도 제한 도달."""

    def __init__(self, source_name: str, message: str | None = None):
        self.source_name = source_name
        super().__init__(message or f"{source_name} 한도에 도달했습니다.")
