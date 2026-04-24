from __future__ import annotations


class AppError(Exception):
    """API 계층에서 공통으로 사용하는 애플리케이션 예외."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class ResourceNotFoundError(AppError):
    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(
            message,
            code="RESOURCE_NOT_FOUND",
            status_code=404,
            details=details,
        )


class UpstreamUnavailableError(AppError):
    def __init__(self, message: str = "상위 데이터 저장소에 접근할 수 없습니다.", *, details: object | None = None) -> None:
        super().__init__(
            message,
            code="UPSTREAM_UNAVAILABLE",
            status_code=503,
            details=details,
        )


class ConfigError(AppError):
    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(
            message,
            code="CONFIG_ERROR",
            status_code=500,
            details=details,
        )


class TimeframeDisabledError(AppError):
    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(
            message,
            code="TIMEFRAME_DISABLED",
            status_code=409,
            details=details,
        )


class InsufficientHistoryError(AppError):
    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(
            message,
            code="INSUFFICIENT_HISTORY",
            status_code=409,
            details=details,
        )
