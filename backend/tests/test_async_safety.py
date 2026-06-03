"""
CP229 — 라우트의 async-blocking 회귀 가드.

라우트 핸들러가 미래에 `async def` 로 바뀌면서 그 본문에서 무거운 동기 pandas
호출 (aggregate_prices, build_features, resample_price_frame 등) 을 await 없이
직접 부르면 이벤트 루프가 정지한다. 현재 모든 라우트는 def 이며 FastAPI 가
threadpool 에서 실행한다.

이 테스트는 AST 정적 검사로 그런 미래 회귀를 잡는다. 서버 기동 / 네트워크
불필요. 현재는 자명 통과 (모든 라우트 def).
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

ROUTE_MODULES = [
    "app.routers.prices",
    "app.routers.predict",
    "app.routers.v1.stocks",
    "app.routers.v1.health",
    "app.routers.v1.predictions",
    "app.routers.v1.strategies",
    "app.routers.v1.ai",
    "app.routers.v1.admin",
]

HEAVY_SYNC_CALLS = {
    "aggregate_prices",
    "build_features",
    "build_latest_feature_rows",
    "build_price_features",
    "resample_price_frame",
    "get_price_response_data",
    "get_indicator_response_data",
    "get_latest_prediction_data",
    "_load_frame",
    "_strategy_results",
}


def _load_module_source(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    src_path = getattr(module, "__file__", None)
    if not src_path:
        return None
    return pathlib.Path(src_path).read_text(encoding="utf-8")


def _is_router_decorated(decorator_list: list[ast.expr]) -> bool:
    for dec in decorator_list:
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "router":
                    return True
        elif isinstance(dec, ast.Attribute):
            if isinstance(dec.value, ast.Name) and dec.value.id == "router":
                return True
    return False


def _find_heavy_calls_without_await(func: ast.AsyncFunctionDef) -> list[str]:
    leaks: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Await):
            continue
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in HEAVY_SYNC_CALLS:
                leaks.append(name)
    return leaks


@pytest.mark.parametrize("module_name", ROUTE_MODULES)
def test_async_route_does_not_block_with_heavy_sync_call(module_name: str) -> None:
    source = _load_module_source(module_name)
    if source is None:
        pytest.skip(f"{module_name}: module not importable")
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and _is_router_decorated(node.decorator_list):
            leaks = _find_heavy_calls_without_await(node)
            if leaks:
                violations.append(f"{module_name}::{node.name} calls {leaks} without await")
    assert not violations, "async-blocking violations: " + " | ".join(violations)
