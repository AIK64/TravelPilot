from __future__ import annotations

from travel_agent.agents.planner.errors import PlannerProviderError


def map_provider_error(error: Exception) -> PlannerProviderError:
    """把 SDK 异常折叠为不泄露上游响应的稳定错误语义。"""
    name = type(error).__name__.casefold()
    status_code = getattr(error, "status_code", None)
    if "timeout" in name:
        code, retryable = "timeout", True
    elif status_code == 429 or "ratelimit" in name or "rate_limit" in name:
        code, retryable = "rate_limit", True
    elif status_code == 401 or "authentication" in name:
        code, retryable = "authentication", False
    elif status_code == 403 or "permission" in name:
        code, retryable = "permission", False
    elif "connection" in name:
        code, retryable = "connection", True
    elif isinstance(status_code, int) and status_code >= 500:
        code, retryable = "upstream_unavailable", True
    elif isinstance(status_code, int) and status_code >= 400:
        code, retryable = "invalid_request", False
    else:
        code, retryable = "upstream_error", True
    return PlannerProviderError(code, "动态规划决策服务暂时不可用", retryable=retryable)


def usage_value(usage: object, name: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and value >= 0 else None
