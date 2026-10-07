from __future__ import annotations

from travel_agent.agents.replanner.errors import ReplannerProviderError


def map_provider_error(error: Exception) -> ReplannerProviderError:
    name = type(error).__name__.casefold()
    status = getattr(error, "status_code", None)
    if "timeout" in name:
        code, retryable = "timeout", True
    elif status == 429 or "ratelimit" in name or "rate_limit" in name:
        code, retryable = "rate_limit", True
    elif status in {401, 403}:
        code, retryable = "authentication_or_permission", False
    elif "connection" in name or (isinstance(status, int) and status >= 500):
        code, retryable = "upstream_unavailable", True
    else:
        code, retryable = "upstream_error", True
    return ReplannerProviderError(code, "修复模型服务暂时不可用", retryable=retryable)


def usage_value(usage: object, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and value >= 0 else None
