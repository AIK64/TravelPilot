from __future__ import annotations


class PlannerProviderError(Exception):
    def __init__(
        self, code: str, safe_message: str, *, retryable: bool = True
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class PlannerUnavailableError(RuntimeError):
    def __init__(
        self, *, provider: str, model: str, code: str, attempt_count: int
    ) -> None:
        super().__init__("动态规划决策服务暂时不可用")
        self.provider = provider
        self.model = model
        self.code = code
        self.attempt_count = attempt_count
