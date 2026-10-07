class ReplannerProviderError(Exception):
    def __init__(self, code: str, safe_message: str, *, retryable: bool = True) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
