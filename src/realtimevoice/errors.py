class BackendError(Exception):
    """A backend (STT/LLM/TTS) failed or timed out."""

    def __init__(self, message: str, *, code: str = "backend_error", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


class ProtocolError(Exception):
    """A client sent something the public protocol rejects."""

    def __init__(self, message: str, *, code: str = "invalid_request_error", fatal: bool = False):
        super().__init__(message)
        self.message = message
        self.code = code
        self.fatal = fatal
