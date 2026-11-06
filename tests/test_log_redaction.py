"""Access-log redaction: instance tokens never reach the journal."""

import logging
import logging.config

from realtimevoice.logfilter import TokenRedactingFilter


def make_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname="p", lineno=1,
        msg=message, args=(), exc_info=None,
    )


def test_query_token_redacted():
    record = make_record('127.0.0.1:0 - "GET /v1/realtime?model=&token=SUPERSECRET-123 HTTP/1.1" 101')
    assert TokenRedactingFilter().filter(record) is True
    assert "SUPERSECRET-123" not in record.getMessage()
    assert "token=[redacted]" in record.getMessage()


def test_bearer_redacted():
    record = make_record("auth header Bearer abc.def.ghi present")
    TokenRedactingFilter().filter(record)
    assert "abc.def.ghi" not in record.getMessage()
    assert "Bearer [redacted]" in record.getMessage()


def test_unrelated_lines_untouched():
    original = '127.0.0.1:0 - "GET / HTTP/1.1" 200'
    record = make_record(original)
    TokenRedactingFilter().filter(record)
    assert record.getMessage() == original


def test_filter_installable_via_dictconfig():
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"token_redact": {"()": TokenRedactingFilter}},
        "handlers": {
            "capture": {
                "class": "logging.StreamHandler",
                "filters": ["token_redact"],
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"uvicorn.access": {"handlers": ["capture"], "level": "INFO"}},
    }
    logging.config.dictConfig(config)
    log = logging.getLogger("uvicorn.access")
    log.info('GET /v1/realtime?token=TOKENTOKENTOKEN HTTP/1.1" 101')
