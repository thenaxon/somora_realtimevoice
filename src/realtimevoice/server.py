"""Entrypoint: python -m realtimevoice [--host H] [--port P] [--print-token]"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .app import create_app
from .config import ServerConfig, SettingsStore

logger = logging.getLogger("realtimevoice")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="realtimevoice", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument(
        "--print-token",
        action="store_true",
        help="print the instance auth token once (from env or data/server.json)",
    )
    parser.add_argument("--data-dir", default=None, help="runtime state directory")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        server_config = ServerConfig.load(host=args.host, port=args.port, data_dir=args.data_dir)
    except OSError as exc:
        logger.error("cannot prepare data dir: %s", exc)
        return 1
    if args.print_token:
        print(server_config.auth_token)
        return 0
    app = create_app(
        settings_store=SettingsStore(data_dir=server_config.data_dir),
        auth_token=server_config.auth_token,
    )
    logger.info(
        "realtimevoice listening on %s:%s (data dir: %s)",
        server_config.host,
        server_config.port,
        server_config.data_dir,
    )
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=args.log_level,
        log_config=_uvicorn_log_config(args.log_level),
    )
    return 0


def _uvicorn_log_config(level: str) -> dict:
    """uvicorn's default logging config, with token redaction on all handlers."""
    import copy

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config["filters"] = {
        "token_redact": {"()": "realtimevoice.logfilter.TokenRedactingFilter"}
    }
    for handler in config["handlers"].values():
        handler["filters"] = ["token_redact"]
    config["loggers"]["uvicorn.access"]["level"] = level.upper()
    return config


if __name__ == "__main__":
    sys.exit(main())
