"""Standalone OpenAI Realtime-compatible voice server."""

__all__ = ["create_app"]
__version__ = "2026.1106.10"


def __getattr__(name):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
