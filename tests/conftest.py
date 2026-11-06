import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

from fake_backends import make_backends
from realtimevoice.config import Settings
from realtimevoice.session import RealtimeSession


def pytest_configure(config):
    config.addinivalue_line("markers", "speech_model: run with the real speech classifier in the VAD")


@pytest.fixture(autouse=True)
def _loudness_only_vad(request, monkeypatch):
    """Most tests drive the VAD with synthetic tones, which a speech model
    rightly does not take for speech: they run with loudness only. Tests of
    the speech model itself are marked `speech_model`."""
    if request.node.get_closest_marker("speech_model"):
        return
    import realtimevoice.session as session_mod

    monkeypatch.setattr(session_mod, "_speech_classifier", lambda td: None)


@pytest.fixture
def make_session():
    created = []

    def _make(**kwargs):
        settings: Settings = kwargs.pop("settings", None) or Settings()
        stt_timeout = kwargs.pop("stt_timeout", None)
        if stt_timeout is not None:
            settings.stt.request_timeout_s = stt_timeout
        backends = kwargs.pop("backends", None) or make_backends()
        session = RealtimeSession(backends=backends, settings=settings, **kwargs)
        created.append(session)
        return session, Driver(session)

    yield _make
    for session in created:
        session.close()


from fake_backends import Driver  # noqa: E402  (needs sys.path first)
