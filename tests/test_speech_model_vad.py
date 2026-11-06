"""A cough is loud, but it is not speech.

With speech_check "model" the VAD asks a small speech classifier as well as
the loudness: a loud burst that is not speech never starts a turn, so it
never reaches the speech recogniser (which would turn it into "ja").
"""

import numpy as np
import pytest

from realtimevoice import speech_model
from realtimevoice.config import TurnDetectionConfig
from realtimevoice.vad import BurstDiscarded, SpeechStarted, SpeechStopped, TurnDetector

RATE = 24000


def _cough(seconds=0.4, seed=0):
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    env = np.exp(-np.arange(n) / RATE * 10)
    return (rng.normal(0, 9000, n) * env).astype("<i2").tobytes()


def _silence(seconds):
    return bytes(int(RATE * seconds) * 2)


class FakeClassifier:
    """Speech probability by script, one value per 20 ms frame."""

    def __init__(self, probs):
        self.probs = list(probs)

    def feed(self, pcm):
        return self.probs.pop(0) if self.probs else 0.0


def _loud(seconds):
    t = np.arange(int(RATE * seconds)) / RATE
    return (6000 * np.sin(2 * np.pi * 200 * t)).astype("<i2").tobytes()


def test_a_loud_frame_the_classifier_does_not_hear_as_speech_starts_nothing():
    det = TurnDetector(classifier=FakeClassifier([0.1] * 50))
    events = det.process(_loud(1.0))
    assert not any(isinstance(e, SpeechStarted) for e in events)


def test_speech_the_classifier_hears_starts_and_ends_a_turn():
    det = TurnDetector(classifier=FakeClassifier([0.9] * 50 + [0.0] * 50))
    events = det.process(_loud(1.0) + _silence(1.0))
    assert any(isinstance(e, SpeechStarted) for e in events)
    assert any(isinstance(e, SpeechStopped) for e in events)


def test_a_short_dip_between_words_does_not_end_the_turn():
    # 0.4 is under the start bar (0.5) but over the bar inside a turn (0.35)
    probs = [0.9] * 20 + [0.4] * 30 + [0.9] * 20
    det = TurnDetector(classifier=FakeClassifier(probs), silence_duration_ms=420)
    events = det.process(_loud(1.4))
    assert sum(isinstance(e, SpeechStarted) for e in events) == 1
    assert not any(isinstance(e, (SpeechStopped, BurstDiscarded)) for e in events)


def test_loudness_only_is_still_a_choice():
    assert TurnDetectionConfig().speech_check == "model"
    assert TurnDetectionConfig(speech_check="loudness").speech_check == "loudness"


@pytest.mark.speech_model
def test_the_model_ships_with_the_server_and_loads():
    assert speech_model.available()


@pytest.mark.speech_model
def test_with_the_model_a_cough_never_starts_a_turn_without_it_it_does():
    cough = _cough() + _silence(0.6)
    with_model = TurnDetector(classifier=speech_model.SpeechClassifier(RATE))
    assert not any(isinstance(e, SpeechStarted) for e in with_model.process(cough))
    loudness_only = TurnDetector()
    assert any(isinstance(e, SpeechStarted) for e in loudness_only.process(cough))


@pytest.mark.speech_model
def test_a_session_builds_its_detector_with_the_model():
    from realtimevoice.session import _speech_classifier

    assert _speech_classifier(TurnDetectionConfig()) is not None
    assert _speech_classifier(TurnDetectionConfig(speech_check="loudness")) is None


async def test_a_client_can_choose_loudness_only_and_sees_the_choice(make_session):
    import asyncio

    session, driver = make_session()
    try:
        await asyncio.sleep(0)
        session.handle_event({"type": "session.update", "session": {"turn_detection": {"speech_check": "loudness"}}})
        updated = await driver.wait_for("session.updated")
        td = updated["session"].get("turn_detection") or updated["session"]["audio"]["input"]["turn_detection"]
        assert td["speech_check"] == "loudness"
        assert session._detector_signature[-1] == "loudness"
        driver.events.clear()
        session.handle_event({"type": "session.update", "session": {"turn_detection": {"speech_check": "maybe"}}})
        err = await driver.wait_for("error")
        assert "speech_check" in err["error"]["message"]
    finally:
        await driver.stop()
