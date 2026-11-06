from realtimevoice.vad import SpeechStarted, SpeechStopped, TurnDetector

from fake_backends import silence_pcm, tone_pcm


def make_detector(**kwargs):
    defaults = dict(
        silence_duration_ms=300,
        prefix_padding_ms=200,
        threshold_rms=350.0,
        frame_ms_len=20,
    )
    defaults.update(kwargs)
    return TurnDetector(**defaults)


def feed(detector, pcm):
    events = []
    for spec in detector.process(pcm):
        events.append(spec)
    return events


def test_silence_produces_no_events():
    d = make_detector()
    assert feed(d, silence_pcm(1000)) == []
    assert not d.speaking


def test_speech_burst_full_turn():
    d = make_detector()
    events = feed(d, tone_pcm(400) + silence_pcm(500))
    started = [e for e in events if isinstance(e, SpeechStarted)]
    stopped = [e for e in events if isinstance(e, SpeechStopped)]
    assert len(started) == 1
    assert len(stopped) == 1
    # prebuffer contains the leading speech audio
    assert len(started[0].prebuffer) > 0
    # utterance contains ~400 ms of speech plus padding/tail
    assert len(stopped[0].utterance) >= 400 * 48


def test_prebuffer_included_at_start():
    d = make_detector(max_prebuffer_ms=1000)
    feed(d, silence_pcm(300))
    events = feed(d, tone_pcm(60))
    started = [e for e in events if isinstance(e, SpeechStarted)]
    assert len(started) == 1
    # prebuffer holds prefix_padding_ms (200 ms) of audio preceding the burst
    assert len(started[0].prebuffer) >= 200 * 48 * 0.9
    assert len(started[0].prebuffer) <= 210 * 48


def test_prebuffer_bounded_by_prefix_padding():
    d = make_detector(prefix_padding_ms=150, max_prebuffer_ms=1000)
    feed(d, silence_pcm(800))
    events = feed(d, tone_pcm(60))
    started = [e for e in events if isinstance(e, SpeechStarted)]
    assert len(started) == 1
    assert len(started[0].prebuffer) <= 160 * 48
    assert len(started[0].prebuffer) >= 140 * 48


def test_short_noise_burst_ignored():
    d = make_detector(min_speech_ms=100)
    events = feed(d, tone_pcm(40) + silence_pcm(500))
    assert not any(isinstance(e, SpeechStopped) for e in events)
    assert not d.speaking


def test_utterance_bounded_by_max_length():
    d = make_detector(max_utterance_ms=1000)
    events = feed(d, tone_pcm(5000))
    stopped = [e for e in events if isinstance(e, SpeechStopped)]
    assert stopped, "long continuous speech must be segmented"
    for event in stopped:
        assert len(event.utterance) / 48 <= 1100  # ms -> bytes/48
    total = sum(len(e.utterance) for e in stopped)
    assert total >= 5000 * 48 * 0.9  # no speech lost between segments


def test_continuous_speech_then_stop():
    d = make_detector()
    all_events = []
    all_events += feed(d, tone_pcm(500))
    all_events += feed(d, silence_pcm(100))
    all_events += feed(d, tone_pcm(200))
    all_events += feed(d, silence_pcm(600))
    stopped = [e for e in all_events if isinstance(e, SpeechStopped)]
    assert len(stopped) == 1
    # one utterance spanning both bursts
    assert len(stopped[0].utterance) >= 700 * 48


def test_clear_resets_state():
    d = make_detector()
    feed(d, tone_pcm(200))
    assert d.speaking
    d.clear()
    assert not d.speaking
    assert feed(d, silence_pcm(400)) == []
