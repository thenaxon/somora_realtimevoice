"""Server VAD with prebuffer and silence-based turn end.

A frame counts as speech when it is loud enough and, with a speech
classifier, when the classifier also hears speech in it (a cough or a knock
is loud but not speech).

process() is synchronous and cheap but is invoked from a thread executor by the
session's ingestor, so the event loop is never blocked by audio work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import TARGET_SAMPLE_RATE, frame_ms, rms


@dataclass
class BurstStats:
    """Per-burst VAD metadata (numbers only, aggregated over the burst).

    Never carries audio or content: RMS levels, durations and the outcome are
    the only members. `outcome` is one of ``stopped`` | ``discarded_short`` |
    ``max_len`` and mirrors what the detector decided.
    """

    speech_ms: float = 0.0
    silence_ms: float = 0.0
    peak_rms: float = 0.0
    mean_rms: float = 0.0
    frames: int = 0
    outcome: str = "stopped"
    gap_ms_since_prev_stop: float = -1.0


@dataclass
class SpeechStarted:
    prebuffer: bytes
    stats: BurstStats | None = None


@dataclass
class SpeechStopped:
    utterance: bytes
    stats: BurstStats | None = None


@dataclass
class BurstDiscarded:
    """A burst that met speech_started criteria but was too short to keep
    (``min_speech_ms``): the response it may have cancelled is NOT fed by it.
    Metadata only."""

    stats: BurstStats = field(default_factory=BurstStats)


@dataclass
class BufferCleared:
    pass


VadEvent = SpeechStarted | SpeechStopped | BurstDiscarded | BufferCleared


class TurnDetector:
    def __init__(
        self,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_ms_len: int = 20,
        threshold_rms: float = 350.0,
        start_frames: int = 2,
        prefix_padding_ms: int = 200,
        silence_duration_ms: int = 420,
        min_speech_ms: int = 100,
        max_utterance_ms: int = 60_000,
        max_prebuffer_ms: int = 1_000,
        tail_ms: int = 120,
        classifier=None,
        speech_probability: float = 0.5,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms_len / 1000) * 2
        self.frame_duration_ms = frame_ms_len
        self.threshold_rms = threshold_rms
        self.start_frames = max(1, start_frames)
        self.prefix_padding_ms = prefix_padding_ms
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms
        self.max_prebuffer_ms = max_prebuffer_ms
        self.tail_ms = tail_ms
        # optional speech classifier (feed(pcm) -> probability); with it a
        # loud frame is speech only if the classifier agrees. Inside a burst
        # a lower bar holds (the model dips between words).
        self.classifier = classifier
        self.speech_probability = speech_probability
        self.keep_probability = max(0.0, speech_probability - 0.15)
        # audio-domain bookkeeping survives clear() so the gap since the last
        # stopped burst stays measurable across a buffer clear
        self._fed_ms = 0.0
        self._last_stop_ms = -1.0
        self._reset()

    def _reset(self) -> None:
        self._pending = bytearray()
        self._prebuffer: list[bytes] = []
        self._speaking = False
        self._above = 0
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._utterance = bytearray()
        # per-burst diagnostic accumulators (numbers only)
        self._burst_frames = 0
        self._burst_energy_sum = 0.0
        self._burst_peak = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def voiced_ms(self) -> float:
        """Audio above the threshold in the burst that is open now (0 when
        none is): a burst stays open through its trailing silence, so being
        open says nothing about how much was actually said."""
        return self._speech_ms if self._speaking else 0.0

    @property
    def prebuffered_ms(self) -> float:
        return sum(frame_ms(len(f), self.sample_rate) for f in self._prebuffer)

    def _burst_stats(self, outcome: str) -> BurstStats:
        mean = self._burst_energy_sum / self._burst_frames if self._burst_frames else 0.0
        return BurstStats(
            speech_ms=round(self._speech_ms, 1),
            silence_ms=round(self._silence_ms, 1),
            peak_rms=round(self._burst_peak, 1),
            mean_rms=round(mean, 1),
            frames=self._burst_frames,
            outcome=outcome,
            gap_ms_since_prev_stop=(
                round(self._last_stop_ms, 1) if self._last_stop_ms >= 0 else -1.0
            ),
        )

    def clear(self) -> VadEvent:
        self._reset()
        return BufferCleared()

    def _keep_prebuffer(self, frame: bytes) -> None:
        self._prebuffer.append(frame)
        # keep ~prefix_padding_ms of pre-speech audio, never more than
        # max_prebuffer_ms; the oldest frames are dropped first
        target_ms = min(self.prefix_padding_ms, self.max_prebuffer_ms)
        while self.prebuffered_ms > target_ms and len(self._prebuffer) > 1:
            self._prebuffer.pop(0)

    def process(self, pcm: bytes) -> list[VadEvent]:
        events: list[VadEvent] = []
        self._pending.extend(pcm)
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[: self.frame_bytes])
            del self._pending[: self.frame_bytes]
            self._fed_ms += frame_ms(len(frame), self.sample_rate)
            event = self._process_frame(frame)
            if event is not None:
                events.append(event)
        return events

    def _voiced(self, frame: bytes, level: float) -> bool:
        if self.classifier is None:
            return level >= self.threshold_rms
        # the classifier sees every frame, so its state follows the audio
        prob = self.classifier.feed(frame)
        bar = self.keep_probability if self._speaking else self.speech_probability
        return level >= self.threshold_rms and prob >= bar

    def _process_frame(self, frame: bytes) -> VadEvent | None:
        level = rms(frame)
        voiced = self._voiced(frame, level)
        if not self._speaking:
            self._keep_prebuffer(frame)
            if voiced:
                self._above += 1
                if self._above >= self.start_frames:
                    self._speaking = True
                    self._utterance = bytearray(b"".join(self._prebuffer))
                    self._prebuffer = []
                    self._above = 0
                    self._silence_ms = 0.0
                    self._speech_ms = frame_ms(len(frame), self.sample_rate)
                    # the burst opens here: reset the diagnostic accumulators
                    # (the starting frame counts too)
                    self._burst_frames = 0
                    self._burst_energy_sum = 0.0
                    self._burst_peak = 0.0
                    self._account_frame(level)
                    return SpeechStarted(bytes(self._utterance), self._burst_stats("started"))
            else:
                self._above = 0
            return None

        self._utterance.extend(frame)
        duration = frame_ms(len(frame), self.sample_rate)
        self._account_frame(level)
        if voiced:
            self._speech_ms += duration
            self._silence_ms = 0.0
        else:
            self._silence_ms += duration
        if self._silence_ms >= self.silence_duration_ms:
            return self._finish("stopped")
        if frame_ms(len(self._utterance), self.sample_rate) >= self.max_utterance_ms:
            return self._finish("max_len")
        return None

    def _account_frame(self, level: float) -> None:
        self._burst_frames += 1
        self._burst_energy_sum += level
        if level > self._burst_peak:
            self._burst_peak = level

    def _finish(self, outcome: str) -> SpeechStopped | BurstDiscarded | None:
        self._speaking = False
        tail_frames = max(1, int(self.tail_ms / self.frame_duration_ms))
        if self._speech_ms < self.min_speech_ms:
            # too short to be speech: discard entirely, stay idle; the burst
            # is surfaced so the caller can log the discarding reason
            stats = self._burst_stats("discarded_short")
            self._last_stop_ms = self._fed_ms
            self._reset_speech()
            return BurstDiscarded(stats)
        stats = self._burst_stats(outcome)
        trimmed = self._trim_trailing_silence(bytes(self._utterance), tail_frames)
        self._last_stop_ms = self._fed_ms
        self._reset_speech()
        return SpeechStopped(trimmed, stats)

    def _reset_speech(self) -> None:
        self._utterance = bytearray()
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def _trim_trailing_silence(self, utterance: bytes, tail_frames: int) -> bytes:
        frame = self.frame_bytes
        n_frames = len(utterance) // frame
        cut = 0
        for i in range(n_frames - 1, -1, -1):
            if rms(utterance[i * frame : (i + 1) * frame]) >= self.threshold_rms:
                cut = min(i + 1 + tail_frames, n_frames)
                break
        if cut == 0:
            cut = min(tail_frames, n_frames)
        return utterance[: cut * frame]
