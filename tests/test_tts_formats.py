"""Speaking models that differ from the usual OpenAI-style server.

A cloning model may want the voice recording under other field names, or as
a data URL instead of plain base64; a model may answer with raw PCM or MP3
instead of WAV. Each is a setting of the model entry.
"""

import asyncio
import base64
import math
import shutil
import subprocess

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from realtimevoice.audio import PcmParser, decode_mp3
from realtimevoice.backends.openai_http import OpenAIHTTPTTS, _tts_payload
from realtimevoice.config import BackendProfile, BackendSettings, VoiceSample
from realtimevoice.profiles import materialise_tts_extra, profile_backend_settings


def _sample():
    return VoiceSample(id="smp_x", sha256="ab" * 32, source="upload", duration_ms=1000,
                       sample_rate=24000, size_bytes=4, transcript="hallo welt", created_at=0.0)


def _tone(ms=300, rate=24000):
    n = int(rate * ms / 1000)
    return (8000 * np.sin(2 * math.pi * 440 * np.arange(n) / rate)).astype("<i2").tobytes()


def test_the_recording_goes_into_the_fields_and_encoding_the_model_wants():
    row = BackendProfile(id="drama", role="tts", model="drama-3", reference_audio_field="speaker_wav",
                         reference_text_field="speaker_text", reference_audio_encoding="data_uri",
                         extra={"ref_audio": "OLD", "speaker_wav": "OLD", "max_new_tokens": 1500})
    tts = profile_backend_settings(row)
    extra = materialise_tts_extra(tts, object(), _sample(), sample_bytes=b"RIFF")
    assert extra["speaker_wav"] == "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode()
    assert extra["speaker_text"] == "hallo welt"
    # stored reference material, under either name, never goes along
    assert "ref_audio" not in extra
    assert extra["max_new_tokens"] == 1500


def test_by_default_the_recording_is_plain_base64_in_ref_audio():
    tts = profile_backend_settings(BackendProfile(id="c", role="tts", model="chatterbox"))
    extra = materialise_tts_extra(tts, object(), _sample(), sample_bytes=b"RIFF")
    assert extra == {"ref_audio": base64.b64encode(b"RIFF").decode(), "ref_text": "hallo welt"}


@pytest.mark.parametrize("name", ["", "input", "model", "voice", "has space", "x" * 70])
def test_a_recording_field_cannot_take_a_base_field_or_odd_name(name):
    with pytest.raises(ValidationError):
        BackendProfile(id="c", role="tts", reference_audio_field=name)


def test_the_answer_format_is_asked_for():
    s = BackendSettings(model="m", audio_format="pcm", pcm_sample_rate=44100)
    assert _tts_payload(s, "hi", None, None)["response_format"] == "pcm"
    assert _tts_payload(BackendSettings(model="m"), "hi", None, None)["response_format"] == "wav"


def test_raw_pcm_is_passed_on_in_whole_20_ms_blocks():
    p = PcmParser(44100)
    block = 882 * 2
    assert p.feed(b"\x01" * (block - 1)) == []
    assert p.feed(b"\x02" * 3) == [(44100, 1, b"\x01" * (block - 1) + b"\x02")]
    assert p.flush() == [(44100, 1, b"\x02\x02")]


@pytest.mark.parametrize("chunk", [2, 7, 1000, 4096])
def test_how_the_network_cuts_raw_pcm_does_not_change_the_audio(chunk):
    from realtimevoice.audio import resample_pcm16

    pcm = _tone(1000, 48000)
    p = PcmParser(48000)
    parts = []
    for i in range(0, len(pcm), chunk):
        parts += p.feed(pcm[i:i + chunk])
    parts += p.flush()
    out = b"".join(resample_pcm16(d, r, 24000) for r, _, d in parts)
    assert abs(len(out) // 2 - 24000) <= 1


def _speak(settings: BackendSettings, body: bytes) -> bytes:
    async def run():
        async def handler(request):
            return httpx.Response(200, content=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tts = OpenAIHTTPTTS(settings, client)
        out = b"".join([c async for c in tts.stream_speech("hallo")])
        await client.aclose()
        return out

    return asyncio.run(run())


def test_a_pcm_answer_at_44_1_khz_reaches_the_call_at_24_khz():
    pcm = _tone(1000, 44100)
    out = _speak(BackendSettings(base_url="http://t/v1", model="m", audio_format="pcm", pcm_sample_rate=44100), pcm)
    assert abs(len(out) / 2 - 24000) < 50


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_an_mp3_answer_is_decoded():
    mp3 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
         "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=_tone(1000), capture_output=True, check=True,
    ).stdout
    out = _speak(BackendSettings(base_url="http://t/v1", model="m", audio_format="mp3"), mp3)
    assert abs(len(out) / 2 - 24000) < 2400


def test_without_ffmpeg_an_mp3_answer_names_the_reason(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ffmpeg"):
        decode_mp3(b"ID3")


def test_the_recording_and_its_text_need_two_field_names():
    with pytest.raises(ValidationError):
        BackendProfile(id="c", role="tts", reference_audio_field="ref", reference_text_field="ref")


def test_a_recording_under_a_custom_field_name_is_masked_and_kept():
    from realtimevoice.catalog import mask_profile, merge_profile_masked

    row = BackendProfile(id="c", role="tts", reference_audio_field="speaker_wav",
                         reference_text_field="speaker_words", extra={"speaker_wav": "SECRET", "speaker_words": "TEXT", "seed": 1})
    view = mask_profile(row)
    assert view["extra"]["speaker_wav"] == "********" and view["extra"]["speaker_words"] == "********"
    assert view["extra"]["seed"] == 1
    # saving the masked view back keeps the stored values
    saved = merge_profile_masked(row, {"extra": view["extra"]})
    assert saved.extra["speaker_wav"] == "SECRET" and saved.extra["speaker_words"] == "TEXT"


def test_an_extra_response_format_decides_how_the_answer_is_read():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(_tone(1000))
    settings = BackendSettings(base_url="http://t/v1", model="m", audio_format="pcm",
                               extra={"response_format": "wav"})
    out = _speak(settings, buf.getvalue())
    # read as WAV: no header bytes in the audio
    assert len(out) == 48000


def test_renaming_a_recording_field_drops_what_was_stored_under_the_old_name():
    from realtimevoice.catalog import mask_profile, merge_profile_masked

    row = BackendProfile(id="c", role="tts", reference_audio_field="speaker_wav",
                         reference_text_field="speaker_words", extra={"speaker_wav": "SECRET", "seed": 1})
    saved = merge_profile_masked(row, {"reference_audio_field": "voice_wav"})
    assert "speaker_wav" not in saved.extra and saved.extra["seed"] == 1
    assert "SECRET" not in str(mask_profile(saved))


def test_other_answer_or_recording_settings_are_another_engine():
    from realtimevoice.engine_groups import engine_key

    a = BackendProfile(id="a", role="tts", base_url="http://t/v1", model="m")
    for change in ({"audio_format": "pcm"}, {"pcm_sample_rate": 44100}, {"reference_audio_field": "x"},
                   {"reference_text_field": "y"}, {"reference_audio_encoding": "data_uri"}):
        assert engine_key(a) != engine_key(a.model_copy(update=change))


def test_renaming_a_recording_field_with_the_masked_form_sent_back_drops_it_too():
    from realtimevoice.catalog import mask_profile, merge_profile_masked

    row = BackendProfile(id="c", role="tts", reference_audio_field="speaker_wav",
                         reference_text_field="speaker_words", extra={"speaker_wav": "SECRET", "seed": 1})
    form = mask_profile(row)
    saved = merge_profile_masked(row, {"reference_audio_field": "voice_wav", "extra": form["extra"]})
    assert "speaker_wav" not in saved.extra
    assert "SECRET" not in str(mask_profile(saved))


def test_a_recording_stored_under_a_custom_field_does_not_split_the_engine():
    from realtimevoice.engine_groups import engine_key

    a = BackendProfile(id="a", role="tts", base_url="http://t/v1", model="m", reference_audio_field="speaker_wav",
                       reference_text_field="speaker_words", extra={"speaker_wav": "A", "speaker_words": "a"})
    assert engine_key(a) == engine_key(a.model_copy(update={"extra": {}}))


def _sse(chunks, fmt="pcm", rate=44100):
    out = ""
    for c in chunks:
        ev = {"type": "speech.audio.delta", "audio": base64.b64encode(c).decode(), "response_format": fmt,
              "sample_rate": rate, "channels": 1}
        out += "event: speech.audio.delta\ndata: " + __import__("json").dumps(ev) + "\n\n"
    out += 'event: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n'
    return out.encode()


def _speak_sse(settings, body):
    async def run():
        async def handler(request):
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream; charset=utf-8"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        out = b"".join([c async for c in OpenAIHTTPTTS(settings, client).stream_speech("hallo")])
        await client.aclose()
        return out

    return asyncio.run(run())


def test_an_event_stream_of_pcm_is_read_as_audio_at_its_own_rate():
    from realtimevoice.audio import resample_pcm16

    pcm = _tone(1000, 44100)
    body = _sse([pcm[i:i + 4096] for i in range(0, len(pcm), 4096)], rate=44100)
    # configured for another rate: the event's own rate wins
    out = _speak_sse(BackendSettings(base_url="http://t/v1", model="m", audio_format="pcm", pcm_sample_rate=24000), body)
    expected = resample_pcm16(pcm, 44100, 24000)
    assert abs(len(out) - len(expected)) <= 4
    assert np.corrcoef(np.frombuffer(out[:len(expected)], "<i2"), np.frombuffer(expected[:len(out)], "<i2"))[0, 1] > 0.99


def test_an_event_stream_of_wav_is_read_as_audio():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(_tone(500))
    data = buf.getvalue()
    out = _speak_sse(BackendSettings(base_url="http://t/v1", model="m"), _sse([data[:44], data[44:]], fmt="wav", rate=24000))
    assert out == _tone(500)


def test_an_error_event_ends_speech_with_the_reason():
    from realtimevoice.errors import BackendError

    body = b'data: {"type":"error","error":{"message":"model not loaded"}}\n\n'
    with pytest.raises(BackendError, match="model not loaded"):
        _speak_sse(BackendSettings(base_url="http://t/v1", model="m"), body)


@pytest.mark.parametrize("status,ok", [(404, True), (405, True), (401, False), (500, False)])
def test_an_address_without_a_model_list_is_still_reachable(status, ok):
    from realtimevoice.backends.openai_http import _probe

    async def run():
        async def handler(request):
            return httpx.Response(status, json={"error": {"message": "x"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        res = await _probe("http://t/drama/v1", {}, client, "drama-3")()
        await client.aclose()
        return res

    assert asyncio.run(run()).ok is ok
