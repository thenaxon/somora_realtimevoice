import struct

import pytest

from realtimevoice.audio import (
    TARGET_SAMPLE_RATE,
    WavParser,
    decode_base64_pcm16,
    encode_base64_pcm16,
    pcm16_to_wav,
    resample_pcm16,
    to_mono,
    wav_header,
)


def test_pcm_base64_roundtrip():
    pcm = b"\x01\x00\x80\xff"
    encoded = encode_base64_pcm16(pcm)
    assert decode_base64_pcm16(encoded) == pcm


def test_decode_odd_length_trims():
    assert len(decode_base64_pcm16("AAAB")) == 2  # 3 bytes -> 2


def test_resample_identity_and_upsample():
    pcm = struct.pack("<10h", *range(-5, 5))
    assert resample_pcm16(pcm, 24000, 24000) == pcm
    up = resample_pcm16(pcm, 12000, 24000)
    assert len(up) == 2 * len(pcm)


def test_resample_downsample_length():
    pcm = bytes(2 * 4800)  # 200 ms at 24 kHz
    out = resample_pcm16(pcm, 24000, 16000)
    assert len(out) == 2 * 3200


def test_to_mono_averages_channels():
    stereo = struct.pack("<4h", 100, 200, -100, -200)
    mono = to_mono(stereo, channels=2)
    assert mono == struct.pack("<2h", 150, -150)


def test_wav_roundtrip():
    pcm = bytes(960)
    wav = pcm16_to_wav(pcm, 24000, 1)
    parser = WavParser()
    out = parser.feed(wav)
    assert out == [(24000, 1, pcm)]


def test_wav_header_lengths():
    header = wav_header(960, 24000, 1)
    assert len(header) == 44
    assert header[:4] == b"RIFF"


def test_wav_parser_split_header():
    pcm = struct.pack("<4h", 1, 2, 3, 4)
    wav = pcm16_to_wav(pcm, 16000, 1)
    parser = WavParser()
    assert parser.feed(wav[:3]) == []
    assert parser.feed(wav[3:30]) == []
    out = parser.feed(wav[30:])
    assert out == [(16000, 1, pcm)]


def wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1) -> bytes:
    byte_rate = rate * channels * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, channels, rate, byte_rate, channels * 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def test_wav_parser_byte_by_byte():
    pcm = struct.pack("<8h", *range(8))
    wav = wav_bytes(pcm)
    parser = WavParser()
    collected = []
    for i in range(len(wav)):
        collected.extend(parser.feed(wav[i : i + 1]))
    collected.extend(parser.flush())
    assert all(rate == 24000 and channels == 1 for rate, channels, _ in collected)
    assert b"".join(p for _, _, p in collected) == pcm


def test_wav_parser_streamed_data_with_unknown_size():
    pcm = struct.pack("<4h", 5, 6, 7, 8)
    raw = wav_bytes(pcm)
    parser = WavParser()
    assert parser.feed(raw[:20]) == []
    out = parser.feed(raw[20:])
    assert out == [(24000, 1, pcm)]


def test_wav_parser_rejects_non_wav():
    parser = WavParser()
    with pytest.raises(ValueError):
        parser.feed(b"NOTWAVES" + bytes(40))


def test_wav_parser_stereo_bigger_chunks():
    from realtimevoice.audio import to_mono

    pcm = struct.pack("<8h", 1000, -1000, 2000, -2000, 3000, -3000, 4000, -4000)
    raw = wav_bytes(pcm, channels=2)
    parser = WavParser()
    out = parser.feed(raw)
    rate, channels, stereo = out[0]
    assert (rate, channels) == (24000, 2)
    assert stereo == pcm
    mono = to_mono(stereo, channels=2)
    assert len(mono) == len(pcm) // 2


def test_target_rate():
    assert TARGET_SAMPLE_RATE == 24000
