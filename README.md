# somora_realtimevoice

The local voice server for [somora](https://github.com/thenaxon/somora_agent):
run somora's voice calls on your own speech recognition, language model and
speech output instead of OpenAI's realtime API.

It speaks the **OpenAI GA Realtime WebSocket protocol** (`/v1/realtime`) and
runs inference through **generic, configurable HTTP STT / LLM / TTS backends**
(for example an OpenAI-compatible server on your LAN). somora is its main
client, but any client that talks the public OpenAI Realtime protocol —
including a plain browser — can connect; backend credentials never reach
clients. To connect somora, see [`docs/somora.md`](docs/somora.md).

- Python 3.11+, asyncio, FastAPI/Starlette. Audio/CPU work runs off the event loop.
- Server-side VAD (loudness plus a small CPU speech model, so a cough is not speech), prebuffer, silence detection, barge-in.
- Cancellation fencing across STT, LLM and TTS; no stale audio after interrupt.
- Bounded buffers and bounded turn queues everywhere; the WebSocket receive loop never
  awaits backend work.
- Authenticated access (Bearer token) for HTTP and WebSocket, including the browser.
- A management UI at `/ui/`: Sets (what an agent speaks with), Voices, Models, a Test page and Settings.

Reference docs:

- [`docs/install.md`](docs/install.md) — installing and running: Docker, or
  directly on a VM as a service; HTTPS; first steps.
- [`docs/models.md`](docs/models.md) — what a model server must offer, every
  field of a model entry, and the models tested with their settings.
- [`docs/somora.md`](docs/somora.md) — connecting somora's voice calls.
- [`docs/sets.md`](docs/sets.md) — sets, realtime `model` namespaces,
  resolution rules and error codes.
- [`docs/diagnostics.md`](docs/diagnostics.md) — lifecycle metadata and the diagnostic
  scripts.

## Install

Docker or directly on any machine with Python 3.11+, no GPU needed; see
[`docs/install.md`](docs/install.md). For development:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
export REALTIMEVOICE_AUTH_TOKEN="choose-a-long-random-token"
.venv/bin/python -m realtimevoice --host 127.0.0.1 --port 8000
```

If `REALTIMEVOICE_AUTH_TOKEN` is not set, a random token is generated on first start
and stored (permission 0600) in `data/server.json`; pass `--print-token` once to show it.

`/` redirects to the management UI at `/ui/` (a secure browser context is required
for microphone access). Log in with the **instance token**:

1. Read the private runtime file `server.json` inside the instance data dir (for
   example with a local file viewer) — it contains a single `auth_token` value.
2. Copy **only** that `auth_token` value into the token field and log in. The
   browser keeps it in `sessionStorage` and sends it as a WebSocket subprotocol
   (`rtv.bearer.<token>`), so it never appears in request-line logs.
3. **Never** open or copy anything from `settings.json` in the same directory —
   that file holds backend credentials (masked in the UI, never shown).

Rotate the token by changing the env value or deleting `server.json` and
restarting the service. WebSocket clients authenticate with `Authorization:
Bearer`, `?token=` or the `rtv.bearer.<token>` subprotocol; access logs redact all
of them.

## Docker

```bash
docker build -t somora-realtimevoice .
docker run --rm -p 8000:8000 -e REALTIMEVOICE_AUTH_TOKEN=... -v "$PWD/data:/app/data" somora-realtimevoice
```

For a persistent single-host deployment (pinned base image, build revision,
bind-mounted data, healthcheck, backup/restore) use the tooling under
[`deploy/`](deploy/README.md).

## Backend adapters

STT/LLM/TTS backends are generic OpenAI-compatible HTTP adapters:

| Role | Endpoint used |
|------|---------------|
| STT  | `POST {base_url}/audio/transcriptions` (multipart, WAV) |
| LLM  | `POST {base_url}/chat/completions` (SSE streaming, tool calls) |
| TTS  | `POST {base_url}/audio/speech` (streaming WAV -> PCM16 24 kHz) |

Each model entry has `base_url`, `api_key` (stored server-side, masked in
responses), `model`, `language_format` (how it is told the call's language:
`de`, `German` or not at all), `voice` (TTS), `request_timeout_s` and `extra`
(arbitrary JSON body fields merged into requests). A speaking model also says
what it answers with (`audio_format`: WAV at any rate, raw 16-bit mono PCM at
`pcm_sample_rate`, or MP3, decoded with ffmpeg) and, if it clones, how it takes
the voice recording (`reference_audio_field` / `reference_text_field`, default
`ref_audio` / `ref_text`; `reference_audio_encoding`: plain base64 or a
`data:audio/wav;base64,…` URL). An OpenAI-style speech event stream
(`speech.audio.delta`) is read too, and `start_buffer_ms` gives a model that
sends its sound in bursts a head start per sentence; the next sentence is
always asked for while the current one is spoken.

Any server that speaks these OpenAI-style endpoints can be used; the models
tested so far, with the settings that work for each, are in
[`docs/models.md`](docs/models.md). Readiness probes call
`GET {base_url}/models`; when that list is non-empty and does not contain the
configured model, the probe fails with `model_not_listed`; an address that
lists no models (404/405) counts as reachable.

## Telling speech from noise

The server decides when you speak by loudness **and** a small speech model
(Silero VAD v5, MIT, shipped in `src/realtimevoice/speech_model/`, run with
onnxruntime on the CPU, a fraction of a millisecond per 20 ms of audio): a
cough, a knock or a chair is loud but not speech, so it neither interrupts an
answer nor reaches the speech recogniser, which would turn it into a word.
`session.turn_detection.speech_check` (Settings › Defaults for calls) can
switch back to loudness only; without onnxruntime the server falls back to it
and says so in its log.

## Sets

What a client calls is a **set**: a listening model, a thinking model, a
speaking model, a voice and a language, with optional call defaults. What is
saved is what calls use: a call resolves its set when it starts, against the
models and the voice recording as they are then, and keeps that copy for its
lifetime. A set that is on must be complete; a change elsewhere that would
stop it from working (a model edit, a new recording) is refused with the
reason.

Clients reach a set through the realtime `model` namespace:

| `model` value | Resolves to |
|---------------|-------------|
| `profile:<id>` | set `<id>` |
| `profile-test:<ticket>` | a short-lived test call, also for unsaved changes |
| `voice-route:auto` | the set named by the first `session.update` voice |
| `voice-route:<id>` | set `<id>` |
| anything else | the set chosen for other apps, else refused |

`GET /v1/models` lists the sets that are on. On connect the server attaches a
vendor-neutral `session.x_realtimevoice.route` block (set, the three model
ids, the resolved voice and language) so clients and the Test page can display
exactly what was bound — never guessed. See [`docs/sets.md`](docs/sets.md) for
the full API, resolution and error codes.

## Voice library

The management UI (`/ui/`) includes a voice library for reference-audio
cloning engines:

- **Create a voice** and add reference samples by **file upload** (WAV, 1–30 s,
  ≤ 10 MB) or by in-browser **microphone recording** (raw audio, no DSP).
- **Automatic transcription** of each sample through the listening model chosen
  in Settings; the transcript can be edited and confirmed. A confirmed reference
  sample is the precondition for a recorded voice in a set.
- **Replace the recording**: confirming a new recording's transcript makes it
  the one the voice speaks with; every set that speaks with the voice uses it
  from its next call.
- **Built-in voices**: the voices a speaking model has of its own are listed
  beside the recorded ones and can be heard.
- **Preview** the voice against a reference-audio TTS profile; the response
  carries server-side measurements (first-PCM latency, total time, audio
  duration).
- Samples are content-addressed (sha256) and stored under the data dir with
  private permissions; reference WAVs are never committed to git.

## Management API

All `/api` endpoints require the instance token. Highlights:

- Models: `GET /api/catalog`, `POST/PUT/DELETE /api/profiles/{stt|llm|tts}[/{id}]`,
  `…/clone`, `POST /api/probe-profile`, `POST /api/tts-synthesis`.
- Sets: list/create/get/save/delete, `…/validate`, `…/restore`,
  `…/test-session`, `…/test-speech`.
- Voices: `GET/POST/PATCH/DELETE /api/voices[/{id}]`, sample upload, sample
  audio, sample transcription, `…/preview`, `…/reference`.
- Server settings (call defaults, the set for other apps, the transcription
  model): `GET/PUT /api/settings`; health at `/api/health`.

## Upgrading

`settings.json` is upgraded automatically when the server starts; an exact
copy of the old file is kept beside it first (`settings.v2.backup.json`).

## Diagnostics and tests

- `scripts/diag_lifecycle_report.py` — turn lifecycle JSON logs into H1–H7
  per-session metrics.
- `scripts/diag_scenarios.py` — deterministic synthetic scenarios against fake
  backends.
- `scripts/two_profile_live.py` — two profiles in parallel against a running
  instance.
- `scripts/ui_smoke_browser.py` — headless browser smoke of the management UI.

See [`docs/diagnostics.md`](docs/diagnostics.md) for usage.

```bash
.venv/bin/pytest
.venv/bin/python scripts/ui_smoke_browser.py --out-dir /tmp/rtv-smoke
```

## Security notes

- The instance token is the only client credential. It is stored privately
  (`server.json`, 0600) and redacted from all logs.
- Backend API keys and reference material live server-side only. API responses
  mask `api_key` values and secret-looking `extra` fields; the TTS reference
  bytes are materialised in memory at connect time and never returned.
- Voice samples are content-addressed under the data dir with private
  permissions. Audio, credentials, backups and runtime state are excluded from
  version control (see `.gitignore`).
- The server exposes only the OpenAI Realtime WebSocket transport; there is no
  WebRTC sideband.

## License

MIT — see [LICENSE](LICENSE).