# Lifecycle diagnostics

The server emits structured, metadata-only lifecycle logs under the logger
`realtimevoice.lifecycle`. They exist to make interruption and duplication
behaviour measurable without ever logging content: no transcripts, prompts,
tool arguments, audio, tokens or secrets are written. Every value is a number,
boolean, id, sha8 prefix or fixed enum-like string, and each string is bounded
to 120 characters.

The bundled scripts read those logs (or reproduce scenarios) and turn them into
reports. Everything here is host-agnostic — the scripts take a file, stdin or a
URL argument and never embed private hosts or paths.

## The lifecycle log record

One JSON object per line:

```json
{"action":"response_done","audio_ms":1840,"generation":4,
 "response_id":"resp_…","session_id":"sess_…","status":"completed",
 "turn_epoch":2,"estimated_playback_tail_ms":0.0}
```

Base fields on every record: `session_id`, `action`, `generation`,
`turn_epoch`, and — when known — `event_id`.

### Actions

| Area | Actions |
|------|---------|
| Session / route | `session_open`, `session_closed`, `route_resolved`, `voice_route_resolved`, `voice_route_frozen`, `voice_route_unknown`, `backend_bound`, `language_overridden_by_profile` |
| VAD | `speech_started`, `speech_stopped`, `vad_burst`, `vad_burst_discarded`, `vad_reset`, `barge_in_pending`, `barge_in_rejected` |
| STT | `stt_started`, `stt_completed`, `stt_failed`, `stt_timeout`, `stt_result_superseded`, `turn_dropped`, `turn_queue_overflow` |
| Response | `response_reserved`, `response_started`, `response_done`, `response_cancelled`, `response_cancel_requested`, `response_create_rejected`, `response_failed`, `auto_response_queued`, `auto_response_flushed`, `auto_response_discarded`, `auto_response_dropped`, `pacing_dropped_audio_ms` |
| LLM | `llm_request`, `llm_finish`, `tool_call`, `tool_result_received` |
| TTS | `tts_started`, `tts_segment`, `tts_completed`, `tts_failed`, `tts_queue_discarded` |
| Errors | `error_emitted` |

### Selected fields

- Timing: `duration_ms`, `audio_ms`, `speech_ms`, `silence_ms`,
  `seg_start_ms`, `ms_since_last_response_done`, `first_audio_delta_at`,
  `last_audio_delta_at`, `estimated_playback_tail_ms`.
- VAD: `peak_rms`, `mean_rms`, `outcome`, `gap_ms_since_prev_stop`,
  `response_cancelled_by_burst`, `barge_in_pending`, `confirm_ms`,
  `held_audio_ms`.
- Response provenance (H4/H5): `source`, `has_instructions`,
  `instructions_sha8`, `new_items_since_last_response`, `new_items_by_type`,
  `last_item_type`, `client_event_id`, `duplicate_request_suspect`.
- Pacing (H1): `output_pacing`, `pacing_dropped_audio_ms`.
- LLM request shape: `role_sequence` (e.g. `s,s,u,a[tc],t` when one response
  produced text AND a tool call — one assistant message carries both),
  `message_count`, `tools_count`, `max_tokens`, `finish_reason`,
  `output_chars`, `tool_calls_count`.
- Quality signals (ratios only): `assistant_repeat_ratio`, `stt_echo_ratio`.
- TTS identity: `tts_profile`, `session_voice`, `payload_voice`, `ref_sha256`,
  `segment_index`, `chars`, `pcm_bytes`, `emoji_removed`.
- Bounded queue depths: `out_queue`, `turn_queue`, `audio_queue`.

The allowlist lives in `LIFECYCLE_FIELDS` in `src/realtimevoice/session.py`;
unknown fields are dropped before anything is logged.

## Session options for interruptions and duplicates

Two additive, default-conservative session options address the interruption and
duplication mechanisms. Both are exposed in `session.updated` and can be set
per connection through `session.update` (or a profile default; see
`docs/sets.md`).

- **Speech check** — `turn_detection.speech_check` (`"model"` default |
  `"loudness"`). With `"model"` a 20 ms frame counts as speech only when it is
  over `threshold_rms` and the speech model gives it a probability of at least
  0.5 (0.35 inside a burst, so short dips between words do not end it). The
  `speech_ms` of `speech_started` / `speech_stopped` / `vad_burst` counts those
  frames; a cough no longer shows up as a burst at all.
- **Soft barge-in** — `turn_detection.barge_in_confirm_ms` (default `250`,
  `0` restores the old immediate behaviour). On `speech_started` during an
  active response the server does not cancel immediately: it marks the response
  pending and holds new audio deltas in a bounded buffer.
  - speech continues for the whole window → cancel as before
    (`action: response_cancelled`, `reason: barge_in`), held audio dropped;
  - the burst is discarded as too short or speech stops below the window →
    the response resumes, held audio is flushed in order and
    `action: barge_in_rejected` is logged with `speech_ms` / `held_audio_ms`.
  `input_audio_buffer.speech_started` is still emitted immediately.
- **Real-time output pacing** — `output_pacing` (`"off"` default |
  `"realtime"`) with `output_pacing_lead_ms` (default `600`). With
  `"realtime"`, TTS still synthesizes ahead but audio is queued server-side and
  emitted at ≈ 1× real time with the configured lead, so `response.done` lands
  only when the last audio is due at the client (about audio duration − lead).
  A barge-in/cancel then drops the not-yet-sent audio and is effective; the
  dropped amount is logged as `action: pacing_dropped_audio_ms`. With `"off"`
  the emitted stream is byte-identical to before.

Protocol compatibility: these change only *when* audio and `response.done` are
sent and whether a tiny burst cancels; the event vocabulary, ids and payload
shapes are unchanged, so an OpenAI-realtime client (e.g. the Somora adapter)
needs no change. `x_realtimevoice.output_pacing` is accepted on `session.update`
as an optional extension.

## `scripts/diag_lifecycle_report.py`

Reads lifecycle JSON lines (a file, several files, or stdin — for example a
`journalctl` stream) and produces per-session metrics plus a short markdown
summary mapped to hypotheses H1–H7:

- H1 playback tail after `response.done`
- H2 VAD sensitivity / burst discards / epoch churn
- H3 TTS echo re-entering the mic
- H4 client response multiplication (including the `duplicate_request_suspect`
  count: a client `response.create` with no new user/tool item and no one-off
  instructions)
- H5 LLM context shape
- H6 TTS repetition inside a segment
- H7 `session.update` rebuilding the VAD detector mid-utterance

```bash
# from a log file
python scripts/diag_lifecycle_report.py journal.log --json /tmp/report.json --md /tmp/report.md

# from a live unit (stdout only; no secrets in the output)
journalctl --user -u realtimevoice.service -o cat | python scripts/diag_lifecycle_report.py -
```

Output contains numbers, ids and enums only — never transcript or prompt text.

## `scripts/diag_scenarios.py`

Runs deterministic synthetic scenarios against a locally started app with fake
STT/LLM/TTS backends over the real `/v1/realtime` WebSocket, reproducing the
mechanism behind H1, H1p, H2, H4, H5a, H5b and H7. Each scenario prints one
metrics JSON object (stdout, or `--out <file>`); those metrics can be piped
into `diag_lifecycle_report.py`. `H1p` and `H2` run the same scenario twice
with the new option off and on and report a before/after comparison (H1p:
pacing reduces the estimated playback tail; H2: a short burst is rejected and
the response survives).

```bash
python scripts/diag_scenarios.py                  # all scenarios
python scripts/diag_scenarios.py --scenario H4 --scenario H7
python scripts/diag_scenarios.py --out /tmp/scenarios.json
```

## `scripts/two_profile_live.py`

End-to-end acceptance harness for **two conversation profiles in parallel**
against a running instance. It synthesises one short utterance, resamples it to
PCM16 24 kHz, streams it to two sessions — session A over the late-bound
`voice-route:auto` + a `session.update` voice, session B over
`profile:<id>` — and proves both resolve to different route models and each
completes a transcription plus an audio response.

```bash
RTV_TOKEN=... python scripts/two_profile_live.py \
    --base-url http://127.0.0.1:8000 \
    --profile-a profile-a --profile-b profile-b \
    --tts-preview-profile tts-profile \
    --out-dir /tmp/m1-live-<ts>
```

Privacy: the token comes from `RTV_TOKEN` (never printed); transcript content
is reduced to length + sha8 before it touches disk. `summary.json` and
`events-metadata.jsonl` carry event types, ids and timings only — no transcript
text, audio or token. A silent stretch between frames does not end collection:
the harness waits until the overall deadline or a terminal `response.done`
(a slow LLM may think for many seconds before the first audio).

Exit codes: `0` pass, `3` assertion failures, `2` setup error.

## Somora client harness

`scripts/somora_client_harness.mjs` proves wire compatibility against the
**actual** Somora realtime client code instead of a hand-written client: it
dynamically imports Somora's own `OpenAiRealtimeProvider` /
`OpenAiRealtimeSession` and builds the `RealtimeSessionRequest` the way Somora's
call path does (its `buildVoiceInstructions` plus the consult / status / result
tool specs). Run it with Somora's own `tsx`:

```bash
<somora-root>/node_modules/.bin/tsx scripts/somora_client_harness.mjs \
  --somora-root <path> \
  --url ws://127.0.0.1:<port>/v1/realtime \
  --key-file <file> \
  --model voice-route:auto --voice <profile-id> \
  --wav <pcm16 24k mono wav> [--wav-barge <wav>] [--wav-tool <wav>] \
  [--consult-policy always|auto|substantive] [--agent-name Ada] \
  [--tool-attempts 2] [--tool-wait-ms 20000] [--barge-delay-ms 800] \
  --out-dir <new dir>
```

The instructions are built with the same consult policy a real agent voice
session gets (the agent's effective `persona.voice.consultPolicy ??
realtimeVoice.consultPolicy`), default `always`; `--agent-name` only sets the
persona name. Instruction length is logged, never its text.

Scenarios (each PASS / FAIL / NOT EXERCISED in `summary.json`):

1. `open()` → `configure()` `session.update` (as Somora sends it) → ready +
   route resolved (the wrapper / `opts.connect` hook captures the raw
   `session.x_realtimevoice.route` block).
2. Real-time-paced 20 ms chunks + 1.5 s silence via `sendAudio` → user
   transcript, model audio, `response.done`.
3. A consult tool call → `sendToolResult` → spoken follow-up; a `speak()`
   filler while a response is active is dropped, a `deliver:true` line is kept
   and spoken; client `response.create` count and duplicate suspicion. Whether
   the model calls the tool is a model decision recorded separately as
   `tool_call_observed`; up to `--tool-attempts` fresh sessions re-send the
   utterance, and if no call happens the transport is `NOT EXERCISED` (never
   FAIL).
4. Barge-in **during playback**: stream the question, wait for the first model
   audio (`model_speech start`), wait `--barge-delay-ms`, then stream
   `--wav-barge`. Asserts Somora `interrupted` + a client `response.cancel`,
   server `response.done cancelled`, zero audio deltas belonging to the
   cancelled response after its own `response.done`, and the next turn
   answered. If the response is already done before the barge window (H1
   playback tail with `output_pacing=off`) that is recorded as an observation
   with timings, not a failure.
5. Explicit `interrupt()` plus a reconnect with another voice/profile id →
   the route differs.

Exit codes: `0` full pass, `3` any FAIL, `4` NOT EXERCISED only (the model
decided not to act, or the profile's pacing made the barge window
inapplicable). `summary.json` carries the consult policy, agent name, the
instance `output_pacing`, per-scenario `tool_call_observed` / barge timings and
observations; `events.jsonl` carries event kinds and timings only — transcripts
as length + sha8, audio as delta counts; never text, audio or the token.

`scripts/somora_harness_fake_instance.py` starts a throwaway realtimevoice
instance plus fake OpenAI-compatible STT/LLM/TTS backends for validating the
harness without any private service. It needs a conversation profile + voice so
`voice-route:auto` resolves, writes a 0600 token file (the harness `--key-file`)
and a non-secret `*.config.json` beside it (WS URL, alternate voice, backend
control URL):

```bash
.venv/bin/python scripts/somora_harness_fake_instance.py \
  --voice harness-voice-a --voice harness-voice-b \
  [--output-pacing realtime] \
  --token-out /tmp/rtv-somora/token --data-dir /tmp/rtv-somora/data
```

The fake LLM emits a tool call only when the request's system instructions
contain the consult-policy line the harness asked for (stand-in for a real
model that consults only when told to), and the fake TTS streams real-time
audio long enough that a barge lands during playback. `--output-pacing realtime`
puts the H1 pacing on the profile so the harness can be run once with pacing off
and once on (`token.config.json` records the choice; the harness reads it into
`summary.json`). The harness pins `SOMORA_HOME` to a throwaway directory and
sets `SOMORA_LOG_LEVEL=silent` before importing anything from Somora, so
running it never writes into an operator's Somora home.

## Browser smoke

`scripts/ui_smoke_browser.py` starts a throwaway instance with a fake backend
and drives the management UI end to end in headless Chromium (login, voice
library upload / microphone / transcription / preview, set create, switch on
and save, Test page route display). Artifacts go to a run directory under
`/tmp`.

```bash
python scripts/ui_smoke_browser.py --out-dir /tmp/rtv-smoke-<ts>
```

Exit codes: `0` pass, `3` step failures, `2` setup error.