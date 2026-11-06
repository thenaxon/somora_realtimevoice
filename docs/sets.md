# Sets — API reference

A **set** (in the API: a conversation profile) is what an agent speaks with:
a listening model, a thinking model, a speaking model, a voice and a
language, with optional call defaults. Clients reach a set over the realtime
WebSocket through a `model` namespace; operators manage sets through the
management HTTP API and the UI at `/ui/`.

**What is saved is what calls use.** A call resolves its set when it starts,
against the model entries and the voice recording as they are at that moment,
and keeps that copy for its lifetime. Saving a set, editing a model entry or
choosing a new recording for a voice changes the next call, never a running
one.

All endpoints are under `/api` and require the instance bearer token (HTTP
`Authorization: Bearer …`, or the WebSocket subprotocols below).

## Data model

```
Set (ConversationProfile)
  id              immutable, the name clients use, [a-z0-9][a-z0-9-]{1,62}
  display_name    free text
  description     free text
  spec            ProfileSpec
  enabled         clients can call it (only with a complete spec)
  history         up to 10 earlier specs, newest first: {saved_at, spec}
  created_at, updated_at, spec_saved_at   epoch seconds

ProfileSpec
  stt_profile           id of a listening model entry
  llm_profile           id of a thinking model entry
  tts_profile           id of a speaking model entry
  voice                 {kind: "library", voice_id}
                      | {kind: "native",  native_voice}   a voice built into the model
                      | {kind: "none"}                    the model's own single voice
  language              the call's language code, e.g. "de"
  instructions_default  used only if the client sends no instructions
  turn_detection_default  partial turn-detection overrides
  output_pacing         "off" | "realtime"
```

A model entry (`/api/profiles/{stt|llm|tts}`) has `language_format`: how that
model wants the set's language — `code` (`de`), `name` (`German`; Qwen3-TTS
wants this) or `none` (not sent). A region (`de-AT`) is sent as its base
language. Model entry ids are derived from the model name when not given.
A speaking model also has `audio_format` (`wav` | `pcm` at `pcm_sample_rate`
| `mp3`) and, for cloning, `reference_audio_field`, `reference_text_field`
and `reference_audio_encoding` (`base64` | `data_uri`).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/conversation-profiles` | list all sets |
| POST   | `/api/conversation-profiles` | create a set |
| GET    | `/api/conversation-profiles/{id}` | one set |
| PUT    | `/api/conversation-profiles/{id}` | save `display_name`, `description`, `spec`, `enabled` |
| DELETE | `/api/conversation-profiles/{id}` | delete a set that is off |
| POST   | `/api/conversation-profiles/{id}/validate` | check the saved spec, or `{"spec": …}`, and probe its models |
| POST   | `/api/conversation-profiles/{id}/restore` | `{"index": n}`: an earlier version becomes the set again |
| POST   | `/api/conversation-profiles/{id}/test-session` | a `profile-test:<ticket>` model for the saved set or `{"spec": …}` |
| POST   | `/api/conversation-profiles/{id}/test-speech` | speak `{"text", "spec"?}` with the set's voice; WAV response |

A set view also carries `is_default` (it is the set for other apps),
`issues` (what keeps it from working now) and `compatibility_warnings`.

### Create and save

```http
POST /api/conversation-profiles
Authorization: Bearer <token>
Content-Type: application/json

{
  "id": "anna-agent",
  "display_name": "Anna",
  "spec": {
    "stt_profile": "whisper-large-v3-turbo",
    "llm_profile": "gemma-4-26b-a4b-it-4bit",
    "tts_profile": "qwen3-tts-12hz-1.7b-base-8bit",
    "language": "de",
    "voice": {"kind": "library", "voice_id": "anna"}
  },
  "enabled": true
}
```

A set that is **on** must be complete: saving or switching it on with a
blocking issue answers `409 set_invalid` with `issues`, and nothing is stored.
A set that is off can be saved half-done. Saving probes the three models and
returns the results as `probes`; a model that does not answer is reported,
never blocking. Each save that changes the spec keeps the previous spec in
`history`.

### Changes elsewhere

A change that would stop a switched-on set from working is refused with
`409 breaks_sets`; `issues` lists each affected set with its issues:

- a model entry edit (e.g. it no longer clones, or a lower recording limit),
- a new reference recording for a voice (an unconfirmed transcript, too long
  for the speaking model),
- an edited or re-transcribed reference transcript left unconfirmed.

Deleting a model entry a set uses, or the listening model that transcribes
recordings, answers `409 model_in_use`; deleting a voice a set speaks with
answers `409 voice_referenced`.

### Test tickets

`POST …/test-session` returns `{"model": "profile-test:<ticket>",
"expires_at": …}`, also for a set that is off. With `spec` the call uses what
the editor shows without saving it. Tickets are bounded (32) and expire after
15 minutes.

## Resolution over the realtime API

The public WebSocket is `/v1/realtime?model=<string>`. The `model` string is
resolved **before any backend call**; an unknown or switched-off set is
refused, never silently substituted.

| `model` value | Resolves to |
|---------------|-------------|
| `profile:<id>` | set `<id>` (must be on) |
| `profile-test:<ticket>` | the set behind the ticket, as the editor showed it |
| `voice-route:auto` | the set named by the first `session.update` voice (see below) |
| `voice-route:<id>` | set `<id>` |
| anything else, or `""` | the set for other apps (`default_set` in `/api/settings`); none chosen → refused with `no_set_selected` |

With `voice-route:auto` the session accepts the connection without models and
resolves the set from the first `session.update` that carries a voice; until
then `response.create` is answered with `voice_route_pending`. An unknown name
answers `unknown_voice_route` and the session stays pending. Once resolved,
the voice is fixed for the session: the same voice again is a no-op, another
one is refused with `voice_route_frozen`.

A set's language wins over a language the client asks for; the client gets
the `language_overridden_by_profile` warning.

### Session options

Two additive, conservative options are available to sets and clients (details
and diagnostics in `docs/diagnostics.md`):

- `turn_detection_default.barge_in_confirm_ms` (default `250`, `0` = immediate
  barge-in): a speech burst during an active response is held briefly and only
  cancels if it is sustained; a discarded short burst is rejected and the
  response resumes.
- `output_pacing` (`"off"` default | `"realtime"`): with `"realtime"`, audio is
  emitted at ≈ 1× real time with a lead, so `response.done` tracks actual
  playback and a barge-in can cancel the still-active response. A client may
  also request it per connection with
  `session.update {"session": {"x_realtimevoice": {"output_pacing": "realtime"}}}`.

### Session metadata

On `session.created`/`session.updated` the server attaches a vendor-neutral
extension (no secrets):

```json
{
  "x_realtimevoice": {
    "route": {
      "mode": "profile",
      "requested": "anna-agent",
      "profile": "anna-agent",
      "stt_model": "…", "llm_model": "…", "tts_model": "…",
      "voice": {"kind": "library", "voice_id": "anna", "sample_id": "smp_…", "ref_sha256": "…"},
      "language": "de"
    },
    "warnings": []
  }
}
```

`/v1/models` lists the sets that are on (`profile:<id>`) and
`voice-route:auto`, each with an `x_realtimevoice` block (display name,
language, `default`).

## Voices

A library voice has samples and one **reference recording**. At call start
the reference is materialised into the speaking model's request: the WAV
bytes in its recording field (default `ref_audio`, plain base64 or a data URL)
and the confirmed transcript in its text field (default `ref_text`); any
reference material stored on the model entry itself is never sent. A built-in
(`native`) voice and `none` send no reference material. Reference bytes never
appear in API responses, logs or the UI.

`PUT /api/voices/{id}/reference {"sample_id"}` makes a recording the
reference; every set speaking with the voice uses it from its next call. A
voice view lists the sets that speak with it in `used_by`. The reference of a
voice that sets speak with cannot be deleted (`sample_is_reference`).

Recordings are transcribed with the listening model chosen in the server
settings (`voice_library.transcription_stt_profile`), in the voice's
`language` (the language spoken in its recordings).

## Server settings

`GET/PUT /api/settings` covers what belongs to the server rather than to a
set: `session` (call defaults used when neither the client nor the set says
otherwise), `default_set` (the set for other apps) and `voice_library`
(the transcription model). Other keys are refused.

## Error codes

| Code | HTTP | Meaning |
|------|------|---------|
| `set_invalid` | 409 | a set that is on would be incomplete (see `issues`) |
| `set_incomplete` | 409 | a test call or test sentence with an incomplete set |
| `breaks_sets` | 409 | the change would stop a switched-on set from working (see `issues`) |
| `model_in_use` | 409 | a set (or recording transcription) uses this model entry |
| `voice_referenced` | 409 | a set speaks with this voice |
| `conversation_profile_active` | 409 | cannot delete a set that is on |
| `conversation_profile_exists` | 409 | id already in use |
| `no_history` | 409 | restore of a version that does not exist |
| `history_changed` | 409 | the set was saved while a restore was being checked; choose again |
| `unknown_conversation_profile` | 404/4404 | `profile:<id>` does not exist |
| `profile_not_available` | 404/4404 | the set is off |
| `no_set_selected` | 404/4404 | the client named no set and none is chosen for other apps |
| `unknown_voice_route` | 4404 | no set has the requested name |
| `voice_route_pending` | — | session error while a `voice-route:auto` route is unresolved |
| `unknown_profile` | 409/502 | a set uses a model entry that no longer exists |
| `sample_is_reference` | 409 | the reference recording of a voice that sets speak with |
| `no_transcription_model` | 409 | no listening model is chosen to transcribe recordings |
| `reference_file_missing` | 502 | the voice's recording file is missing at call start |

Validation issue codes: `missing`, `unknown_profile`, `missing_reference`,
`transcript_missing`, `transcript_unconfirmed`, `file_missing`,
`sha_mismatch`, `voice_incompatible`, `reference_too_long`,
`native_on_cloning_engine` (warning).

Probe error codes: `model_not_listed`, `backend_timeout`,
`backend_unreachable`, `probe_failed`, `unknown_profile`, `no_base_url`.

## Settings file

`settings.json` in the data dir is versioned. An older file is upgraded when
the server starts, after an exact copy of it is kept beside it
(`settings.v1.backup.json` / `settings.v2.backup.json`, written once).
Upgrading to version 3 keeps every model entry (with an id derived from its
model name), voice and set; a set takes the spec it was last switched on
with.
