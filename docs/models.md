# Models

A set listens, thinks and speaks with three **model entries**. Each entry is
one model on one server: an address, a model name, a key, and how that
server wants to be talked to. The server runs none of these models itself;
any server that speaks the OpenAI-style HTTP endpoints below can be used, on
your LAN or in the cloud.

## What a model server must offer

| Role | Request | The server must |
|---|---|---|
| Listening (speech-to-text) | `POST {base_url}/audio/transcriptions` | take a WAV file (multipart, field `file`) and return `{"text": …}` (plain JSON, or an event stream) |
| Thinking (language model) | `POST {base_url}/chat/completions` | stream (`stream: true`, server-sent events); tool calls if your agents use them |
| Speaking (text-to-speech) | `POST {base_url}/audio/speech` | return WAV (any rate), raw 16-bit mono PCM, MP3, or an OpenAI-style event stream of those |

"Check connection" and the readiness check ask `GET {base_url}/models`. A
server that lists models must list the configured one; an address that
lists none (404) still counts as reachable.

## The fields of a model entry

The Models page explains each field next to it; this is the reference.

| Field | Meaning |
|---|---|
| Display name | what the entry is called on the pages |
| Server address (`base_url`) | usually ending in `/v1`; the request path is appended |
| Key (`api_key`) | sent as `Authorization: Bearer`; stays on the server, shown masked |
| Model name (`model`) | the name as that server knows it |
| Language as (`language_format`) | how the call's language reaches the model: `code` (`de`), `name` (`German`) or `none`. The language itself belongs to the set. A region (`de-AT`) is sent as its base language. |
| Timeout (`request_timeout_s`) | how long to wait for an answer |
| Extra fields (`extra`) | JSON merged into every request, e.g. `{"temperature": 0.7}`; overrides the fields above |
| Model thinking (listening: no; thinking only) | sets `chat_template_kwargs.enable_thinking`; the voice never speaks what a model thinks |

Speaking models also have:

| Field | Meaning |
|---|---|
| Voices | what a set can choose: clones from a recording, has its own voices, or speaks with one voice |
| Its own voices (`capabilities.voices`) | the names of built-in voices; offered in sets only when *voice names are reliable* (`native_voice_labels: honoured`) |
| Longest recording (`max_reference_ms`) | a cloning model that uses only the start of a recording; longer ones are refused with the offer to cut an excerpt; 0 = no limit |
| Recording fields | the request fields for the voice recording and its text, default `ref_audio` / `ref_text` |
| Recording sent as | plain base64, or a data URL (`data:audio/wav;base64,…`) |
| Answer format (`audio_format`) | `wav`, `pcm` (at `pcm_sample_rate`) or `mp3` (decoded with ffmpeg) |
| Head start (`start_buffer_ms`) | how much of each sentence is collected before it is spoken; for a model that sends its sound in bursts barely as fast as it is spoken |

Whatever you set, a call asks for the next sentence while the current one
is spoken, so a model's start-up time is not heard as a gap.

## Tested models

All of these ran behind an OpenAI-compatible server on Apple silicon
(mlx-audio / oMLX), some through a LiteLLM router. "In use" means used in
daily calls; "tried" means set up, checked and spoken with, but not used for
long.

### Listening

| Model | Status | Settings | Notes |
|---|---|---|---|
| Qwen3-ASR 1.7B (8-bit) | in use | Language as: code | 0.1–0.2 s for a spoken sentence (measured in calls). The `language German<asr_text>` prefix it adds is removed. |
| Whisper large-v3-turbo | in use | Language as: code | as fast; Whisper is known to make up words from silence and noise, which the speech check keeps away |
| Qwen3-ASR 0.6B (8-bit) | tried | Language as: code | good enough to transcribe voice recordings; for calls take 1.7B |

### Thinking

| Model | Status | Settings | Notes |
|---|---|---|---|
| Gemma 4 26B A4B (4-bit) | in use | no extra fields | with a low `max_tokens` (e.g. 200) longer answers stop mid-sentence; leave it out unless you need it |
| Qwen3 14B (4-bit) | tried | Model thinking: Off | thinks by default, which only adds delay in a call; what it thinks is never spoken |

somora sends its own instructions with every call, so the model mainly
needs to follow them and to call tools; a small model that is fast matters
more here than a large one.

### Speaking

| Model | Status | Voices | Settings |
|---|---|---|---|
| Qwen3-TTS 12Hz 1.7B **Base** (8-bit) | in use | clones from a recording; needs its text | Language as: **name**. Fast and steady; the safe choice. |
| Drama 3 (Fish-based, via a LiteLLM router) | in use | clones from a recording; needs its text | address of the router's cloning route (its general speech route drops the recording), Language as: **none**, Recording sent as: **data URL**, Answer format: **pcm** at **44100**, Head start **800 ms**, extra `{"max_new_tokens": 1500}`. Answers as an event stream: the first sound after 0.6–0.9 s, then faster than it is spoken; without the head start words break up (measured: 6 gaps in a 17 s answer without, none with 800 ms). Fish Audio Community License. |
| Chatterbox (fp16) | tried | clones from a recording | Language as: code, Longest recording **10 s** (it ignores the rest) |
| Qwen3-TTS 12Hz 1.7B **CustomVoice** (8-bit) | tried | its own voices (per its model card e.g. Vivian, Serena, Dylan, Eric, Ryan, Aiden) | Language as: **name**; list the voices and set *voice names are reliable* |
| Voxtral 4B TTS (4-bit) | tried | its own voices: `de_male`, `de_female`, `casual_male` | Language as: code; voice names are reliable |
| Fish S2 Pro (8-bit) | tried | one voice | Language as: code. Non-commercial licence; the installed server ignored a recording, so set it to "speaks with one voice". |

A good recording to clone from: 5–15 s, one person, no music or room echo,
spoken as the voice should sound, with a text that matches word for word.

## Should work, not tested here

OpenAI's own endpoints; vLLM, llama.cpp server and LM Studio for thinking;
faster-whisper / Speaches servers for listening; Kokoro-FastAPI and
openedai-speech for speaking (their own voices); LocalAI for all three.
Cloning servers that take the recording in other fields or as a data URL
work through the recording settings above. Services with their own API
(ElevenLabs, Azure, Google, Deepgram, Cartesia) need an adapter that does
not exist yet.

## Telling speech from noise

When a caller is speaking is decided by the loudness **and** a small speech
model (Silero VAD v5, shipped with the server, run on the CPU): a cough or a
knock is loud but not speech, so it neither interrupts an answer nor turns
into a word like "ja". Settings › Defaults for calls can switch back to
loudness only. How loud speech must be is a setting there too; somora sends
its own value with each call.
