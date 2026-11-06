# Connecting somora

somora_realtimevoice is the local voice server for
[somora](https://github.com/thenaxon/somora_agent)'s voice calls. somora
speaks the OpenAI Realtime protocol; pointed at this server instead of OpenAI,
a call runs on your own speech recognition, language model and speech output.

## 1. The server

Run the server ([install](install.md): Docker, or directly on a VM as a
service), log in to `/ui/` with the instance token and set up:

1. **Models**: one that listens (speech-to-text), one that thinks (a
   language model), one that speaks (text-to-speech), each an
   OpenAI-compatible HTTP server ([which ones, and their settings](models.md)). A speaking model shows what it does with
   voices: it clones them from a recording, has built-in voices, or speaks
   with one voice.
2. **Voices**: for a model that clones, record or upload a voice per agent
   that should sound different.
3. **Sets**: one per agent voice — its three models, its voice and its
   language. Switch it on and save: what is saved is what the next call uses.
   The set's name for somora is what you give the agent in somora.

## 2. somora

Put the instance token into a file only you can read:

```bash
install -m 600 /dev/null ~/.somora/secrets/realtimevoice.token
# paste the token from the server's data/server.json into it
```

Then in `~/.somora/config.yaml`:

```yaml
realtimeVoice:
  enabled: true
  provider: local
  url: ws://<server>:8766/v1/realtime    # wss:// behind the TLS container
  model: voice-route:auto
  apiKeyFile: ~/.somora/secrets/realtimevoice.token
  defaultVoice: <profile-id>
```

- **`provider: local`** keeps somora's OpenAI adapter but connects to `url`.
- **`apiKeyFile`** is required: the server refuses every connection without
  the instance token. somora sends it as `Authorization: Bearer`.
- **`model: voice-route:auto`** lets each agent pick its profile by its voice:
  somora sends the agent's voice (or `defaultVoice`) in its first
  `session.update`, and the server resolves it to the set with that name.
  Set an agent's voice to a set's name and that agent always speaks with it.
- **`model: profile:<name>`** binds every call to one set instead.

Any other `model` value gets the set chosen for other apps in the server's
Settings; with none chosen the call is refused (`no_set_selected`), so a typo
never lands on the wrong set unnoticed. The server's `session.updated` names
the set that was bound (`x_realtimevoice.route`).

## 3. What to expect

| In somora | On the server |
|---|---|
| The agent consults while the voice keeps talking | a function call, answered with `function_call_output` |
| You talk over the answer | `speech_started`, then the answer is cancelled |

Errors the server sends during a call carry `fatal: false` unless the session
cannot go on.

## 4. Checking the connection

`scripts/somora_client_harness.mjs` runs a call with somora's own client code
against an instance (see [diagnostics](diagnostics.md#somora-client-harness)):
connect, speak, a consult round trip, an interruption and a reconnect. With
`scripts/somora_harness_fake_instance.py` it needs no real backends.
