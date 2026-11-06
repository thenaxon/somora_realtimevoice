# Installing and running

somora_realtimevoice is one Python program (FastAPI + uvicorn). It runs the
same way in Docker, on a VM or on any Linux or macOS machine; Docker only
packs Python, its dependencies and ffmpeg into one image. It needs **no GPU**:
the models that listen, think and speak run on your model servers, and the
only model it runs itself, the speech check that tells speech from noise, is
a ~2 MB model that takes a fraction of a millisecond per 20 ms of audio on
one CPU core.

| | Docker | Directly (VM, server, laptop) |
|---|---|---|
| Needs | Docker with compose | Python 3.11 or newer; ffmpeg only for models that answer in MP3 |
| Updates | build a new image, switch the tag | `git pull`, `pip install`, restart the service |
| HTTPS | optional Caddy container in `deploy/` | a reverse proxy of your choice (examples below) |
| Backups | `deploy/backup-data.sh` | copy the data directory |

Either way, **all state lives in one data directory**; the code holds no
configuration of yours.

## What is in the data directory

| File | What it holds |
|---|---|
| `settings.json` | model entries (addresses, keys), sets, voices with their recording texts, server settings. Private (0600); keys never leave the server. |
| `server.json` | the access token for the UI and clients (0600) |
| `voices/<voice>/<sha256>.wav` | the voice recordings |
| `settings.v*.backup.json` | the exact old file, kept once when an update upgrades `settings.json` |

Back up the whole directory. Move it to another machine and the server runs
there as it did. Never put it in a git repository.

## Docker

Quick start:

```bash
docker build -t somora-realtimevoice .
docker run -d --name somora-realtimevoice -p 8766:8000 \
  -v "$PWD/data:/app/data" somora-realtimevoice
docker exec somora-realtimevoice python -m realtimevoice --print-token
```

For a lasting setup — a pinned base image, the build revision in
`/api/version`, a healthcheck, an optional TLS container, backup and restore —
use the tooling in [`deploy/`](../deploy/README.md): build from a commit with
`deploy/build-image.sh`, smoke-test the image with `deploy/smoke-image.sh`,
back up with `deploy/backup-data.sh --stop-container …`, then point
`RTV_IMAGE_TAG` in your `deploy.env` at the new image and
`docker compose … up -d`.

## Directly on a VM or server

```bash
sudo useradd --system --create-home --home-dir /opt/realtimevoice rtv
sudo -u rtv git clone https://github.com/thenaxon/somora_realtimevoice /opt/realtimevoice/app
cd /opt/realtimevoice/app
sudo -u rtv python3 -m venv .venv
sudo -u rtv .venv/bin/pip install .
# only if a speaking model answers in MP3:
sudo apt install ffmpeg
```

Create the data directory and print the access token (it is generated on
the first run and kept in `server.json`):

```bash
sudo -u rtv .venv/bin/python -m realtimevoice --data-dir /opt/realtimevoice/data --print-token
```

### As a systemd service

`/etc/systemd/system/realtimevoice.service`:

```ini
[Unit]
Description=somora_realtimevoice
After=network-online.target
Wants=network-online.target

[Service]
User=rtv
WorkingDirectory=/opt/realtimevoice/app
ExecStart=/opt/realtimevoice/app/.venv/bin/python -m realtimevoice --host 127.0.0.1 --port 8766 --data-dir /opt/realtimevoice/data
Restart=on-failure
RestartSec=3
# the token can also come from the environment instead of server.json:
# Environment=REALTIMEVOICE_AUTH_TOKEN=...

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now realtimevoice
journalctl -u realtimevoice -f
```

Bind to `127.0.0.1` and put a reverse proxy in front (next section), or to
the machine's LAN address if only your network reaches it. Never expose it
without the token; every HTTP and WebSocket route but `/api/health` requires
it.

### Updating

```bash
cd /opt/realtimevoice/app
sudo -u rtv git pull
sudo -u rtv .venv/bin/pip install .
sudo systemctl restart realtimevoice
```

Restarting ends running calls; check `GET /api/session-auth-check`
(`active_sessions`) first. If an update changes the settings format, the
server upgrades `settings.json` at start and keeps the exact old file beside
it; to go back to the older version, restore that file as well.

## HTTPS

Browsers allow the microphone only in a secure context: `https://`, or
`http://localhost`. The Test page and recording a voice therefore need HTTPS
when you open the UI from another machine. somora itself connects over
`ws://` on the LAN or `wss://` behind a proxy.

The proxy must pass WebSockets through. With Caddy:

```
voice.example.org {
    reverse_proxy 127.0.0.1:8766
}
```

With nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8766;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;
    client_max_body_size 20m;   # voice recordings are uploaded
}
```

The Docker tooling has the same as an optional Caddy container (`--profile
tls`, with your own certificate files or a DNS challenge).

## First steps

1. Open `https://<server>/ui/` and sign in with the token (`--print-token`,
   or the `auth_token` in `server.json`).
2. **Models**: add one that listens, one that thinks and one that speaks
   ([which ones, and how](models.md)). "Check connection" asks the server
   with what is on the form, before you save.
3. **Settings**: choose the listening model that transcribes voice
   recordings.
4. **Voices**: for a model that clones, create a voice, record or upload
   5–15 s of one person speaking clearly, and check the transcript.
5. **Sets**: create one per agent, pick how it listens, thinks and speaks,
   its voice and language, switch it **On** and save. "Try it" speaks a
   sentence or opens a test call with what is on screen.
6. **somora**: give each agent its set's name as its voice
   ([connecting somora](somora.md)).

## Checking a running server

| | |
|---|---|
| `GET /api/health` | liveness, no token |
| `GET /api/version` | version and build |
| `GET /api/session-auth-check` | `active_sessions`: calls running now |
| logs | one JSON line per call event (`realtimevoice.lifecycle`); no audio and no text, see [diagnostics](diagnostics.md) |
