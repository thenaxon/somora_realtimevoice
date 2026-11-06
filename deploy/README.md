# Deployment (Docker)

Host-neutral tooling for running the app on a single internal LAN Docker host.
No host addresses, paths, tokens or credentials live in these files: every
host-specific value is an environment variable with a neutral example, and the
runtime data directory is mounted from the host (never a named volume).

## Names

| Object | Name |
|--------|------|
| Compose project | `somora-realtimevoice` |
| App container | `somora-realtimevoice` |
| TLS container | `somora-realtimevoice-tls` |
| Image repository | `somora-realtimevoice:<shortsha>` |

## Layout

| Path | Purpose |
|------|---------|
| `compose.yaml` | `app` service (published on an explicit bind address) + optional `tls` profile |
| `deploy.env.example` | template for the required/optional environment variables |
| `build-image.sh <commit>` | builds from `git archive <commit>` (never the worktree) |
| `smoke-image.sh <image> [port]` | starts the image on loopback and checks `/api/health` |
| `compose.override.example.yaml` | optional host-local override template (e.g. bridge subnet) |
| `backup-data.sh <data_dir> <backup_dir> [--stop-container <name>]... [--allow-running]` | consistent `tar.gz` + `.sha256` + manifest |
| `restore-data.sh <backup.tar.gz> <new_empty_dir>` | verified restore into an empty dir |
| `data_manifest.py <data_dir> [--compare <m>]` | secret-free inventory (counts/ids/hashes only) |
| `build-tls-image.sh [tag]` | builds the DNS-01 TLS image (Caddy + Cloudflare plugin) |
| `check-cert.sh <hostname> <addr:port>` | prints served-cert subject/SAN/notAfter and days left |
| `tls/Caddyfile.files` | TLS terminator, mode `files` (operator-provided certificate) |
| `tls/Caddyfile.dns-cloudflare` | TLS terminator, mode `dns-cloudflare` (ACME DNS-01) |
| `tls/Dockerfile.dns-cloudflare` | xcaddy build of Caddy with `caddy-dns/cloudflare` |

## Build

```bash
deploy/build-image.sh "$(git rev-parse HEAD)"
```

The script refuses a dirty working tree and an unknown commit, exports the
commit via `git archive`, passes `--build-arg REVISION=<shortsha>` and tags
`somora-realtimevoice:<shortsha>`. The image bakes that revision into
`REALTIMEVOICE_REVISION`, so `/api/version` reports it without a `.git`
checkout. The base image is pinned by digest in the `Dockerfile`.

Smoke-test a built image (removes the container and its temp data dir on exit):

```bash
deploy/smoke-image.sh somora-realtimevoice:<shortsha>
```

## Run

```bash
cp deploy/deploy.env.example deploy/deploy.env
# edit deploy/deploy.env: RTV_IMAGE_TAG, RTV_BIND_ADDR, RTV_DATA_DIR
docker compose --env-file deploy/deploy.env up -d
```

Required variables are guarded (`${VAR:?}`) so a missing value fails the
compose parse instead of silently binding to every interface. The app listens
on `:8000` inside the container; only the explicit `${RTV_BIND_ADDR}` is
published on the host (HTTP default `8766`).

The healthcheck calls the unauthenticated liveness path `/api/health` with the
interpreter's stdlib (the slim image has no `curl`). `/api/version` and the
management APIs remain token-protected.

### Optional host-local compose override

For host-specific tweaks (for example a dedicated bridge subnet) keep an
override file next to — not instead of — `compose.yaml` and pass both:

```bash
docker compose -f deploy/compose.yaml -f deploy/compose.override.yaml \
    --env-file deploy/deploy.env up -d
```

`deploy/compose.override.example.yaml` is a template with a placeholder subnet.
Copy it to `deploy/compose.override.yaml` (gitignored) and edit. Some hosts keep
stale Docker raw `PREROUTING ... DROP` filter rules for the IP addresses of old
containers (for instance after a subnet rename); a dedicated subnet with a
stable address range avoids reusing an address that is still being dropped.

### Optional TLS

Browser access is served over **publicly trusted HTTPS** for a configurable
LAN hostname (in production resolved only by internal DNS to the LAN address).
Certificates come from Let's Encrypt via **DNS-01 only** — there is no internal
CA and nothing to trust manually. The profile is disabled by default:

```bash
docker compose --profile tls --env-file deploy/deploy.env up -d
```

Set `RTV_TLS_HOSTNAME` to the hostname and `RTV_TLS_MODE` to one of:

**`files`** (default; no DNS API needed). Obtain/renew the certificate with a
DNS-01 challenge elsewhere, copy the two files into `${RTV_TLS_DIR}/certs`
read-only (`fullchain.pem`, `privkey.pem`, mode 0600), and reload without
downtime:

```bash
docker exec somora-realtimevoice-tls caddy reload --config /etc/caddy/Caddyfile
```

Use a digest-pinned stock Caddy for `RTV_TLS_IMAGE` in this mode, e.g.
`caddy:2.10.0-alpine@sha256:<digest>`.

**`dns-cloudflare`** (automatic renewal once a token exists). Build the image
with the Cloudflare plugin:

```bash
deploy/build-tls-image.sh            # -> somora-realtimevoice-tls:<caddy-version>
```

Set `RTV_TLS_IMAGE=somora-realtimevoice-tls:<caddy-version>` and provide the
API token only through `${RTV_TLS_DIR}/cloudflare.env` (0600, never in the
repo):

```bash
CF_API_TOKEN=<cloudflare-token-with-DNS-edit>
```

Also set `RTV_ACME_EMAIL`; Caddy stores its ACME account/certificates in the
bind mount `${RTV_TLS_DIR}/caddy-data`. The token file is optional in `files`
mode (`required: false`).

Check the served certificate and get a non-zero exit below
`RTV_CERT_MIN_DAYS` (default 21) for monitoring:

```bash
deploy/check-cert.sh "$RTV_TLS_HOSTNAME" "$RTV_BIND_ADDR:8444"
```

Server-to-server clients (for example Somora) bypass the terminator and use
the app port directly: `ws://<lan-addr>:${RTV_HTTP_PORT}/v1/realtime`.

## Backup and restore

A backup is crash-consistent, content-checked and secret-free:

```bash
# hot snapshot (app keeps running) — consistent: true, but the archive may
# capture an in-flight write; prefer the stopped variant before a migration
deploy/backup-data.sh /srv/realtimevoice/data /srv/realtimevoice/backups

# consistent snapshot: stop the container that writes data, archive, restart
deploy/backup-data.sh /srv/realtimevoice/data /srv/realtimevoice/backups \
    --stop-container somora-realtimevoice
```

### Stop semantics

`--stop-container <name>` (repeatable; the app container
`somora-realtimevoice` is the one that writes data) stops each named container
that is currently running with `docker stop --time 20`, archives the data dir,
then starts **exactly the containers it stopped** again. The restart runs from
an `EXIT`/`ERR` trap, so it also happens when archiving fails, and the script
verifies each container is running again afterwards.

If a requested stop **fails**, the script aborts with a non-zero exit and
creates **no archive** — a "consistent snapshot before migration" can never
silently become an inconsistent one. To override deliberately, pass
`--allow-running`: the archive is written and its manifest records
`"consistent": false`. A manifest records `"consistent": true` when the app was
stopped for the archive, and also for a plain backup with no stop requested.

The old `--stop <compose-project-dir>` flag is rejected with a clear error: it
ran `docker compose stop` without the operator's `--env-file`/override files, so
a failed interpolation could yield a warning followed by an inconsistent
archive.

Each run writes `rtv-data-<UTC>.tar.gz` (0600), its `.sha256` (0600) and a
`.manifest.json` produced by `data_manifest.py`. The manifest contains only
counts, public ids, per-sample presence/hash and total bytes — never a
credential, token, transcript or URL. It exits non-zero if a sample file is
missing or its bytes no longer match the content-addressed filename.

Retention: only files matching this tool's own `rtv-data-*` pattern and older
than `RTV_BACKUP_KEEP_DAYS` (default 30) are deleted from the backup directory.

Restore into a **new, empty** directory (live data is never overwritten):

```bash
deploy/restore-data.sh /srv/realtimevoice/backups/rtv-data-<UTC>.tar.gz \
    /srv/realtimevoice/data-restored
```

The script verifies the checksum before extracting, refuses a non-empty
target, then re-runs `data_manifest.py --compare` against the stored manifest.

## Notes

- No named volumes are declared, so `docker compose down -v` cannot delete the
  runtime data; it lives only in the host bind mount.
- Container logs use the `json-file` driver with `max-size`/`max-file` rotation.
- This stack is self-contained: it references no other compose project.
