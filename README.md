# Pictura MCP

A local, GPU-backed image generation MCP server. Works with any MCP client
(Claude Desktop, Cursor, VS Code, Windsurf, Claude Code, …).

```
MCP client ──────────────────▶ pictura_server.py ──────────────▶ GPU
            (stdio|HTTP|SSE)                      (SDXL/SD3.5/Qwen-Image 2.1)
```

- **Server**: `server/pictura_server.py` — a standard MCP server (stdio /
  streamable HTTP / SSE) running Stable Diffusion (SDXL / SD3.5) or
  Qwen-Image 2.1 via Hugging Face `diffusers`.
- **Client**: any MCP client; see § [Client configuration](#client-configuration).

## Run modes

Start here. Which of these is you?

- **Local** — try it on this machine: your MCP client launches the server as
  a child process (stdio). Follow **§Local (stdio)**.
- **Remote** — an always-on server that clients reach over the network,
  deployed as a systemd service. The other path, §Local (stdio), prepares a
  repo `.venv` + client config; a remote deployment is self-contained:
  follow **§Remote (http)** below from start to finish; it creates
  everything it needs under `/opt` / `/etc` and does not use anything from
  the local path.
- **The server is already running somewhere** — skip both; you only need
  §Client configuration.

Not sure yet? Do the local path first — §Local (stdio) shows the server
generating an image; having done it or not makes no difference to a later
§Remote (http) deployment (the two paths do not share `.venv` or config).

## Tools

| Tool | Description |
|---|---|
| `generate_image` | text → image (model-family defaults) |
| `edit_image` | image → image (img2img / edit from a prompt) |
| `upload_image` | local file → upload reservation (one-time token + POST URL; http/sse; stdio: pass the path to `edit_image`) |
| `list_loras` | supported LoRA ids with descriptions (for `lora`) |
| `list_control_types` | abstract ControlNet types (for `control_type`); not registered for families without `control_types` in the config |
| `server_status` | model + image-size policy (stdio adds device / VRAM etc.) |

Both generation tools accept optional **LoRA** adapters, and `edit_image`
additionally supports **ControlNet** where the model family provides it (see
§ [LoRA & ControlNet](#lora--controlnet)). Parameter meanings come with each
tool's input schema; `list_loras` / `list_control_types` list the valid
values; ControlNet/LoRA model ids stay server-side in `server/model.json`.

Images are **never written to the server disk** (stateless):

- **stdio** (local): inline base64 `ImageContent` + a `TextContent` note.
- **http/sse** (remote): the note contains a short-lived download URL
  (`.../images/<id>`, default TTL 600 s) served from an in-memory cache; the
  client fetches and saves it. Local files are uploaded over http/sse via the
  `upload_image` tool (one-time `X-UPLOAD-TOKEN`; see §Image editing).

**Concurrency**: rendering runs on a pool of **slots** (independent pipeline
instances per slot, so LoRA/offload state never races). Pool size is derived
from measured free VRAM; override with `PICTURA_MAX_CONCURRENT`. With
several CUDA devices visible (`PICTURA_CUDA_DEVICE=0,1`), the Qwen-Image 2.1
pipeline is loaded model-parallel across the cards instead of one card +
CPU offload; set `PICTURA_MAX_CONCURRENT=2` to run two independent slots
(one per GPU-pair, best for throughput).

## Local (stdio)

What you get: a repo-local `.venv`, `server/model.json` and a client config,
so the server runs from this repo. Expect a few GB of downloads (torch
CUDA build ~3 GB + model weights); the first generation loads the model and
can take a minute. Every step is safe to re-run.

```bash
# 1) Python venv + dependencies (torch CUDA build, ~3 GB)
#    prereq: C toolchain + Python 3 dev headers for the interpreter used
#    below (Debian/Ubuntu: build-essential python3-dev; Fedora/RHEL: gcc python3-devel)
#    V100 / Volta (compute capability 7.0) machines:
#      ./.venv/bin/pip install -r server/requirements-v100.txt
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt

# 2) Activate a model preset -> server/model.json
cp server/examples/model.sd35-large.json server/model.json
#    other presets: model.sdxl.json / model.sd35-medium.json /
#    model.qwen-image-2.1.json / model.qwen-image-2.1-turbo.json

# 3) Provision the models
scripts/fetch-models.sh

# 4) Smoke test (writes outputs/smoke_test.png)
./.venv/bin/python server/pictura_server.py --smoke

# 5) Connect from your MCP client (see below)
```

You're done when step 4 prints success and `outputs/smoke_test.png` exists.
If it fails: the CUDA driver / torch build is incompatible (§CUDA driver
version), or step 3 left `models/` incomplete — re-run it.

Disk layout (this path) — per-machine files created in the repo
(gitignored): `.mcp.json` (from `deploy/mcp.json.example`),
`server/model.json` (copied preset), `.venv/`, `outputs/`. In a §Remote
(http) deployment the same roles live under `/opt/pictura-mcp` and
`/etc/pictura-mcp` instead.

### CUDA driver version

`requirements.txt` pulls the latest torch from PyPI (CUDA 13 wheels, driver
≥ 580 — check with `nvidia-smi`). **V100 (Volta, compute capability 7.0)**
machines must use `server/requirements-v100.txt` instead: newer torch builds
dropped the sm_70 kernels and generation fails with
`CUDA error: no kernel image is available`.

## Client configuration

Pick the block that matches how the server runs — **Local (stdio)** or
**Remote (http)**. Both are just connection info for your MCP client; the
server is what has to be running first.

**Local (stdio)** — client on the same machine:

```jsonc
{
  "mcpServers": {
    "pictura": {
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["<PROJECT_ROOT>/server/pictura_server.py"],
      "env": { "PICTURA_MODEL_CONFIG": "<PROJECT_ROOT>/server/model.json" },
      "requestTimeoutMs": 600000
    }
  }
}
```

**Remote (http)** — a systemd service (§Remote (http)): see
`deploy/mcp.remote.json.example` — `url` + `PICTURA_API_KEY` header instead of
`command`/`args`.

`requestTimeoutMs` must allow for GPU rendering time: a generation at the
family default (1024², 30–40 steps) takes tens of seconds to minutes. The MCP
SDK default (60 s) times out on ordinary generations; use the templates'
600000 (10 min).

| Client | Config location |
|---|---|
| Claude Code, Cursor, Windsurf, VS Code, generic | project `.mcp.json` (template: `deploy/mcp.json.example`) |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| VS Code | `.vscode/mcp.json` |

Quick start for any client that reads `.mcp.json`:

```bash
cp deploy/mcp.json.example .mcp.json   # then replace <PROJECT_ROOT>
```

## Remote (http)

The deployment path for an always-on server that clients reach over the
network: the HTTP/SSE server runs as a systemd service. Declarative: the
commands below run from the **repo root**
(`deploy/…` is relative to it), and every step is safe to re-run.

When you're done you'll have:

- **app** at `/opt/pictura-mcp` — from the repo only `server/`,
  `scripts/fetch-models.sh` and `README.md` are copied in (see
  `deploy/install-files.txt`); `.venv/` and `models/` are created on the host.
  Templates and docs stay in the repo.
- **configuration** at `/etc/pictura-mcp/env` (secrets, 0600 root:pictura-mcp)
- a dedicated, unprivileged **service account** `pictura-mcp` (no login)
- a hardened **unit** `/etc/systemd/system/pictura-mcp.service`; logs go to
  the journal

### Before you start

- **CUDA driver** working (`nvidia-smi`) and the build toolchain: a C
  compiler plus the Python 3 dev headers matching the interpreter used in
  step 1 (Debian/Ubuntu: `build-essential python3-dev`; Fedora/RHEL:
  `gcc python3-devel`)
- **an API key and port**, and — if clients connect over the network —
  a **`PICTURA_PUBLIC_URL`**: it is required for non-loopback binds (the
  server refuses to start without it)
- disk: a few GB (pip deps ~3 GB + model weights)

### 1) App, venv, weights

```bash
# app into /opt (source = this repo root; deploy/... is relative to it)
rsync -a --files-from=deploy/install-files.txt ./ /opt/pictura-mcp/

# model preset -> /opt/pictura-mcp/server/model.json (choose one from server/examples/)
cp server/examples/model.qwen-image-2.1-turbo.json /opt/pictura-mcp/server/model.json

# python venv + deps
python3 -m venv /opt/pictura-mcp/.venv
/opt/pictura-mcp/.venv/bin/pip install -r /opt/pictura-mcp/server/requirements.txt

# weights (downloads once)
/opt/pictura-mcp/scripts/fetch-models.sh
```

### 2) Service account

```bash
sudo install -d -m 755 /etc/sysusers.d
sudo cp deploy/sysusers.d/pictura-mcp.conf /etc/sysusers.d/
sudo systemd-sysusers
```

Creates `pictura-mcp` (home `/var/lib/pictura-mcp`, shell
`/usr/sbin/nologin`).

### 3) Configuration (secrets)

```bash
sudo install -d -m 750 -o root -g pictura-mcp /etc/pictura-mcp
sudo install -m 600 -o root -g pictura-mcp deploy/pictura-mcp.env.example /etc/pictura-mcp/env
```

Then **edit `/etc/pictura-mcp/env`** — this is the one file you'll keep
touching:

- `PICTURA_API_KEY` — replace the placeholder (clients authenticate with this)
- `PICTURA_HOST=0.0.0.0` / `PICTURA_PORT` — uncomment/adjust for network access
- `PICTURA_PUBLIC_URL` — externally visible base (required, see above)
- `PICTURA_MODEL_CONFIG` — can stay commented: the unit runs with
  `WorkingDirectory=/opt/pictura-mcp`, so the default `server/model.json`
  resolves to `/opt/pictura-mcp/server/model.json`
- `PICTURA_CUDA_DEVICE`, `PICTURA_MAX_CONCURRENT` — GPU / slot tuning, see
  REFERENCE §6

### 4) Unit + start

```bash
sudo install -m 644 deploy/pictura-mcp.service /etc/systemd/system/
# SELinux only — default-on in Fedora/RHEL; skip when `getenforce` shows Disabled
sudo restorecon -RFv /etc/pictura-mcp /etc/sysusers.d/pictura-mcp.conf \
           /etc/systemd/system/pictura-mcp.service
sudo systemctl daemon-reload && sudo systemctl enable --now pictura-mcp
```

### Verify it is up

```bash
systemctl is-active pictura-mcp     # expect: active
journalctl -u pictura-mcp -f        # watch until the "Model ready" line
```

If it is not active, check the journal: a placeholder `PICTURA_API_KEY` or a
missing `PICTURA_PUBLIC_URL` makes the server refuse to start, and step 1
must have completed (toolchain + weights). `env` and `model.json` are read
at startup — after changing either, `systemctl restart pictura-mcp`.

### Make it reachable

- open the port in the host firewall if one is active — examples:
  Fedora/RHEL `firewall-cmd --permanent --add-port=<port>/tcp && firewall-cmd --reload`,
  Debian/Ubuntu `ufw allow <port>/tcp`
- clients connect with `url` (the `PICTURA_PUBLIC_URL`) + the `PICTURA_API_KEY`
  header — template `deploy/mcp.remote.json.example`, see §Client configuration

### Updates

Re-run step 1 (rsync picks up new code/presets; re-run `fetch-models.sh` if
weights changed), then `systemctl restart pictura-mcp`. Configuration changes
touch `/etc/pictura-mcp/env` only.

## Model configuration

All model settings live in one file: `server/model.json` (gitignored; copy a
preset from `server/examples/` or point `PICTURA_MODEL_CONFIG` at it):

- `model` — base model: local path or `org/repo` id; family is auto-detected
  (`sdxl`, `sd35-medium`, `sd35-large`, `qwen-image-2.1`)
- `vae` — optional custom VAE id (`null` = auto)
- `families.<id>` — per-family settings: `desc`, `steps`, `guidance`,
  `width`/`height`, `buckets` (supported resolutions; rotations are added
  automatically), `auto_vae`, and optionally `turbo`
- `supported_loras` — id → description map (client-usable LoRAs; `*` = any id)
- `control_types` — `pre` / `model` / `prep_model` per abstract type
  (`canny` / `depth` / `openpose`); ids hidden from clients

The server refuses to start when the configured model is not a supported
family (exit 2).

**Turbo mode** (`families.<id>.turbo`) enables a few-step distilled adapter
(preset `model.qwen-image-2.1-turbo.json`, the Viggle Qwen-Image-2.1 turbo):
generations drop from 40 to **6 steps (~5× faster)** with quality close to
the base model. When the `turbo` block is set:

- the distilled LoRA (`lora`) is applied automatically at scale 1.0 on every
  generate/edit call — client `lora` parameters are ignored;
- generations run with the adapter's raw sigma schedule (`sigmas`), no CFG
  and no negative prompt (`guidance_scale` / `negative_prompt` ignored);
- the distilled scheduler config (`scheduler`, `shift_terminal: null`) is
  substituted at pipeline build time;
- `steps` in the family config becomes the turbo step count (6).

Provisioning pulls only the r256 LoRA and the scheduler config from the
Viggle repo (`scripts/fetch-models.sh` handles it via the `turbo` config).

**Model provisioning** is a deployment step — the server never contacts
Hugging Face; every weight is a plain local directory under
`PICTURA_MODELS_DIR` (default `<project>/models`):

```bash
scripts/fetch-models.sh   # base model + VAE + LoRA + ControlNet + preprocessors
```

Manually (`hf` ships in the project venv):

```bash
./.venv/bin/hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen/Qwen-Image-2.1
./.venv/bin/hf download SG161222/RealVisXL_V5.0 --local-dir models/SG161222/RealVisXL_V5.0
./.venv/bin/hf download stabilityai/stable-diffusion-3.5-large --local-dir models/sd35-large
./.venv/bin/hf download prithivMLmods/SD3.5-Large-Photorealistic-LoRA --local-dir models/prithivMLmods/SD3.5-Large-Photorealistic-LoRA
./.venv/bin/hf download diffusers-internal-dev/sd35-controlnet-depth-8b --local-dir models/diffusers-internal-dev/sd35-controlnet-depth-8b
./.venv/bin/hf download Intel/dpt-hybrid-midas --local-dir models/Intel/dpt-hybrid-midas
# openpose preprocessor (not an HF repo):
curl -fsSL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt -o models/yolov8n-pose.pt
```

Layout:

```
models/<name>/          base model directories (config "model" points here)
models/<org>/<repo>/    LoRA / ControlNet / preprocessor / repo-id base models
models/yolov8n-pose.pt  openpose preprocessor
```

Defaults per model family:

| Family | Default W×H | Steps | Guidance |
|---|---|---|---|
| SDXL | 1024×1024 | 30 | 7.0 |
| SD3.5 Medium/Large | 1024×1024 | 40 | 4.5 |
| Qwen-Image 2.1 | 1024×1024 | 40 | 1.0 (no CFG; `>1` + negative prompt enables CFG) |
| Qwen-Image 2.1 (Viggle Turbo) | 1024×1024 | 6 | 1.0 fixed (no CFG, no negative prompt; raw sigma schedule) |

Requested sizes snap to the nearest configured aspect-ratio bucket. The shipped
presets carry each model's officially recommended ratios: SDXL/SD3.5 use 9
ratios at ~1MP; Qwen-Image 2.1 carries its 7 model-card ratios in both the
~1MP class (1024² … 1536×864) and the official 2K class (2048×2048 …
2752×1536). `buckets` is per-deployment configuration: trim it in
`server/model.json` if the host's VRAM cannot cover a class.
Lower-VRAM cards auto-fall back to CPU offload; CUDA-OOM at runtime also
auto-offloads and retries.


## Image editing (img2img)

`edit_image(prompt, image, ...)` transforms an existing image:

- **`image`**: an `http(s)://` URL. A server image URL
  (`http://<host>/images/<id>`) resolves from the server's in-memory cache;
  an external image URL is fetched server-side (SSRF-guarded: private /
  loopback refused). On a local stdio run a host file path or `file://` URI is
  also accepted; over http/sse the server never reads host files. To edit a
  local image against a remote server, upload it first (see
  § [Tools](#tools) / `upload_image`).
- **`strength`** (0..1, default 0.6): higher = larger change. Not accepted on
  `qwen-image-2.1` — its edit always runs the full step count.
- **`width`/`height`** (0 = keep source size; any value snaps to the nearest
  configured bucket — so the output aspect can differ from a non-bucket source).

img2img reuses the loaded pipeline weights (no second model copy). `--smoke`
also exercises the img2img path.

## Input size & VRAM notes

- Uploaded / fetched source images are bounded by `--max-body-mb`
  (default 16 MB); raise it only for very large source images.
- fp16/bf16 weights + attention slicing + proactive CPU offload + OOM
  auto-retry are built in for low-VRAM cards. VAE decode slicing/tiling is
  **opt-in** (`PICTURA_VAE_SLICING` / `PICTURA_VAE_TILING`) because the tiled
  path leaves faint periodic color bands in the output.
- `PICTURA_CUDA_DEVICE` (e.g. `0` or `0,1`) restricts which CUDA GPUs the
  server uses.
- `PICTURA_LOG_FILE` (or `--log-file`) appends the `[pictura-mcp]` log to a
  file (default stderr/journald).
  **Privacy: prompts and tool arguments are never written to any log.**

## LoRA & ControlNet

**LoRA** (both `generate_image` and `edit_image`): pass `lora` as
comma-separated `huggingface/repo:weight` entries (weight defaults to 1.0);
`list_loras` lists the allowed ids with descriptions.

```text
lora = "prithivMLmods/SD3.5-Large-Photorealistic-LoRA:0.8"
```

- Client-supplied ids are restricted to `supported_loras` in `model.json`;
  URLs, local paths and path traversal are rejected; weights load
  safetensors-only. SD3.5-family kohya-style LoRAs are converted server-side.

**ControlNet** (`edit_image` only, families with `control_types` configured):
pass `control_type` — the abstract type applied to the source image (see
`list_control_types`); `control_scale` (0.4–1.0) tunes the strength. The
preprocessor runs in-memory and the backing model stays server-side.

```text
control_type = "canny"   # auto-extract edges from the source, then ControlNet
control_scale = 0.9
```

- **Families without `control_types` hide the knobs entirely**: for such
  deployments `edit_image`'s schema omits `control_type` / `control_scale` /
  `strength` (Qwen-Image 2.1) and `list_control_types` is not registered.
  Adding a `control_types` entry re-exposes them.

## Development checks

`bash scripts/check.sh` — compile + import + docs-consistency checks (no GPU
or model download needed).

## Docs

- `REFERENCE.md` — server reference (tool parameters, CLI, env vars, security, size limits)
- `CHANGELOG.md` — release history
- `skills/pictura-mcp/SKILL.md` — Agent Skill for end-user agents
- `deploy/` — client config templates, env template, sysusers definition, systemd unit, install manifest
