# Pictura MCP

A **local, GPU-backed image generation MCP server** (codename `pictura-mcp`) exposed through the standard Model Context Protocol. It works with **any
MCP client** — Claude Desktop, Cursor, VS Code, Windsurf, Claude Code, etc.

```
MCP client ──────────────────▶ pictura_server.py ──────────────▶ GPU
            (stdio|HTTP|SSE)                      (SDXL/SD3.5/Qwen-Image 2.1)
```

- **Server**: `server/pictura_server.py` — a standard MCP server (stdio / HTTP /
  SSE) running Stable Diffusion (SDXL / SD3.5) or Qwen-Image 2.1 via Hugging
  Face `diffusers`.
- **Client**: whatever MCP client you already use. Examples for several clients
  are in § [Client configuration](#client-configuration).

## Features / tools

| Tool | Description |
|---|---|
| `generate_image` | text → image (model-family defaults) |
| `edit_image` | image → image (img2img / edit from a prompt) |
| `upload_image` | local file → upload reservation (one-time token + POST URL; http/sse; stdio: pass the path to `edit_image`) |
| `list_loras` | supported LoRA ids with descriptions (for `lora`) |
| `list_control_types` | abstract ControlNet types (for `control_type`); **not registered for families without `control_types`** in the config |
| `server_status` | model + image-size policy (stdio adds device / VRAM etc.) |

Both generation tools accept optional **LoRA** adapters, and `edit_image`
additionally supports **ControlNet** via an abstract `control_type` (see
[LoRA & ControlNet](#lora--controlnet)). The `upload_image` reservation flow
applies over http/sse; on a local stdio run the tool instead tells you to pass
the file path to `edit_image` directly. Parameter meaning and value formats are
embedded in each tool's input schema (visible to MCP clients); `list_loras` /
`list_control_types` return the valid values, and ControlNet model identifiers
stay server-side.

Stateless by design: images are **never written to the server disk**. How they
are returned depends on the transport:

- **stdio** (local): inline base64 `ImageContent` + a `TextContent` note. The
  assistant must decode the returned base64 (data field) into a local file
  (e.g. save as PNG) and open it to inspect the result.
- **http/sse** (remote): the result note contains a **short-lived download
  URL** (e.g. `http://<host>:8000/images/<unguessable-id>`); the client fetches
  it with HTTP tools and saves it — no shared filesystem needed. Each URL is
  valid for a short TTL (default 600 s) from a small in-memory cache; nothing
  is persisted. Result download URLs are **short-lived capability links** that
  the client fetches and saves. To upload a local file, call the `upload_image`
  tool and POST the bytes to the returned URL with the `X-UPLOAD-TOKEN` header.

The tool descriptions and the result note explain this on every call.

**Concurrent requests** are accepted: the server handles multiple MCP clients
and parallel tool calls without ever blocking its event loop. Rendering runs
on a pool of **slots** — each slot is an independent pipeline instance (own
LoRA/adapter state, scheduler and offload hooks), so jobs never corrupt each
other. The pool size is sized automatically from measured free VRAM once the
model is loaded (per extra slot costs a full set of weights plus one job's
activation footprint, with a safety margin) — e.g. 3 slots on a 32 GB V100
with SDXL fp16 / SD3.5, and 1 (strictly serial) on smaller cards. Override it with
`PICTURA_MAX_CONCURRENT` (an integer pins the pool; `auto` = VRAM-based). Note
that extra slots are built lazily: the first burst of parallel jobs may pay
one model-load per extra slot.

## Setup

```bash
# 1) Python venv + dependencies (torch is the CUDA build, ~3GB)
#    V100 (Volta, compute capability 7.0) machines must use the V100 file:
#    ./.venv/bin/pip install -r server/requirements-v100.txt
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt

# 2) Activate a model preset (deployment-local config; see "Model configuration")
cp server/examples/model.sd35-large.json server/model.json
#    use example/model.sdxl.json, model.sd35-medium.json, or model.qwen-image-2.1.json for another family

# 3) Provision the models (deployment step; the service never downloads)
sudo scripts/fetch-models.sh

# 4) Smoke test
./.venv/bin/python server/pictura_server.py --smoke
# -> OK if outputs/smoke_test.png is created

# 5) Connect from your MCP client (see below)
```

### CUDA driver version

The default install above pulls the latest torch from PyPI, whose wheels are
built for **CUDA 13** (`cu130`) — this needs a CUDA-13-capable driver
(**≥ 580**). Check yours with `nvidia-smi` (top-right corner).

**Volta (V100, compute capability 7.0):** requires torch ≤2.7.1 from the cu126
index — later torch builds do not include the sm_70 kernels, so generation
fails with `CUDA error: no kernel image is available`. Use
`server/requirements-v100.txt` (do **not** install `requirements.txt`, whose
`torch>=2.9` would upgrade torch again):

```bash
./.venv/bin/pip install -r server/requirements-v100.txt
```

> **Repo layout note (gitignored files).** The committed repository ships
> **templates** (`deploy/mcp.json.example`, `deploy/pictura-mcp.env.example`,
> `deploy/pictura-mcp.service`). Local files that are **gitignored** and created
> per machine: your real client config (e.g. `.mcp.json` — copy from
> `deploy/mcp.json.example`, replace `<PROJECT_ROOT>`), `deploy/pictura-mcp.env`
> (secrets — never commit), the active model config `server/model.json` (copy a
> preset from `server/examples/`), plus `.venv/` and `outputs/`.

## Client configuration

Every MCP client stores server definitions in the same shape
(`mcpServers` → name → `command`/`args`/`env`). Point it at the server:

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

`requestTimeoutMs` (client-side, supported by pi's MCP adapter and most
harnesses) must allow for GPU rendering time: a generation at the family
default (1024², 30-40 steps) takes tens of seconds to minutes, and under
parallel load a job may additionally wait for a free slot or a lazily built
one. The MCP SDK default (60 s) therefore times out on ordinary generations;
the shipped templates use `600000` (10 min).

Where that block goes depends on the client:

| Client | Config location | Notes |
|---|---|---|
| Claude Code, Cursor, Windsurf, VS Code, generic | project **`.mcp.json`** (`deploy/mcp.json.example`) | industry-standard project config; many clients auto-detect it |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` | add the same `mcpServers` block under `"mcpServers"` |
| VS Code | `.vscode/mcp.json` | |

Any other harness (e.g. a local agent framework) uses the identical
`mcpServers` block in its own client config — the server does not care.

Generic quick start (any client that reads `.mcp.json`):

```bash
cp deploy/mcp.json.example .mcp.json
# edit: replace <PROJECT_ROOT> with the absolute project path
# restart your client, then ask it to "generate an image"
```

## Running the server standalone / remote

You can also run the server as an independent process that clients reach over
**HTTP (streamable HTTP)** or **SSE**:

```bash
./.venv/bin/python server/pictura_server.py \
  --transport http \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key my-secret-key
```

- `--transport http` (endpoint `/mcp`) or `--transport sse` (endpoint `/sse`);
  `--host 127.0.0.1` is the safe default — use `0.0.0.0` for remote clients
- `--api-key <key>` (or env `PICTURA_API_KEY`): **required for http/sse** — the
  server refuses to start without it (remote users authenticate via the
  **`PICTURA_API_KEY`** header)
- **Uploading source images**: call the `upload_image` tool over http/sse — it
  returns a **one-time token** (TTL `PICTURA_IMAGE_UPLOAD_TICKET_TTL`, default 120 s)
  and a ready-to-run `curl` line; then POST the raw image bytes or a
  multipart/form-data `file` field to the returned `/images/upload` URL with
  the `X-UPLOAD-TOKEN` header (body capped by `--max-body-mb`). This returns a
  short-lived `http://<base>/images/<id>` URL that `edit_image` accepts and
  `GET /images/<id>` serves back. Direct `PICTURA_API_KEY` uploads also work.
  Examples:
  `curl -X POST -H 'X-UPLOAD-TOKEN: <token>' -H 'Content-Type: image/jpeg' --data-binary @photo.jpg http://<host>:8000/images/upload`
  or `curl -F 'file=@photo.jpg' -H 'PICTURA_API_KEY: <key>' http://<host>:8000/images/upload`
- **`edit_image` takes an `http(s)://` URL as the source**: a server image URL
  (from `generate_image` / `edit_image` / the upload endpoint) resolves from
  the in-memory cache, and any external image URL is fetched server-side with
  an **SSRF guard** (private/loopback addresses are refused). Host files are
  never read over http/sse — only a **local stdio run** may pass a file path
  (see § Image editing).
- `--max-body-mb <MB>` (default 16) caps the HTTP request body: image uploads
  and external image fetches
- The server is **stateless**: no image files are written on the server in any
  mode. Over http/sse each generated image is returned as a **short-lived
  download URL** (served from an in-memory TTL cache at `GET /images/<id>`);
  uploaded images live in the same cache. Over stdio results are returned
  inline as base64. Clients save the image wherever they like.
- **`PICTURA_PUBLIC_URL` is required when binding a non-loopback address**
  (the server refuses to start without it): set it to the externally visible
  base, including any path prefix when a reverse proxy serves the server
  (e.g. `https://www.ni.riken.jp/llm/pictura`). On loopback it falls back to
  the request `Host`. Troubleshooting knobs:
  `PICTURA_IMAGE_CACHE_TTL` (600 s), `PICTURA_IMAGE_CACHE_MAX` (64),
  `PICTURA_IMAGE_CACHE_MAX_MB` (512).

Remote client config (`deploy/mcp.remote.json.example`):

```json
{
  "mcpServers": {
    "pictura": {
      "url": "http://<SERVER_HOST_OR_IP>:8000/mcp",
      "headers": { "PICTURA_API_KEY": "<TOKEN>" },
      "requestTimeoutMs": 600000,
      "toolPrefix": ""
    }
  }
}
```

### systemd (recommended for long-running / remote)

Prerequisite: §Setup step 1 above — the `.venv`+dependencies must exist at
`<PROJECT_ROOT>/.venv` (the service runs that interpreter). The manual smoke
test is optional here. Before the first start, activate a model preset and
provision the weights (the service never downloads):

```bash
cp server/examples/model.sd35-large.json server/model.json
sudo scripts/fetch-models.sh
```

There are two ways to run it as a service.

**A) Dedicated service account (system scope, recommended):**

```bash
# As root - creates the service user, prepares dirs, renders & installs the unit
# (boot-autostart is registered but the service is NOT started)
sudo deploy/install-systemd.sh /absolute/path/to/this/repo pictura-mcp 8000
```

The installer prints the next steps; the essentials are already prepared:

- `deploy/pictura-mcp.env` is created from the template with an
  **auto-randomized `PICTURA_API_KEY`**, and `PICTURA_HOST=0.0.0.0` /
  `PICTURA_PORT` / `PICTURA_PUBLIC_URL` (this host's IP:port — edit it to the
  public URL behind a reverse proxy) are **pre-seeded** — edit only what needs
  changing (`sudoedit deploy/pictura-mcp.env`; e.g. `PICTURA_MODEL_CONFIG`,
  `PICTURA_CUDA_DEVICE`, `PICTURA_LOG_FILE`). Activate a preset and provision
  the models first:
  `cp server/examples/model.sd35-large.json server/model.json && sudo scripts/fetch-models.sh`.
- start and verify:

```bash
sudo systemctl start pictura-mcp
journalctl -u pictura-mcp -f
# wait for:  MCP http server: http://0.0.0.0:8000/mcp ... Model ready (...)

# client config: copy deploy/mcp.remote.json.example and set
#   url: http://<this-box-ip>:8000/mcp   (port = PICTURA_PORT from the env file)
#   PICTURA_API_KEY: <PICTURA_API_KEY from deploy/pictura-mcp.env>
```

```bash
sudo systemctl status pictura-mcp
```

This runs under the unprivileged `pictura-mcp` system account with hardening
(`NoNewPrivileges`, `ProtectSystem`, `PrivateTmp`, …). `ProtectSystem=strict`
makes the FS read-only, so the installer whitelists the models dir and log
file in `ReadWritePaths` (derived from `deploy/pictura-mcp.env`; log file is
0640, owner = service account, group = service account). The installer also
activates commented-out defaults in the env file (`PICTURA_HOST`,
`PICTURA_PORT`); weights are provisioned ahead of time by
`scripts/fetch-models.sh` — the server never downloads. If you later change
`PICTURA_LOG_FILE`, re-run the installer (it re-renders the unit; then
`systemctl restart pictura-mcp`). Non-root operators
read the log by joining the group once: `sudo usermod -aG pictura-mcp <username>`
(then log out/in). If the service crashes at startup, remove
`MemoryDenyWriteExecute=true` from the unit (torch sometimes conflicts) and
`systemctl daemon-reload && restart`.

**Env template change (non-destructive).** The installer records a sha256
fingerprint of `deploy/pictura-mcp.env.example` in the env file. When the
template changes, your `deploy/pictura-mcp.env` is left untouched and the
current rendered template is written to **`deploy/pictura-mcp.env.new`**
(machine values such as cache / host / port / public URL pre-filled, and the
`PICTURA_API_KEY` carried over from your env so merging it keeps clients
working). Diff & merge
what you want, delete the file, then run **`deploy/install-systemd.sh
--adopt-env`** (a standalone step: records the template and removes the `.new`
file, no reinstall). The API key is generated on first install and kept
afterwards.

**B) Current user (user scope, quick):**

```bash
# activate a preset + provision weights (the service never downloads)
cp server/examples/model.sd35-large.json server/model.json
sudo scripts/fetch-models.sh
cp deploy/pictura-mcp.env.example deploy/pictura-mcp.env   # set PICTURA_API_KEY
chmod 600 deploy/pictura-mcp.env
mkdir -p ~/.config/systemd/user
sed 's#<PROJECT_ROOT>#/absolute/path/to/this/repo#' \
  deploy/pictura-mcp.service > ~/.config/systemd/user/pictura-mcp.service
# remove the User= / Group= lines for a user unit
systemctl --user daemon-reload && systemctl --user enable --now pictura-mcp
systemctl --user status pictura-mcp
```

Log rotation: `deploy/logrotate.example` (copytruncate, or SIGHUP postrotate).

> Verified end-to-end: `tools/list` returns all six tools (`generate_image`,
> `edit_image`, `list_loras`, `list_control_types`, `server_status`,
> `upload_image`); missing/wrong tokens get 401; clients connect over stdio
> and over HTTP.

## Model configuration

All model settings live in **one file**: `server/model.json` — the base model
(`model`; local path, relative to the project root, or `org/repo`), an
optional custom `vae`, **per-family settings** (`families`, keyed by the
internal id `sdxl` / `sd35-medium` / `sd35-large` / `qwen-image-2.1`:
`desc`, `steps`,
`guidance`, `width`/`height`, `buckets`, `auto_vae`), the **supported LoRA
ids** (`supported_loras`: id → description map) and the **ControlNet types**
(`control_types`: `pre`/`model`/`prep_model` per type). No other model env
vars are read. `server/model.json` is **deployment-local and gitignored** like
`deploy/pictura-mcp.env`: copy a preset from `server/examples/`
(`model.sdxl.json` / `model.sd35-medium.json` / `model.sd35-large.json` /
`model.qwen-image-2.1.json`) to
`server/model.json`, or point `PICTURA_MODEL_CONFIG` at a preset directly. The
server refuses to start when the configured model is not a supported family
(exit 2).

**Model provisioning (deployment step):** the service never contacts Hugging
Face - every weight is a **plain local directory** under `PICTURA_MODELS_DIR`
(default `<project>/models`), prepared before start:

    sudo scripts/fetch-models.sh            # base model (repo id) + VAE + LoRA + ControlNet + preprocessors

Manually, one by one (`hf` ships in the project venv):

    # base model as a local dir (config "model" points at it)
    ./.venv/bin/hf download stabilityai/stable-diffusion-3.5-large --local-dir models/sd35-large
    # repo id base / custom VAE -> models/<org>/<repo>
    ./.venv/bin/hf download SG161222/RealVisXL_V5.0 --local-dir models/SG161222/RealVisXL_V5.0
    # Qwen-Image 2.1 (7B transformer + 8B text encoder + VAE, ~31 GB bf16)
    ./.venv/bin/hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen/Qwen-Image-2.1
    # LoRA / ControlNet / preprocessors -> the same plain-dir layout
    ./.venv/bin/hf download prithivMLmods/SD3.5-Large-Photorealistic-LoRA --local-dir models/prithivMLmods/SD3.5-Large-Photorealistic-LoRA
    ./.venv/bin/hf download diffusers-internal-dev/sd35-controlnet-depth-8b --local-dir models/diffusers-internal-dev/sd35-controlnet-depth-8b
    ./.venv/bin/hf download Intel/dpt-hybrid-midas --local-dir models/Intel/dpt-hybrid-midas
    # openpose preprocessor (not an HF repo):
    curl -fsSL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt -o models/yolov8n-pose.pt
    # Qwen-Image 2.1 uncensored LoRA (a single file in the GGUF repo, not a
    # standalone HF repo); place it manually where the config id points:
    mkdir -p models/abenzerps/qwen-image-2.1-uncensored-lora
    curl -fsSL https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/qwen-image-2.1-uncensored-lora.safetensors \
      -o models/abenzerps/qwen-image-2.1-uncensored-lora/qwen-image-2.1-uncensored-lora.safetensors

Layout:

    models/<name>/          base model directories (config "model" points here)
    models/<org>/<repo>/    LoRA / ControlNet / preprocessor / repo-id base models (plain dirs)
    models/yolov8n-pose.pt  openpose preprocessor

Defaults per model family (from `families.*`): **SDXL** 1024×1024 / 30 steps /
guidance 7.0, **SD3.5 (Medium & Large)** 1024×1024 / 40 steps / guidance 4.5,
**Qwen-Image 2.1** 1024×1024 / 40 steps / guidance 1.0 (no CFG by default;
pass `guidance_scale` > 1 with a `negative_prompt` to enable classifier-free
guidance)
(0 = family default in the tools). Requested sizes are snapped to the nearest
native training bucket of the active model (SDXL ~1MP; SD3.5 up to ~2MP;
Qwen-Image 2.1 up to ~2.1MP; sizes must stay multiples of 32) —
off-bucket sizes (e.g. 512×512) produce tiled/duplicated patterns. On
lower-VRAM cards weights auto-fall back to CPU
offload; CUDA-OOM at runtime also auto-offloads and retries.

## Image editing (img2img)

`edit_image(prompt, image, ...)` transforms an existing image:

- **`image`**: an `http(s)://` URL. A **server image URL**
  (`http://<host>/images/<id>`) — from `generate_image`, a previous
  `edit_image`, or `POST /images/upload` — resolves from the server's
  in-memory cache. Any **external image URL** is fetched server-side
  (SSRF-guarded: private/loopback addresses are refused). On a **local stdio
  run** you may also pass a **host file path** or `file://` URI; over
  http/sse the server never reads host files.
  To edit a local image against a remote server, call the `upload_image`
  tool first and POST the bytes with the returned `X-UPLOAD-TOKEN` (see
  **Uploading source images** above) to get a server image URL.
- **`strength`** (0..1, default 0.6): higher = larger change
- **`width`/`height`** (0 = keep source size; any value is snapped to the
  nearest native training bucket of the active model — including 0, so the
  output aspect can differ slightly from a non-bucket source)

Reuses loaded weights via `AutoPipelineForImage2Image.from_pipe` (no second
model copy). `--smoke` also exercises the img2img path.

## Input size & VRAM notes

- Uploaded and fetched source images are bounded by `--max-body-mb`
  (default **16 MB**), ample headroom over the ~1.6 MB outputs and typical
  camera JPEGs. Raise it only if you need to pass very large source images.
- fp16 + attention/VAE slicing + proactive CPU offload + OOM auto-retry are all
  baked in for low-VRAM cards.
- GPU selection: set `PICTURA_CUDA_DEVICE` (e.g. `0` or `0,1`) to restrict which
  CUDA GPU(s) the server uses (`CUDA_VISIBLE_DEVICES`).
- Log destination: set `PICTURA_LOG_FILE` (or `--log-file <path>`) to append the
  `[pictura-mcp]` log to a file (default: stderr/journald).
- edit_image host-path reads: allowed on a local stdio run, denied over
  http/sse (see § Image editing).
  Logrotate-ready: the server reopens its log file on `SIGHUP`, and a
  `copytruncate`-based config is provided in `deploy/logrotate.example`.
  **Privacy: user prompts and tool arguments are never written to any log.**

## LoRA & ControlNet

**LoRA** (both `generate_image` and `edit_image`): pass `lora` as comma-separated
`huggingface/repo:weight` entries (weight defaults to 1.0). `list_loras`
returns the allowed ids **with what each does** (style / trigger word /
target family). Example:

```text
lora = "prithivMLmods/SD3.5-Large-Photorealistic-LoRA:0.8"
```

**ControlNet** (`edit_image` only, families with `control_types` configured):
pass `control_type` — an abstract type
applied to the source image (see `list_control_types`). The server runs the
preprocessor in-memory and picks/hides the backing model; `control_scale`
(0.4–1.0) tunes the strength. Example:

```text
control_type = "canny"   # auto-extract edges from the source, then ControlNet
control_scale = 0.9
```

**Families without ControlNet hide the knobs entirely.** When the active
`model.json` has no `control_types` entries (e.g. the `qwen-image-2.1` preset,
whose pipeline does image-conditioned editing without ControlNet),
`edit_image` is registered **without** the `control_type`/`control_scale`
parameters and the `list_control_types` tool is **not registered at all** —
as far as an MCP client can tell, those knobs do not exist for that
deployment. Adding a `control_types` entry to the config re-exposes them
(unchanged tool names, so clients just see the parameters reappear).

Requires the `peft` dependency (listed in `server/requirements.txt`).

**Supported LoRAs & provisioning**
- Client-supplied `lora` ids are restricted to `supported_loras` in
  `model.json` (id → description map; `*` = any bare `org/repo` id). URLs,
  local paths and path traversal are always rejected, and weights load
  safetensors-only. SD3.5-family kohya-style LoRAs are converted server-side;
  Qwen-Image 2.1 LoRAs load through `QwenImageLoraLoaderMixin`.
- **ControlNet ids are server-side and hidden** — clients only choose an
  abstract `control_type`; backing model and preprocessor are configured in
  `control_types`. ControlNet is **not supported for the `qwen-image-2.1`
  family** (the unified pipeline does image-conditioned editing without it):
  there its parameters and the `list_control_types` tool are hidden entirely
  (see above).
- All weights are **plain local dirs** provisioned by `scripts/fetch-models.sh`
  — no downloads at startup or at tool-call time; a missing model raises a
  provisioning error.

## Development checks

`bash scripts/check.sh` compiles and imports the server and verifies that docs
stay in sync (tool / env-var names, no stale wording, env example safe for
systemd). No GPU or model download needed.

## Docs

- `SPEC.md` — full technical specification
- `skills/` — distributable **Agent Skill** (`pictura-mcp`) for end-user agents (see `skills/README.md`)
- `deploy/` — configuration templates + systemd unit
- `scripts/check.sh` — lightweight dev checks (no GPU needed)
