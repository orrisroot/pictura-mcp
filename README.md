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

## Setup

```bash
# 1) Python venv + dependencies (torch CUDA build, ~3 GB)
#    V100 / Volta (compute capability 7.0) machines:
#      ./.venv/bin/pip install -r server/requirements-v100.txt
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt

# 2) Activate a model preset -> server/model.json
cp server/examples/model.sd35-large.json server/model.json
#    other presets: model.sdxl.json / model.sd35-medium.json /
#    model.qwen-image-2.1.json

# 3) Provision the models (weights are never downloaded by the server)
sudo scripts/fetch-models.sh

# 4) Smoke test (writes outputs/smoke_test.png)
./.venv/bin/python server/pictura_server.py --smoke

# 5) Connect from your MCP client (see below)
```

Disk layout — the repo ships templates; these files are created per machine
(gitignored): `.mcp.json` (from `deploy/mcp.json.example`),
`deploy/pictura-mcp.env` (secrets), `server/model.json` (copied preset),
`.venv/`, `outputs/`.

### CUDA driver version

`requirements.txt` pulls the latest torch from PyPI (CUDA 13 wheels, driver
≥ 580 — check with `nvidia-smi`). **V100 (Volta, compute capability 7.0)**
machines must use `server/requirements-v100.txt` instead: newer torch builds
dropped the sm_70 kernels and generation fails with
`CUDA error: no kernel image is available`.

## Client configuration

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

## Running the server standalone / remote

```bash
./.venv/bin/python server/pictura_server.py \
  --transport http \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key my-secret-key
```

- `--transport http` (endpoint `/mcp`) or `--transport sse` (endpoint `/sse`)
- `--api-key <key>` (or `PICTURA_API_KEY`) — **required for http/sse**; clients
  send it via the `PICTURA_API_KEY` header
- `--max-body-mb <MB>` (default 16) caps the HTTP request body: image uploads
  and external image fetches
- `PICTURA_PUBLIC_URL` is required when binding a non-loopback address (the
  externally visible base, including any reverse-proxy path prefix); on
  loopback the request `Host` is used. Cache knobs:
  `PICTURA_IMAGE_CACHE_TTL` / `_MAX` / `_MAX_MB`.

Remote client config (`deploy/mcp.remote.json.example`):

```json
{
  "mcpServers": {
    "pictura": {
      "url": "http://<SERVER_HOST_OR_IP>:8001/mcp",
      "headers": { "PICTURA_API_KEY": "<TOKEN>" },
      "requestTimeoutMs": 600000,
      "toolPrefix": ""
    }
  }
}
```

### systemd (long-running / remote)

Prerequisite: Setup steps 1–3 (`.venv`, `server/model.json`, weights). The
installer registers the unit but does not start it:

```bash
sudo deploy/install-systemd.sh /absolute/path/to/repo pictura-mcp 8001
```

- `deploy/pictura-mcp.env` is created from the template with an
  auto-randomized `PICTURA_API_KEY` and pre-seeded host/port/public URL —
  edit only what needs changing (`sudoedit deploy/pictura-mcp.env`).
- Start and verify:

```bash
sudo systemctl start pictura-mcp
journalctl -u pictura-mcp -f   # wait for "Model ready (...)"
```

Runs under the unprivileged `pictura-mcp` account with systemd hardening; the
models dir and log file are whitelisted in `ReadWritePaths`. If the service
crashes at startup, remove `MemoryDenyWriteExecute=true` from the unit and
`systemctl daemon-reload && restart`.

**Env template change (non-destructive)**: the installer records a fingerprint
of the env template. When the template changes in a repo update, your
`deploy/pictura-mcp.env` is untouched and a rendered copy with your values is
written to `deploy/pictura-mcp.env.new`; diff and merge, delete the file, then
run `deploy/install-systemd.sh --adopt-env`.

**User-scope service (quick alternative)**:

```bash
cp deploy/pictura-mcp.env.example deploy/pictura-mcp.env   # set PICTURA_API_KEY
chmod 600 deploy/pictura-mcp.env
mkdir -p ~/.config/systemd/user
sed 's#<PROJECT_ROOT>#/absolute/path/to/repo#' \
  deploy/pictura-mcp.service > ~/.config/systemd/user/pictura-mcp.service
# remove the User= / Group= lines for a user unit
systemctl --user daemon-reload && systemctl --user enable --now pictura-mcp
```

Log rotation: `deploy/logrotate.example` (copytruncate, or SIGHUP postrotate).

## Model configuration

All model settings live in one file: `server/model.json` (gitignored; copy a
preset from `server/examples/` or point `PICTURA_MODEL_CONFIG` at it):

- `model` — base model: local path or `org/repo` id; family is auto-detected
  (`sdxl`, `sd35-medium`, `sd35-large`, `qwen-image-2.1`)
- `vae` — optional custom VAE id (`null` = auto)
- `families.<id>` — per-family settings: `desc`, `steps`, `guidance`,
  `width`/`height`, `buckets` (native resolutions; rotations are added
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
sudo scripts/fetch-models.sh   # base model + VAE + LoRA + ControlNet + preprocessors
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
# Qwen-Image 2.1 uncensored LoRA (single file inside the GGUF repo):
mkdir -p models/abenzerps/qwen-image-2.1-uncensored-lora
curl -fsSL https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/qwen-image-2.1-uncensored-lora.safetensors \
  -o models/abenzerps/qwen-image-2.1-uncensored-lora/qwen-image-2.1-uncensored-lora.safetensors
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

Requested sizes snap to the nearest native training bucket of the active
model (SDXL ~1MP; SD3.5 up to ~2MP; Qwen-Image 2.1 up to ~2.1MP) — off-bucket
sizes (e.g. 512×512) produce tiled/duplicated patterns. Lower-VRAM cards
auto-fall back to CPU offload; CUDA-OOM at runtime also auto-offloads and
retries.

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
  native bucket — so the output aspect can differ from a non-bucket source).

img2img reuses the loaded pipeline weights (no second model copy). `--smoke`
also exercises the img2img path.

## Input size & VRAM notes

- Uploaded / fetched source images are bounded by `--max-body-mb`
  (default 16 MB); raise it only for very large source images.
- fp16/bf16 weights + attention/VAE slicing + proactive CPU offload + OOM
  auto-retry are built in for low-VRAM cards.
- `PICTURA_CUDA_DEVICE` (e.g. `0` or `0,1`) restricts which CUDA GPUs the
  server uses.
- `PICTURA_LOG_FILE` (or `--log-file`) appends the `[pictura-mcp]` log to a
  file (default stderr/journald; reopened on SIGHUP for logrotate).
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

- `SPEC.md` — full technical specification
- `skills/pictura-mcp/SKILL.md` — Agent Skill for end-user agents
- `deploy/` — configuration templates + systemd unit
