# Image generation MCP server — Specification

Status: **final** (all behavior is implemented, tested on a consumer GPU)

This document is the authoritative specification of `pictura-mcp`. It
covers the image-gen backend (`server/pictura_server.py`) and how to connect any
MCP client (stdio, streamable HTTP, SSE).

---

## 1. Overview

Local, GPU-backed image generation wrapped as an MCP server:

```
MCP client ──────────────────▶ pictura_server.py ──────────────▶ GPU
            (stdio|HTTP|SSE)                      (SDXL/SD3.5/Qwen-Image 2.1)
```

It is client-agnostic — any harness that speaks MCP (Claude Desktop, Cursor,
VS Code, Windsurf, local agent frameworks, …) can connect.

- Text-to-image (`generate_image`)
- Image-to-image / editing (`edit_image`)
- Upload reservation (`upload_image`)
- Status introspection (`server_status`)

**Key policies**
- **Stateless**: the server never writes image files to disk in any mode. How
  results are returned depends on the transport: **stdio** returns the image
  inline as base64 `ImageContent` (text clients decode the data field into a
  file and open it to inspect); **http/sse** returns a **short-lived download
  URL** in the result note (image sits in a small in-memory TTL cache served at
  `GET /images/<id>`, so the client needs no shared filesystem). Everything is
  gone when the process exits.
- **Identical local & remote behavior**: the only mode-dependent differences
  are deliberate: image delivery (inline vs URL) and `edit_image` host-path
  reads (stdio only, see §2 / §5). Output is never written to disk in any mode.
- **Concurrency**: requests are accepted concurrently (the event loop never
  blocks on GPU work). Rendering runs on a slot pool; the pool size is derived
  from measured free VRAM (per extra slot: another full weight set + activation
  footprint + fixed reserve; capped; `PICTURA_MAX_CONCURRENT` overrides). Each
  slot is an independent pipeline instance so LoRA/offload/scheduler state
  never races between concurrent jobs.
- **Tools never accept a path**: the save location is not controllable through
  the MCP interface.

---

## 2. MCP tools

### `generate_image`
Text-to-image.

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required |
| `negative_prompt` | string | `""` | only used when CFG is active (Qwen-Image 2.1: `guidance_scale > 1`) |
| `width` | int | 0 = family default | any positive value is snapped to the nearest native training bucket of the active model (SDXL ~1MP / SD3.5 up to ~2MP / Qwen-Image 2.1 up to ~2.1MP, multiple of 32; min 256, multiple of 8) |
| `height` | int | 0 = family default | any positive value is snapped to the nearest native training bucket |
| `num_inference_steps` | int | 0 = family default (SDXL 30 / SD3.5 40 / Qwen 40) | clamped to [10, 100] |
| `guidance_scale` | float | 0 = family default (SDXL 7.0 / SD3.5 4.5 / Qwen 1.0) | mapped to `true_cfg_scale` for Qwen-Image 2.1 |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |

### `edit_image`
Edits an existing image with a prompt (img2img).

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required |
| `image` | string | — | required. Source image: an `http(s)://` URL — a server image URL (`http://<host>/images/<id>` from `generate_image` / `edit_image` / `POST /images/upload`, resolved from the in-memory cache) or an external image URL (fetched server-side, SSRF-guarded). On a local stdio run a host file path / `file://` URI is also accepted; over http/sse the server reads no host files |
| `negative_prompt` | string | `""` | only used when CFG is active (Qwen-Image 2.1: `guidance_scale > 1`) |
| `strength` | float | 0.6 | 0..1, higher = more change (not used by Qwen-Image 2.1, which edits on the unified pipeline) |
| `width` / `height` | int | 0 | 0 = keep the source size (also snapped); any positive value is snapped to the nearest native training bucket of the active model (min 256, multiple of 8; Qwen-Image 2.1 keeps the source aspect and derives size from `output_resolution`) |
| `num_inference_steps` | int | 0 = family default (SDXL 30 / SD3.5 40 / Qwen 40) | effective steps ≈ `steps × strength` |
| `guidance_scale` | float | 0 = family default (SDXL 7.0 / SD3.5 4.5 / Qwen 1.0) | mapped to `true_cfg_scale` for Qwen-Image 2.1 |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |
| `control_type` | string | `""` | optional abstract ControlNet type applied to the source (`canny`, `depth`, `openpose`); server hides the model; empty = disabled. **Not supported for `qwen-image-2.1`** |
| `control_scale` | float | 1.0 | ControlNet conditioning strength (~0.4–1.0) |

### `upload_image`
| Param | Type | Default | Description |
|---|---|---|---|
| `filename` | string | `""` | informational hint for the saved file name |

Reserves an upload over http/sse: returns a **one-time token** (TTL
`PICTURA_IMAGE_UPLOAD_TICKET_TTL`, default 120 s, single use) plus the
`POST /images/upload` URL and a ready-to-run `curl` line. The client then POSTs
the image bytes (raw or multipart `file`) with the `X-UPLOAD-TOKEN` header —
no long-lived API key is used on the upload request. The token is claimed at
POST time (a concurrent POST with the same token is rejected), consumed only
after a successful upload, and released for retry when the upload fails. The
`PICTURA_API_KEY` header is also accepted (direct/LAN clients).

### `list_loras`
Returns the supported LoRA ids valid for the `lora` parameter.

### `list_control_types`
Returns the abstract ControlNet types valid for `control_type`, with short
guidance. **No model identifiers are exposed** (they stay server-side).

### `server_status`
Reports `model` and the image-size policy (`size_policy`, `snap_buckets`,
`native_size`). On a local stdio run it additionally reports the internal
`device`, `dtype`, `offload`, `weights_gb`, `vram_gb`, `concurrency_slots` and
`load_seconds`; over **http/sse** those are not exposed.

**All tools return**: over **stdio** an `ImageContent` (base64 PNG, mime
`image/png`) + a `TextContent` note; over **http/sse** a single `TextContent`
note containing a short-lived download URL (`http://<base>/images/<id>`, TTL
`PICTURA_IMAGE_CACHE_TTL`, default 600 s). On failure a text error is returned.

---

## 3. Models & pipeline

### Model configuration (single file)
All model settings live in **one deployment-local file**, `server/model.json`
(gitignored; presets in `server/examples/*.json` — copy one to
`server/model.json` or point `PICTURA_MODEL_CONFIG` at a preset).
`PICTURA_MODEL_CONFIG` overrides the path):
- `model` — base model: local path (absolute or project-root-relative) or
  `org/repo` id; family is auto-detected (`sdxl`, `sd35-medium`, `sd35-large`,
  `qwen-image-2.1`)
- `vae` — optional custom VAE id (`null` = auto)
- `families.<id>` — per-family settings: `desc`, `steps`, `guidance`,
  `width` / `height`, `auto_vae`, `buckets` (native resolutions; rotations are
  added automatically)
- `supported_loras` — id → description map (client-usable LoRAs)
- `control_types` — `pre` / `model` / `prep_model` per abstract type
  (`canny` / `depth` / `openpose`; ids hidden from clients)

Weights are **plain local directories** under `PICTURA_MODELS_DIR` (default
`<project>/models`; `models/<org>/<repo>` for repo ids), provisioned by
`scripts/fetch-models.sh` before start. The server never downloads: a missing
model raises a provisioning error. SDXL families use the configured
`auto_vae` (fp16-safe VAE) automatically.

### Model-aware defaults
| Family | Default W×H | Default steps | Guidance |
|---|---|---|---|
| `sdxl` | 1024×1024 | 30 | 7.0 |
| `sd35-medium` / `sd35-large` | 1024×1024 | 40 | 4.5 |
| `qwen-image-2.1` | 1024×1024 | 40 | 1.0 (no CFG; `>1` + negative prompt enables CFG) |

Requested sizes are snapped to the family's native training buckets (`buckets`
in the config), so output always stays on the aspect/area combinations the
model was trained on.

### Memory strategy (low VRAM)
1. fp16 weights loaded (bf16 for Qwen-Image 2.1); attention slicing + VAE
   slicing + VAE tiling enabled.
2. **Proactive CPU offload** when the model is SDXL with default res ≥ 768, or
   whenever weights exceed ~80% of VRAM. Qwen-Image 2.1 (~31 GB bf16) exceeds
   80% of a 24–32 GB card, so it always runs with model CPU offload
   (`text_encoder->transformer->vae`). Otherwise weights stay fully on GPU.
3. Runtime CUDA-OOM → auto `enable_model_cpu_offload()` + one retry.
4. img2img reuses the loaded components via `AutoPipelineForImage2Image.from_pipe`
   (shared weights, no second model copy). Qwen-Image 2.1 has a single unified
   pipeline: image conditioning / editing runs on the same `QwenImage21Pipeline`
   via its `image` argument (no separate img2img class).

### Model family & configuration (single JSON file)

The active model (base model / optional VAE / supported LoRAs / ControlNet
types) is defined in one file, `server/model.json` (`PICTURA_MODEL_CONFIG`
points at an alternative; presets in `server/examples/`). Supported families:
**SDXL** (`sdxl`), **SD3.5 Medium / Large** (`sd35-medium` / `sd35-large`),
**Qwen-Image 2.1** (`qwen-image-2.1`).
The family is detected from the model directory's `model_index.json` (and the
transformer config for SD3.5 variants); repo ids fall back to the id string.
Unsupported families are rejected at startup (exit 2).

LoRA ids are restricted to `supported_loras` in the config (`["*"]` = any id);
ControlNet is exposed as abstract `control_type`s (`canny` / `depth` /
`openpose`, in-memory preprocessed server-side); the backing model and
preprocessor weights are configured server-side in `control_types` and are
**not exposed to clients**.

### Downloads (manual, before first start)
- URLs and local paths are always rejected for client-supplied ids; weights
  load safetensors-only.
- Pre-download everything before starting the service with
  `sudo scripts/fetch-models.sh` (reads `server/model.json` — copy a preset
  from `server/examples/` first if it is missing; everything is
  written as **plain local dirs**: repo ids → `models/<org>/<repo>/`).
  See the README for one-by-one `hf download --local-dir` examples.
- The server never contacts Hugging Face (no startup prefetch, no lazy
  downloads); a missing model surfaces as a clear provisioning error pointing
  at `scripts/fetch-models.sh`.

---

## 4. Transports & CLI

```
python server/pictura_server.py [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--transport stdio\|http\|sse` | `stdio` | MCP transport |
| `--host` | `127.0.0.1` | bind address (use `0.0.0.0` for remote) |
| `--port` | `8000` | TCP port |
| `--api-key <key>` | — | **required for http/sse**; remote clients send it via the `PICTURA_API_KEY` header |
| `--max-body-mb <n>` | 16 | max HTTP request body (http/sse); bounds image uploads and external image fetches |
| `--log-file <path>` | stderr | append `[pictura-mcp]` logs to a file (also `$PICTURA_LOG_FILE`) |
| `--smoke` | — | self-test (txt2img + img2img) writing to `<repo>/outputs/` |

- http → endpoint `/mcp` (streamable HTTP, JSON responses)
- sse → endpoint `/sse`
- **Image upload**: `upload_image` issues a one-time token (TTL
  `PICTURA_IMAGE_UPLOAD_TICKET_TTL`, default 120 s); POST raw image bytes or a
  multipart/form-data `file` field to `/images/upload` with the `X-UPLOAD-TOKEN`
  header (body capped by `--max-body-mb`; the `PICTURA_API_KEY` header is also
  accepted) and it returns a short-lived `http://<base>/images/<id>` URL; the
  image is stored in the same in-memory TTL cache and served back by
  `GET /images/<id>`.
- **Image downloads**: `GET /images/<id>` serves cached generated / uploaded
  images (see §1 / §6). `PICTURA_PUBLIC_URL` is required when binding a
  non-loopback address (the server refuses to start without it) so the URLs
  returned to clients are reachable; on loopback the request `Host` is used.
  Result download URLs are short-lived capability links that the client
  fetches and saves.
- **Image return mode**: stdio → inline base64; http/sse → short-lived URL
  (URL only; no inline bytes), unless no public base is resolvable (falls back
  to inline).

---

## 5. Security

- **Remote transports require an API key**: http/sse refuses to start without
  one (`--api-key` / `PICTURA_API_KEY`); the GPU is otherwise reachable by any
  caller.
- **Non-loopback http/sse requires `PICTURA_PUBLIC_URL`** (refuses to start):
  the externally visible base (scheme + host + any path prefix) so returned
  image URLs are reachable behind reverse proxies. On loopback it falls back
  to the request `Host`.
- **`server_status` leaks no internal state over http/sse**: only `model` and
  the image-size policy are reported remotely (device / dtype / VRAM / slots /
  load time are local stdio-only).
- **Upload tickets are one-shot and concurrency-safe**: a ticket is consumed
  only after a successful upload; a failed upload releases it for retry, and a
  concurrent POST using the same token is rejected (401).
- **Arbitrary-path writes are impossible**: tools accept no output path; the
  server never persists files.
- **Privacy**: user prompts and tool arguments are **never written to logs**
  (by design; `_log` only receives internal status text, and framework loggers
  are capped at WARNING so request data is not emitted).
- Input images (uploaded bytes or fetched URLs) are decoded in memory only;
  not retained.
- **`edit_image` reads host files only on a local stdio run**: over http/sse,
  `edit_image` accepts source images as `http(s)://` URLs — a server image URL
  or an external image URL. Server-side file paths / `file://` URIs are
  rejected outright (a `ValueError`), so a remote client cannot point the
  server at an arbitrary host file. A local (stdio) run may read paths because
  the client is on the same host and already trusted with the filesystem.
- **External fetches are SSRF-guarded**: URLs must be `http(s)` and resolve to
  public addresses only (loopback / private / link-local / reserved / CGNAT
  ranges are refused); every redirect hop is re-checked; fetches are bounded by
  `--max-body-mb` and a 30 s timeout. Server `/images/<id>` URLs resolve from
  the in-memory cache without network.
- **Image URLs are capability links**: each generated/uploaded-image URL embeds
  an unguessable id (192-bit random) and expires after `PICTURA_IMAGE_CACHE_TTL`;
  images are cached in RAM only (never on disk) and vanish with the process.
  `POST /images/upload` accepts a one-time `X-UPLOAD-TOKEN` (from
  `upload_image`) or the configured API key, unlike `GET /images/<id>`, which
  is capability-based.
- No output-directory control is offered: tools accept no output path and the
  server never persists images.

---

## 6. Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PICTURA_MODEL_CONFIG` | deployment-local `server/model.json` | path to the model config (model / vae / supported_loras / control_types) |
| `PICTURA_DEVICE` | `cuda` | `cuda` or `cpu` (auto-fallback to cpu) |
| `PICTURA_CUDA_DEVICE` | unset | restrict CUDA GPU(s) (`0`, `0,1`) → `CUDA_VISIBLE_DEVICES` |
| `PICTURA_MODELS_DIR` | `<project>/models` | root of the standard local model layout (plain dirs: `models/<org>/<repo>/`, `models/yolov8n-pose.pt`) |
| `PICTURA_MAX_CONCURRENT` | `auto` | render slot pool size: integer pins it, `1` = strictly serial, `auto` = sized from free VRAM |
| `PICTURA_PUBLIC_URL` | unset | required for non-loopback http/sse binds; externally visible base (scheme + host + path prefix); loopback fallback: request `Host` |
| `PICTURA_IMAGE_CACHE_TTL` | `600` | seconds an image download URL stays valid |
| `PICTURA_IMAGE_CACHE_MAX` | `64` | max images kept in the in-memory URL cache |
| `PICTURA_IMAGE_CACHE_MAX_MB` | `512` | max total bytes of the URL cache |
| `PICTURA_HOST` | `127.0.0.1` | bind address for http/sse (CLI `--host` overrides) |
| `PICTURA_PORT` | `8000` | TCP port for http/sse (CLI `--port` overrides) |
| `PICTURA_IMAGE_MAX_BODY_MB` | `16` | body cap for http/sse; bounds image uploads and external image fetches |
| `PICTURA_IMAGE_UPLOAD_TICKET_TTL` | `120` | upload ticket TTL (seconds) from `upload_image` |
| `PICTURA_API_KEY` | unset | API key; clients send it in the `PICTURA_API_KEY` header; fallback when `--api-key` not given |
| `PICTURA_LOG_FILE` | unset (stderr) | append `[pictura-mcp]` logs to a file (also `--log-file`); reopened on SIGHUP for logrotate |

---

## 7. Size limits (image input)

Source images are bounded server-side: `POST /images/upload` bodies and
external image fetches use the `--max-body-mb` cap (default **16 MB**), plenty
above the ~1.6 MB outputs and typical camera JPEGs. Raise
`PICTURA_IMAGE_MAX_BODY_MB` / `--max-body-mb` only if you really pass very large
sources. Stdio (local) has no body cap.

Generation and edit sizes stay at the native-bucket level of the active model
family, so compute and memory do not depend on the requested aspect ratio
(e.g. 1536×640 costs about the same as 1024×1024).

---

## 8. Client configuration (any MCP client)

The server is a standard MCP server (stdio, streamable HTTP, SSE) and is not
bound to any client. Every client stores server definitions in the same shape:

```jsonc
{
  "mcpServers": {
    "pictura": {
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["<PROJECT_ROOT>/server/pictura_server.py"],
      "env": { "PICTURA_MODEL_CONFIG": "<PROJECT_ROOT>/server/model.json" }
    }
  }
}
```

- **Generic / Claude Code / Cursor / Windsurf / VS Code**: place the block in a
  project `.mcp.json` (template: `deploy/mcp.json.example`) or in the client's
  own config file. See the README table for exact per-client locations.
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Remote (HTTP)**: `deploy/mcp.remote.json.example` — `url` + `PICTURA_API_KEY`
  header instead of `command`/`args`.

```json
{
  "mcpServers": {
    "pictura": {
      "url": "http://<HOST>:8000/mcp",
      "headers": { "PICTURA_API_KEY": "<TOKEN>" },
      "requestTimeoutMs": 600000
    }
  }
}
```

**Client timeout** (`requestTimeoutMs`, supported by pi's MCP adapter and most
harnesses): set it generously — a generation at the family default (1024²,
30-40 steps) takes tens of seconds to minutes, and under parallel load a job
may additionally wait for a free slot or a lazily built one. The MCP SDK
default (60 s) times out on ordinary generations; the shipped templates use
`600000` (10 min).

Any other harness simply points its own client config at the same server — the
block above is identical regardless of harness.

---

## 9. Deployment

**Tracked vs gitignored:** only templates are committed. Real configs and
artifacts are gitignored and created locally on each machine (see §11): the
client config (e.g. `.mcp.json` — copy of `deploy/mcp.json.example`, real paths),
`deploy/pictura-mcp.env` (secrets), `server/model.json` (copy a preset from
`server/examples/`), `.venv/`, and `outputs/`.

- **Venv**: `.venv` (Python 3.12+; `requirements.txt` pins `torch>=2.9,<3.0`,
  verified on 2.11.0+cu130, CUDA-13-capable driver, ≥580). Rebuild with
  `server/requirements.txt` (needs a CUDA-13-capable driver, ≥580).
  **Volta/V100 (compute capability 7.0) machines**: use
  `server/requirements-v100.txt` (torch 2.7.1+cu126) instead — newer torch
  builds dropped the sm_70 kernels.
- **systemd (recommended for long-running / remote)**: can run under a
  **dedicated service account**. `deploy/install-systemd.sh <PROJECT_ROOT>
  [SERVICE_USER] [PORT]` (root) creates the unprivileged account, prepares the
  models/log dirs, renders `deploy/pictura-mcp.service` (system unit with
  `User=`/`Group=` + hardening) and enables it. The env file
  (`deploy/pictura-mcp.env`, gitignored) holds the API key / model / serving /
  log settings.
- **One process at a time**: keeping several servers alive exhausts VRAM and
  causes CUDA OOM. Use systemd instead of ad-hoc background processes.

---

## 10. Verified behavior (tests on this host)

- SD3.5 Large: 1344×768 / 40 steps ≈ 115 s per image (1 slot, proactive CPU
  offload, ~30/34 GB); depth ControlNet edit ≈ 181 s, canny ≈ 331 s.
- SD3.5 Large LoRAs verified (Photorealistic, trigger `photorealistic`; Anime,
  trigger `Anime 35`); `list_loras` returns the supported ids with
  descriptions.
- Large camera-JPEG source images accepted for img2img input (bounded by
  `--max-body-mb`).
- Remote with token: 401 on missing/wrong token; `POST /images/upload` +
  `GET /images/<id>` round-trip OK; initialize + tools/list OK.
- Bucket snapping: requested sizes snap to the active family's native
  training buckets; `server_status` reports `size_policy` / `snap_buckets` /
  `native_size`.
- Statelessness: `outputs/` unchanged after generation via MCP.

---

## 11. Project structure

**You get these when you clone the repo (sources & templates):**
```
README.md                        # usage guide (any MCP client)
SPEC.md                          # this document
LICENSE                          # MIT license
scripts/
  check.sh                       # lightweight dev checks (no GPU needed)
  fetch-models.sh                # provision weights (deployment step, needs hf CLI)
deploy/
  mcp.json.example               # client config TEMPLATE (project .mcp.json)
  mcp.remote.json.example        # HTTP client config TEMPLATE
  pictura-mcp.service              # systemd system-unit TEMPLATE (service account)
  install-systemd.sh             # root installer: account + dirs + unit
  pictura-mcp.env.example          # env TEMPLATE (API key / model / serving / log)
  logrotate.example              # logrotate config (copytruncate + SIGHUP option)
server/
  pictura_server.py                # MCP image server (the implementation)
  requirements.txt               # python deps
  requirements-v100.txt          # python deps for Volta/V100 (torch 2.7.1+cu126)
  examples/                      # model config PRESETS (model.sdxl / sd35-medium / sd35-large / qwen-image-2.1)
skills/
  README.md                      # Agent Skill install guide
  pictura-mcp/SKILL.md           # the Agent Skill (operating policy for agents)
.gitignore
```
**You must make these yourself on each machine (after cloning):**
```
.mcp.json                        # YOUR client config - copy deploy/mcp.json.example and fill in
deploy/pictura-mcp.env            # YOUR secrets - copy deploy/pictura-mcp.env.example, set the token
server/model.json                 # YOUR model config - copy server/examples/model.sd35-large.json (or model.qwen-image-2.1.json, another preset)
.venv/                           # python env - create with: python3 -m venv .venv (+ pip install -r server/requirements.txt; V100: -r server/requirements-v100.txt)
outputs/                         # created automatically later by: server/pictura_server.py --smoke
```
The exact steps live in the README (Setup → Repo layout note). Git tracking
policy (.gitignore) is not part of this spec.
