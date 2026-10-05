# Pictura MCP — Specification

This document is the authoritative specification of `pictura-mcp`: the
image-gen backend (`server/pictura_server.py`) and how to connect any MCP
client (stdio, streamable HTTP, SSE).

---

## 1. Overview

```
MCP client ──────────────────▶ pictura_server.py ──────────────▶ GPU
            (stdio|HTTP|SSE)                      (SDXL/SD3.5/Qwen-Image 2.1)
```

- Text-to-image (`generate_image`)
- Image-to-image / editing (`edit_image`)
- Upload reservation (`upload_image`)
- Status introspection (`server_status`)

**Key policies**

- **Stateless**: the server never writes image files to disk. **stdio**
  returns the image inline as base64 `ImageContent`; **http/sse** returns a
  short-lived download URL in the result note (the image sits in an in-memory
  TTL cache served at `GET /images/<id>`). Everything is gone when the process
  exits.
- **Identical local & remote behavior**, except deliberately: image delivery
  (inline vs URL) and `edit_image` host-path reads (stdio only — see §2/§5).
- **Concurrency**: the event loop never blocks on GPU work. Rendering runs on
  a slot pool; the pool size is derived from measured free VRAM (per extra
  slot: another full weight set + activation footprint + reserve; capped;
  `PICTURA_MAX_CONCURRENT` overrides). Each slot owns independent pipeline
  instances so LoRA/offload/scheduler state never races.
- **Tools never accept a path**: the save location is not controllable
  through the MCP interface.

---

## 2. MCP tools

### `generate_image` — text-to-image

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required |
| `negative_prompt` | string | `""` | only used when CFG is active (Qwen-Image 2.1: `guidance_scale > 1`) |
| `width` / `height` | int | 0 = family default | any positive value snaps to the nearest native training bucket of the active model (min 256, multiple of 8) |
| `num_inference_steps` | int | 0 = family default | clamped to [10, 100] |
| `guidance_scale` | float | 0 = family default | mapped to `true_cfg_scale` for Qwen-Image 2.1 |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |

### `edit_image` — img2img editing

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required |
| `image` | string | — | required. Source: a server image URL (`http://<host>/images/<id>`, resolved from the in-memory cache) or an external image URL (fetched server-side, SSRF-guarded). On a local stdio run a host file path / `file://` URI is also accepted; over http/sse the server reads no host files |
| `negative_prompt` | string | `""` | only used when CFG is active (Qwen-Image 2.1: `guidance_scale > 1`) |
| `strength` | float | 0.6 | 0..1, higher = more change. Omitted from the `qwen-image-2.1` schema (its edit always runs the full step count) |
| `width` / `height` | int | 0 | 0 = keep the source size (also snapped); any positive value snaps to the nearest native bucket. Qwen-Image 2.1: the source is resized to the snapped pair first, then the pipeline derives the output from that aspect |
| `num_inference_steps` | int | 0 = family default | effective steps ≈ `steps × strength` |
| `guidance_scale` | float | 0 = family default | mapped to `true_cfg_scale` for Qwen-Image 2.1 |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |
| `control_type` | string | `""` | optional abstract ControlNet type applied to the source (`canny`, `depth`, `openpose`); empty = disabled. Exposed only when the config has `control_types` (see §3); otherwise absent from the schema |
| `control_scale` | float | 1.0 | ControlNet conditioning strength (~0.4–1.0); exposed under the same condition as `control_type` |

### `upload_image`

Reserves an upload over http/sse: returns a **one-time token** (TTL
`PICTURA_IMAGE_UPLOAD_TICKET_TTL`, default 120 s, single use) plus the
`POST /images/upload` URL and a ready-to-run `curl` line. The client then
POSTs the image bytes (raw or multipart `file`) with the `X-UPLOAD-TOKEN`
header. The token is consumed only after a successful upload, released for
retry on failure, and rejected when used twice or concurrently; the
`PICTURA_API_KEY` header is accepted as an alternative.

### `list_loras`

Returns the supported LoRA ids valid for the `lora` parameter.

### `list_control_types`

Returns the abstract ControlNet types valid for `control_type`. Registered
only when the active config has `control_types` and the family supports
ControlNet; a family without ControlNet does not expose this tool and
`edit_image`'s schema omits the ControlNet parameters.

### `server_status`

Reports `model` and the image-size policy (`size_policy`, `snap_buckets`,
`native_size`). On a local stdio run it additionally reports the internal
`device`, `dtype`, `offload`, `weights_gb`, `vram_gb`, `concurrency_slots`
and `load_seconds`; over http/sse the internal state is not exposed.

**All tools return**: over stdio an `ImageContent` (base64 PNG) + a
`TextContent` note; over http/sse a single `TextContent` note containing a
short-lived download URL (`http://<base>/images/<id>`, TTL
`PICTURA_IMAGE_CACHE_TTL`). On failure a text error is returned.

---

## 3. Models & pipeline

### Model configuration (single file)

All model settings live in one deployment-local file, `server/model.json`
(gitignored; presets in `server/examples/*.json`, selected by copying one to
`server/model.json` or pointing `PICTURA_MODEL_CONFIG` at it):

- `model` — base model: local path or `org/repo` id; family is auto-detected
  (`sdxl`, `sd35-medium`, `sd35-large`, `qwen-image-2.1`)
- `vae` — optional custom VAE id (`null` = auto)
- `families.<id>` — per-family settings: `desc`, `steps`, `guidance`,
  `width` / `height`, `auto_vae`, `buckets` (native resolutions; rotations
  are added automatically)
- `supported_loras` — id → description map (client-usable LoRAs)
- `control_types` — `pre` / `model` / `prep_model` per abstract type
  (`canny` / `depth` / `openpose`; ids hidden from clients)

Weights are plain local directories under `PICTURA_MODELS_DIR` (default
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

Requested sizes snap to the family's native training buckets (`buckets` in
the config), so output stays on the aspect/area combinations the model was
trained on.

### Memory strategy

1. fp16 weights loaded (bf16 for Qwen-Image 2.1); attention slicing + VAE
   slicing + VAE tiling enabled.
2. **Proactive CPU offload** when the model is SDXL with default res ≥ 768, or
   whenever weights exceed ~80% of VRAM. Qwen-Image 2.1 (~31 GB bf16) exceeds
   80% of a 24–32 GB card, so it always runs with model CPU offload
   (`text_encoder->transformer->vae`). Otherwise weights stay fully on GPU.
3. Runtime CUDA-OOM → auto `enable_model_cpu_offload()` + one retry.
4. img2img reuses the loaded pipeline (shared weights, no second copy).
   Qwen-Image 2.1 edits on the same unified `QwenImage21Pipeline` via its
   `image` argument (no separate img2img class).

### Family detection & exposure rules

The family is detected from the model directory's `model_index.json`
(`_class_name`; the transformer config distinguishes SD3.5 variants); repo
ids fall back to the id string. Unsupported families are rejected at startup
(exit 2).

LoRA ids are restricted to `supported_loras` in the config (`["*"]` = any
id). ControlNet is exposed as abstract `control_type`s, preprocessed
in-memory; backing models and preprocessors are configured server-side and
hidden from clients. **Empty `control_types` exposes no ControlNet**:
`edit_image` is registered without the `control_type` / `control_scale`
parameters (and, for `qwen-image-2.1`, `strength`) and the
`list_control_types` tool is not registered. Adding a `control_types` entry
re-exposes them under the same tool names.

### Downloads (manual, before first start)

- Client-supplied ids are bare `org/repo` only; URLs and local paths are
  rejected; weights load safetensors-only.
- Provision everything with `sudo scripts/fetch-models.sh` (reads
  `server/model.json`; written as plain local dirs: repo ids →
  `models/<org>/<repo>/`). See the README for per-model `hf download`
  examples, including the Qwen-Image 2.1 uncensored LoRA (a single file
  fetched from the `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` repo).
- The server never contacts Hugging Face at startup or at tool-call time; a
  missing model surfaces as a provisioning error pointing at
  `scripts/fetch-models.sh`.

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
| `--api-key <key>` | — | **required for http/sse**; clients send it via the `PICTURA_API_KEY` header |
| `--max-body-mb <n>` | 16 | max HTTP request body (http/sse); bounds image uploads and external image fetches |
| `--log-file <path>` | stderr | append `[pictura-mcp]` logs to a file (also `$PICTURA_LOG_FILE`) |
| `--smoke` | — | self-test (txt2img + img2img) writing to `<repo>/outputs/` |

- http → endpoint `/mcp` (streamable HTTP); sse → endpoint `/sse`
- **Image upload**: `upload_image` issues a one-time token (TTL
  `PICTURA_IMAGE_UPLOAD_TICKET_TTL`, default 120 s); POST raw bytes or
  multipart/form-data `file` to `/images/upload` with the `X-UPLOAD-TOKEN`
  header (body capped by `--max-body-mb`; the `PICTURA_API_KEY` header is
  also accepted). The image lands in the same in-memory TTL cache and is
  served back by `GET /images/<id>`.
- **Image downloads**: `GET /images/<id>` serves cached generated / uploaded
  images. `PICTURA_PUBLIC_URL` is required when binding a non-loopback
  address (the server refuses to start without it) so returned URLs are
  reachable; on loopback the request `Host` is used.
- **Image return mode**: stdio → inline base64; http/sse → short-lived URL
  only, unless no public base is resolvable (falls back to inline).

---

## 5. Security

- **Remote transports require an API key**: http/sse refuses to start without
  one; the GPU is otherwise reachable by any caller.
- **Non-loopback http/sse requires `PICTURA_PUBLIC_URL`** (refuses to start):
  the externally visible base so returned image URLs are reachable behind
  reverse proxies.
- **`server_status` exposes no internal state over http/sse**: only `model`
  and the size policy are reported remotely.
- **Upload tickets are one-shot and concurrency-safe**: consumed only after a
  successful upload; a failed upload releases the token for retry; a
  concurrent POST with the same token is rejected (401).
- **Arbitrary-path writes are impossible**: tools accept no output path; the
  server never persists files.
- **Privacy**: prompts and tool arguments are never written to logs
  (framework loggers are capped at WARNING so request data is not emitted).
- Input images (uploaded bytes or fetched URLs) are decoded in memory only,
  never retained.
- **`edit_image` reads host files only on a local stdio run**: over http/sse,
  host file paths / `file://` URIs are rejected outright, so a remote client
  cannot point the server at an arbitrary host file.
- **External fetches are SSRF-guarded**: URLs must be http(s) and resolve to
  public addresses only (loopback / private / link-local / reserved / CGNAT
  refused); every redirect hop is re-checked; fetches are bounded by
  `--max-body-mb` and a 30 s timeout. Server `/images/<id>` URLs resolve from
  the in-memory cache without network.
- **Image URLs are capability links**: an unguessable random id per image,
  expired after `PICTURA_IMAGE_CACHE_TTL`; images are cached in RAM only and
  vanish with the process.

---

## 6. Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PICTURA_MODEL_CONFIG` | deployment-local `server/model.json` | path to the model config (model / vae / supported_loras / control_types) |
| `PICTURA_DEVICE` | `cuda` | `cuda` or `cpu` (auto-fallback to cpu) |
| `PICTURA_CUDA_DEVICE` | unset | restrict CUDA GPU(s) (`0`, `0,1`) → `CUDA_VISIBLE_DEVICES` |
| `PICTURA_MODELS_DIR` | `<project>/models` | root of the standard local model layout |
| `PICTURA_MAX_CONCURRENT` | `auto` | render slot pool size: integer pins it, `1` = strictly serial, `auto` = sized from free VRAM |
| `PICTURA_PUBLIC_URL` | unset | required for non-loopback http/sse binds; externally visible base (scheme + host + path prefix) |
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
external image fetches use the `--max-body-mb` cap (default **16 MB**).
Raise `PICTURA_IMAGE_MAX_BODY_MB` / `--max-body-mb` only for very large
sources. Stdio (local) has no body cap.

Generation and edit sizes stay at the native-bucket level of the active
model family, so compute and memory do not depend on the requested aspect
ratio (e.g. 1536×640 costs about the same as 1024×1024).

---

## 8. Client configuration

The server is a standard MCP server (stdio, streamable HTTP, SSE); every
client stores server definitions in the same shape:

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

- **Generic / Claude Code / Cursor / Windsurf / VS Code**: project
  `.mcp.json` (template: `deploy/mcp.json.example`) or the client's own
  config file.
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Remote (HTTP)**: `deploy/mcp.remote.json.example` — `url` +
  `PICTURA_API_KEY` header instead of `command`/`args`:

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

**Client timeout** (`requestTimeoutMs`): set it generously — a generation at
the family default (1024², 30–40 steps) takes tens of seconds to minutes, and
under parallel load a job may additionally wait for a free slot. The MCP SDK
default (60 s) times out on ordinary generations; the templates use 600000
(10 min).

---

## 9. Deployment

**Tracked vs gitignored**: only templates are committed. Real configs and
artifacts are created locally on each machine (see §10): `.mcp.json`,
`deploy/pictura-mcp.env` (secrets), `server/model.json`, `.venv/`, `outputs/`.

- **Venv**: `.venv` (Python 3.12+; `requirements.txt` pins `torch>=2.9,<3.0`,
  CUDA-13-capable driver, ≥580). **Volta/V100 machines**: use
  `server/requirements-v100.txt` (torch 2.7.1+cu126) instead — newer torch
  builds dropped the sm_70 kernels.
- **systemd (recommended for long-running / remote)**: `deploy/install-systemd.sh
  <PROJECT_ROOT> [SERVICE_USER] [PORT]` (root) creates the unprivileged
  account, prepares the models/log dirs, renders the unit
  (`deploy/pictura-mcp.service`) and enables it. The env file
  (`deploy/pictura-mcp.env`, gitignored) holds the API key / model / serving /
  log settings.
- **One process at a time**: keeping several servers alive exhausts VRAM and
  causes CUDA OOM. Use systemd instead of ad-hoc background processes.

---

## 10. Project structure

**Committed (sources & templates):**
```
README.md                        # usage guide (any MCP client)
SPEC.md                          # this document
LICENSE                          # MIT license
scripts/
  check.sh                       # lightweight dev checks (no GPU needed)
  fetch-models.sh                # provision weights (deployment step)
deploy/
  mcp.json.example               # client config TEMPLATE (project .mcp.json)
  mcp.remote.json.example        # HTTP client config TEMPLATE
  pictura-mcp.service            # systemd system-unit TEMPLATE
  install-systemd.sh             # root installer: account + dirs + unit
  pictura-mcp.env.example        # env TEMPLATE (API key / model / serving / log)
  logrotate.example              # logrotate config (copytruncate + SIGHUP)
server/
  pictura_server.py              # MCP image server (the implementation)
  requirements.txt               # python deps
  requirements-v100.txt          # python deps for Volta/V100 (torch 2.7.1+cu126)
  examples/                      # model config PRESETS (sdxl / sd35-medium /
                                 #   sd35-large / qwen-image-2.1)
skills/
  README.md                      # Agent Skill install guide
  pictura-mcp/SKILL.md           # the Agent Skill (operating policy for agents)
.gitignore
```
**Created per machine (gitignored):**
```
.mcp.json                        # client config - from deploy/mcp.json.example
deploy/pictura-mcp.env           # secrets - from deploy/pictura-mcp.env.example
server/model.json                # model config - from server/examples/*.json
.venv/                           # python env (requirements.txt or
                                 #   requirements-v100.txt on V100)
outputs/                         # created by: server/pictura_server.py --smoke
```
Steps: README (§Setup). Git tracking policy (.gitignore) is not part of this
spec.
