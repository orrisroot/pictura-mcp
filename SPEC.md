# Image generation MCP server — Specification

Status: **final** (all behavior is implemented, tested on a consumer GPU)

This document is the authoritative specification of `pictura-mcp`. It
covers the image-gen backend (`server/pictura_server.py`) and how to connect any
MCP client (stdio, streamable HTTP, SSE).

---

## 1. Overview

Local, GPU-backed image generation wrapped as an MCP server:

```
any MCP client/harness --(stdio | streamable HTTP | SSE)--> pictura_server.py
                                                        │
                                 diffusers (Stable Diffusion XL) ──┴─▶ local GPU
```

It is client-agnostic — any harness that speaks MCP (Claude Desktop, Cursor,
VS Code, Windsurf, local agent frameworks, …) can connect.

- Text-to-image (`generate_image`)
- Image-to-image / editing (`edit_image`)
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
| `negative_prompt` | string | `""` | |
| `width` | int | model-aware (1024 for SDXL, 512 else) | clamped to [256, 1024], multiple of 8 |
| `height` | int | model-aware | clamped to [256, 1024], multiple of 8 |
| `num_inference_steps` | int | 30 | clamped to [10, 100] |
| `guidance_scale` | float | 7.5 | |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |

### `edit_image`
Edits an existing image with a prompt (img2img).

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required |
| `image` | string | — | required. Source image: local file path / `file://` URI (local stdio run only) or `data:image/...;base64,...` URI (everywhere, portable). Over http/sse only `data:` URIs are accepted — the server reads no host files |
| `negative_prompt` | string | `""` | |
| `strength` | float | 0.6 | 0..1, higher = more change |
| `width` / `height` | int | 0 | 0 = keep source size; clamped to [256, 1024], multiple of 8 |
| `num_inference_steps` | int | 25 | effective steps ≈ `steps × strength` |
| `guidance_scale` | float | 7.5 | |
| `seed` | int | -1 | -1 = random |
| `lora` | string | `""` | optional LoRA adapters, `'huggingface/repo:weight,...'` |
| `control_type` | string | `""` | optional abstract ControlNet type applied to the source (`canny`, `depth`, `openpose`); server hides the model; empty = disabled |
| `control_scale` | float | 1.0 | ControlNet conditioning strength (~0.4–1.0) |

### `list_loras`
Returns the allowlisted LoRA ids valid for the `lora` parameter.

### `list_control_types`
Returns the abstract ControlNet types valid for `control_type`, with short
guidance. **No model identifiers are exposed** (they stay server-side).

### `server_status`
Reports `model`, `device`, `dtype`, `offload`, `weights_gb`, `vram_gb`,
`concurrency_slots`, `load_seconds`.

**All tools return**: over **stdio** an `ImageContent` (base64 PNG, mime
`image/png`) + a `TextContent` note; over **http/sse** a single `TextContent`
note containing a short-lived download URL (`http://<base>/images/<id>`, TTL
`PICTURA_IMAGE_URL_TTL`, default 600 s). On failure a text error is returned.

---

## 3. Models & pipeline

### Default model
- **`stabilityai/stable-diffusion-xl-base-1.0`** (fp16) with the fp16-safe VAE
  `madebyollin/sdxl-vae-fp16-fix` automatically swapped in (avoids black-image /
  NaN artifacts; no fp32-upcast / vae-tiling conflict).
- Nothing is stored in the repo; weights are cached by Hugging Face in the
  standard hub cache on load.

### Model-aware defaults
| Model | Default W×H | Default steps |
|---|---|---|
| SDXL (required) | 1024×1024 | 30 |

### Memory strategy (low VRAM)
1. fp16 weights loaded; attention slicing + VAE slicing + VAE tiling enabled.
2. **Proactive CPU offload** when the model is SDXL with default res ≥ 768, or
   whenever weights exceed ~80% of VRAM. Otherwise weights stay fully on GPU.
3. Runtime CUDA-OOM → auto `enable_model_cpu_offload()` + one retry.
4. img2img reuses the loaded components via `AutoPipelineForImage2Image.from_pipe`
   (shared weights, no second model copy).

### Model family (SDXL only)

This server supports the **SDXL family only**. `PICTURA_MODEL` must reference an
SDXL checkpoint (default `stabilityai/stable-diffusion-xl-base-1.0`; the id
must contain `xl`); finetunes and SDXL-derivative checkpoints work by simply
swapping the model id. Non-SDXL values are rejected at startup (exit 2).

When swapping in an SDXL finetune, review the LoRA / ControlNet allowlists:
LoRA ids are default-allowlisted for the base checkpoints, so other adapters
require `PICTURA_LORA_ALLOWLIST` (or `*`), and the ControlNet models must be
SDXL-compatible (the internal allowlist handles that).

### LoRA / ControlNet allowlist & downloads
- **Default allowlist** of generic SDXL LoRAs (`nerijs/pixel-art-xl`,
  `CiroN2022/toy-face`) via `PICTURA_LORA_ALLOWLIST`. ControlNet is exposed as
  abstract `control_type`s (`canny` / `depth` / `openpose`, in-memory
  preprocessed server-side); the backing model is resolved from an
  internal `PICTURA_CONTROLNET_ALLOWLIST` and is **not exposed to clients**.
- URLs and local paths are always rejected; weights load safetensors-only.
- Allowlisted models are **pre-downloaded at service startup** (skip with
  `PICTURA_SKIP_PREFETCH=1`); download location via `PICTURA_MODEL_CACHE_DIR`.

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
| `--api-key <key>` | none | require the API key (http/sse) via the `PICTURE_API_KEY` header |
| `--max-body-mb <n>` | 16 | max HTTP request body (http/sse); base64 image input lives here |
| `--smoke` | — | self-test (txt2img + img2img) writing to `<repo>/outputs/` |

- http → endpoint `/mcp` (streamable HTTP, JSON responses)
- sse → endpoint `/sse`
- **Image downloads**: `GET /images/<id>` serves cached generated images
  (see §1 / §6). Set `PICTURA_PUBLIC_URL` when binding `0.0.0.0` or behind
  NAT / a reverse proxy so the URLs returned to clients are reachable; without
  it the server falls back to inline returns with a warning.
- **Image return mode**: stdio → inline base64; http/sse → short-lived URL
  (URL only; no inline bytes), unless no public base is resolvable (falls back
  to inline).

---

## 5. Security

- **Remote transports require an API key** (see §6 `PICTURE_API_KEY`, sent via
  the `PICTURE_API_KEY` header) when the server is exposed beyond localhost.
  The GPU is otherwise reachable by any caller.
- **Arbitrary-path writes are impossible**: tools accept no output path; the
  server never persists files.
- **Privacy**: user prompts and tool arguments are **never written to logs**
  (by design; `_log` only receives internal status text, and framework loggers
  are capped at WARNING so request data is not emitted).
- Input images (data URIs) are decoded in memory only; not retained.
- **`edit_image` reads host files only on a local stdio run**: over http/sse,
  `edit_image` accepts source images only as `data:image/...;base64,...` URIs —
  server-side file paths / `file://` URIs are rejected outright (a
  `ValueError`), so a remote client cannot point the server at an arbitrary
  host file. A local (stdio) run may read paths because the client is on the
  same host and already trusted with the filesystem.
- **Image URLs are capability links**: each generated-image URL embeds an
  unguessable id (192-bit random) and expires after `PICTURA_IMAGE_URL_TTL`;
  images are cached in RAM only (never on disk) and vanish with the process.
- No output-directory control is offered: tools accept no output path and the
  server never persists images.

---

## 6. Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PICTURA_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | HF model id |
| `PICTURA_VAE` | unset | optional VAE override |
| `PICTURA_DEVICE` | `cuda` | `cuda` or `cpu` (auto-fallback to cpu) |
| `PICTURA_CUDA_DEVICE` | unset | restrict CUDA GPU(s) (`0`, `0,1`) → `CUDA_VISIBLE_DEVICES` |
| `PICTURA_MODEL_CACHE_DIR` | HF cache | model download/cache directory |
| `PICTURA_LORA_ALLOWLIST` | built-in default | override LoRA allowlist (comma-separated; `*` = any `org/repo`) |
| `PICTURA_CONTROLNET_ALLOWLIST` | built-in default | override ControlNet allowlist (same semantics) |
| `PICTURA_SKIP_PREFETCH` | unset | `1` = skip pre-downloading allowlisted models at startup |
| `PICTURA_MAX_CONCURRENT` | `auto` | render slot pool size: integer pins it, `1` = strictly serial, `auto` = sized from free VRAM |
| `PICTURA_PUBLIC_URL` | unset (request Host) | force the externally visible base URL for image links; default = the request's `Host` header (X-Forwarded-Proto honored when uvicorn trusts the proxy) |
| `PICTURA_FORWARDED_ALLOW_IPS` | `127.0.0.1` | uvicorn `--forwarded-allow-ips`; set to the reverse proxy's IP when it is not on localhost |
| `PICTURA_IMAGE_URL_TTL` | `600` | seconds an image download URL stays valid |
| `PICTURA_IMAGE_URL_MAX` | `64` | max images kept in the in-memory URL cache |
| `PICTURA_IMAGE_URL_MAX_MB` | `512` | max total bytes of the URL cache |
| `PICTURA_HOST` | `127.0.0.1` | bind address for http/sse (CLI `--host` overrides) |
| `PICTURA_PORT` | `8000` | TCP port for http/sse (CLI `--port` overrides) |
| `PICTURA_MAX_BODY_MB` | `16` | body cap for http/sse |
| `PICTURE_API_KEY` | unset | API key; clients send it in the `PICTURE_API_KEY` header; fallback when `--api-key` not given |
| `PICTURA_LOG_FILE` | unset (stderr) | append `[pictura-mcp]` logs to a file (also `--log-file`); reopened on SIGHUP for logrotate |

---

## 7. Size limits (image input)

Base64 images arrive inside the HTTP request body, so `--max-body-mb` governs.
Default is **16 MB ≈ 12 MB actual image** — plenty of headroom over the ~1.6 MB
outputs and typical camera JPEGs, and above Claude's ≈5 MB per-image inline
limit. (OpenAI allows up to 512 MB/request and Gemini ≈100 MB inline; raise
`PICTURA_MAX_BODY_MB` / `--max-body-mb` only if you really pass such large
sources.) Stdio (local) has no body cap.

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
      "env": { "PICTURA_MODEL": "stabilityai/stable-diffusion-xl-base-1.0" }
    }
  }
}
```

- **Generic / Claude Code / Cursor / Windsurf / VS Code**: place the block in a
  project `.mcp.json` (template: `deploy/mcp.json.example`) or in the client's
  own config file. See the README table for exact per-client locations.
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Remote (HTTP)**: `deploy/mcp.remote.json.example` — `url` + `Authorization`
  header instead of `command`/`args`.

```json
{
  "mcpServers": {
    "pictura": {
      "url": "http://<HOST>:8000/mcp",
      "headers": { "PICTURE_API_KEY": "<TOKEN>" },
      "requestTimeoutMs": 600000
    }
  }
}
```

**Client timeout** (`requestTimeoutMs`, supported by pi's MCP adapter and most
harnesses): set it generously — SDXL at default 1024²·30 steps takes tens of
seconds, and under parallel load a job may additionally wait for a free slot
or a lazily built one (a cold parallel burst can run minutes). The MCP SDK
default (60 s) times out on ordinary generations; the shipped templates use
`600000` (10 min — only a first-run cold cache download could exceed that).

Any other harness simply points its own client config at the same server — the
block above is identical regardless of harness.

---

## 9. Deployment

**Tracked vs gitignored:** only templates are committed. Real configs and
artifacts are gitignored and created locally on each machine (see §11): the
client config (e.g. `.mcp.json` — copy of `deploy/mcp.json.example`, real paths),
`deploy/pictura-mcp.env` (secrets), `.venv/`, and `outputs/`.

- **Venv**: `.venv` (@ Python 3.14 + torch 2.14 CUDA). Rebuild with
  `server/requirements.txt`. For CUDA-12.8-driver machines, install the
  `cu128` torch build first (torch 2.11.0, see `server/requirements-cu128.txt`).
- **systemd (recommended for long-running / remote)**: can run under a
  **dedicated service account**. `deploy/install-systemd.sh <PROJECT_ROOT>
  [SERVICE_USER] [PORT]` (root) creates the unprivileged account, prepares the
  model cache/log dirs, renders `deploy/pictura-mcp.service` (system unit with
  `User=`/`Group=` + hardening) and enables it. The env file
  (`deploy/pictura-mcp.env`, gitignored) holds the token/model/cache/log settings.
- **One process at a time**: keeping several servers alive exhausts VRAM and
  causes CUDA OOM. Use systemd instead of ad-hoc background processes.

---

## 10. Verified behavior (tests on this host)

- SDXL load ~3 s cached; 1024×1024/30 ≈ 31 s (CPU-offload), img2img 1024 ≈ 14 s.
- 10.7 MB JPEG (14 MB base64 body) accepted for img2img input.
- Remote with token: 401 on missing/wrong token; initialize + tools/list OK.
- Statelessness: `outputs/` unchanged after generation via MCP.

---

## 11. Project structure

**You get these when you clone the repo (sources & templates):**
```
README.md                        # usage guide (any MCP client)
SPEC.md                          # this document
COMPARISON.md                    # vs other image-gen MCPs
deploy/
  mcp.json.example               # client config TEMPLATE (project .mcp.json)
  mcp.remote.json.example        # HTTP client config TEMPLATE
  pictura-mcp.service              # systemd system-unit TEMPLATE (service account)
  install-systemd.sh             # root installer: account + dirs + unit
  pictura-mcp.env.example          # env TEMPLATE (token/model/cache/log)
  logrotate.example              # logrotate config (copytruncate + SIGHUP option)
server/
  pictura_server.py                # MCP image server (the implementation)
  requirements.txt               # python deps
.gitignore
```
**You must make these yourself on each machine (after cloning):**
```
.mcp.json                        # YOUR client config - copy deploy/mcp.json.example and fill in
deploy/pictura-mcp.env            # YOUR secrets - copy deploy/pictura-mcp.env.example, set the token
.venv/                           # python env - create with: python3 -m venv .venv (+ pip install -r server/requirements.txt)
outputs/                         # created automatically later by: server/pictura_server.py --smoke
```
The exact steps live in the README (Setup → Repo layout note). Git tracking
policy (.gitignore) is not part of this spec.
