# Image generation MCP server — Specification

Status: **final** (all behavior is implemented, tested on a consumer GPU)

This document is the authoritative specification of `pictura-mcp`. It
covers the image-gen backend (`server/image_server.py`) and how to connect any
MCP client (stdio, streamable HTTP, SSE).

---

## 1. Overview

Local, GPU-backed image generation wrapped as an MCP server:

```
any MCP client/harness --(stdio | streamable HTTP | SSE)--> image_server.py
                                                        │
                                 diffusers (Stable Diffusion XL) ──┴─▶ local GPU
```

It is client-agnostic — any harness that speaks MCP (Claude Desktop, Cursor,
VS Code, Windsurf, local agent frameworks, …) can connect.

- Text-to-image (`generate_image`)
- Image-to-image / editing (`edit_image`)
- Status introspection (`server_status`)

**Key policies**
- **Stateless**: the server never writes image files to disk in any mode.
  Results are returned inline as base64; the client decides where to save.
- **Identical local & remote behavior**: no mode-dependent output handling.
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
| `image` | string | — | required. Source image: host-local file path, `file://` URI, or `data:image/...;base64,...` URI (portable across machines) |
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
`load_seconds`.

**All tools return** `ImageContent` (base64 PNG, mime `image/png`) + a
`TextContent` note. The note states the suggested client-side filename and that
no file was written on the server. On failure a text error is returned.

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
| SDXL (`xl` in id) | 1024×1024 | 30 |
| any other (SD1.5, sd-turbo, FLUX…) | 512×512 | 30 |

### Memory strategy (low VRAM)
1. fp16 weights loaded; attention slicing + VAE slicing + VAE tiling enabled.
2. **Proactive CPU offload** when the model is SDXL with default res ≥ 768, or
   whenever weights exceed ~80% of VRAM. Otherwise weights stay fully on GPU.
3. Runtime CUDA-OOM → auto `enable_model_cpu_offload()` + one retry.
4. img2img reuses the loaded components via `AutoPipelineForImage2Image.from_pipe`
   (shared weights, no second model copy).

### Switching models
Set `IMAGE_MODEL`. Reasonable choices on this hardware:

| Model | Notes |
|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` (default) | Best balance of quality/speed |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | Lightweight |
| `stabilityai/sd-turbo` | Use `num_inference_steps=4, guidance_scale=1` |
| `black-forest-labs/FLUX.1-schnell` | Needs extra setup; img2img unsupported |

**Supported families:**

| Family | txt2img | img2img | LoRA | ControlNet |
|---|---|---|---|---|
| SDXL (`*xl*`) | ✅ | ✅ | ✅ | ✅ (first-class; abstract `control_type`) |
| SD 1.5 (`stable-diffusion-v1-5/*`) | ✅ | ✅ | ✅ (allowlist override) | ❌ (SDXL-only guard) |
| FLUX (`*flux*`) | ✅ | ❌ | ⚠️ | ❌ |

**SD 1.5 usage example** (fast/lightweight txt2img & img2img): set
`IMAGE_MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5` (+ `IMAGE_LORA_ALLOWLIST=*`
or SD1.5 LoRA ids; the built-in LoRA allowlist is SDXL). Defaults become
512×512 / 30 steps, ≈3 s @512/25. `control_type` on a non-SDXL model is rejected
with a clear error.

### LoRA / ControlNet allowlist & downloads
- **Default allowlist** of generic SDXL LoRAs (`nerijs/pixel-art-xl`,
  `CiroN2022/toy-face`) via `IMAGE_LORA_ALLOWLIST`. ControlNet is exposed as
  abstract `control_type`s (`canny` / `depth` / `openpose`, in-memory
  preprocessed server-side, **SDXL-only**); the backing model is resolved from an
  internal `IMAGE_CONTROLNET_ALLOWLIST` and is **not exposed to clients**.
- URLs and local paths are always rejected; weights load safetensors-only.
- Allowlisted models are **pre-downloaded at service startup** (skip with
  `IMAGE_SKIP_PREFETCH=1`); download location via `IMAGE_MODEL_CACHE_DIR`.

---

## 4. Transports & CLI

```
python server/image_server.py [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--transport stdio\|http\|sse` | `stdio` | MCP transport |
| `--host` | `127.0.0.1` | bind address (use `0.0.0.0` for remote) |
| `--port` | `8000` | TCP port |
| `--token <t>` | none | require `Authorization: Bearer <t>` (http/sse) |
| `--max-body-mb <n>` | 16 | max HTTP request body (http/sse); base64 image input lives here |
| `--smoke` | — | self-test (txt2img + img2img) writing to `<repo>/outputs/` |

- http → endpoint `/mcp` (streamable HTTP, JSON responses)
- sse → endpoint `/sse`

---

## 5. Security

- **Remote transports require a bearer token** (see §6 `IMAGE_MCP_TOKEN`) when
  the server is exposed beyond localhost. The GPU is otherwise reachable by any
  caller.
- **Arbitrary-path writes are impossible**: tools accept no output path; the
  server never persists files.
- **Privacy**: user prompts and tool arguments are **never written to logs**
  (by design; `_log` only receives internal status text, and framework loggers
  are capped at WARNING so request data is not emitted).
- Input images (data URIs) are decoded in memory only; not retained.
- Remote `output_dir`-style control is not offered at all (removed by design).

---

## 6. Environment variables

| Var | Default | Meaning |
|---|---|---|
| `IMAGE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | HF model id |
| `IMAGE_VAE` | unset | optional VAE override |
| `IMAGE_DEVICE` | `cuda` | `cuda` or `cpu` (auto-fallback to cpu) |
| `IMAGE_CUDA_DEVICE` | unset | restrict CUDA GPU(s) (`0`, `0,1`) → `CUDA_VISIBLE_DEVICES` |
| `IMAGE_LOG_FILE` | unset (stderr) | append `[image-mcp]` logs to a file (also `--log-file`); reopened on SIGHUP for logrotate |
| `IMAGE_MCP_TOKEN` | unset | bearer token; fallback when `--token` not given |
| `IMAGE_MAX_BODY_MB` | `16` | body cap for http/sse |
| `IMAGE_MODEL_CACHE_DIR` | HF cache | model download/cache directory |
| `IMAGE_LORA_ALLOWLIST` | built-in default | override LoRA allowlist (comma-separated; `*` = any `org/repo`) |
| `IMAGE_CONTROLNET_ALLOWLIST` | built-in default | override ControlNet allowlist (same semantics) |
| `IMAGE_SKIP_PREFETCH` | unset | `1` = skip pre-downloading allowlisted models at startup |

---

## 7. Size limits (image input)

Base64 images arrive inside the HTTP request body, so `--max-body-mb` governs.
Default is **16 MB ≈ 12 MB actual image** — plenty of headroom over the ~1.6 MB
outputs and typical camera JPEGs, and above Claude's ≈5 MB per-image inline
limit. (OpenAI allows up to 512 MB/request and Gemini ≈100 MB inline; raise
`IMAGE_MAX_BODY_MB` / `--max-body-mb` only if you really pass such large
sources.) Stdio (local) has no body cap.

---

## 8. Client configuration (any MCP client)

The server is a standard MCP server (stdio, streamable HTTP, SSE) and is not
bound to any client. Every client stores server definitions in the same shape:

```jsonc
{
  "mcpServers": {
    "generate-image": {
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["<PROJECT_ROOT>/server/image_server.py"],
      "env": { "IMAGE_MODEL": "stabilityai/stable-diffusion-xl-base-1.0" }
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
    "generate-image": {
      "url": "http://<HOST>:8000/mcp",
      "headers": { "Authorization": "Bearer <TOKEN>" }
    }
  }
}
```

Any other harness simply points its own client config at the same server — the
block above is identical regardless of harness.

---

## 9. Deployment

**Tracked vs gitignored:** only templates are committed. Real configs and
artifacts are gitignored and created locally on each machine (see §11): the
client config (e.g. `.mcp.json` — copy of `deploy/mcp.json.example`, real paths),
`deploy/image-mcp.env` (secrets), `.venv/`, and `outputs/`.

- **Venv**: `.venv` (@ Python 3.14 + torch 2.14 CUDA). Rebuild with
  `server/requirements.txt`.
- **systemd (recommended for long-running / remote)**: can run under a
  **dedicated service account**. `deploy/install-systemd.sh <PROJECT_ROOT>
  [SERVICE_USER] [PORT]` (root) creates the unprivileged account, prepares the
  model cache/log dirs, renders `deploy/image-mcp.service` (system unit with
  `User=`/`Group=` + hardening) and enables it. The env file
  (`deploy/image-mcp.env`, gitignored) holds the token/model/cache/log settings.
- **One process at a time**: keeping several servers alive exhausts VRAM and
  causes CUDA OOM. Use systemd instead of ad-hoc background processes.

---

## 10. Verified behavior (tests on this host)

- SDXL load ~3 s cached; 1024×1024/30 ≈ 31 s (CPU-offload), img2img 1024 ≈ 14 s.
- 10.7 MB JPEG (14 MB base64 body) accepted for img2img input.
- Remote with token: 401 on missing/wrong token; initialize + tools/list OK.
- Statelessness: `outputs/` unchanged after generation via MCP.
- Latent bug found/removed during validation: stray `path` reference in
  `edit_image`.

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
  image-mcp.service              # systemd system-unit TEMPLATE (service account)
  install-systemd.sh             # root installer: account + dirs + unit
  image-mcp.env.example          # env TEMPLATE (token/model/cache/log)
  logrotate.example              # logrotate config (copytruncate + SIGHUP option)
server/
  image_server.py                # MCP image server (the implementation)
  requirements.txt               # python deps
.gitignore
```
**You must make these yourself on each machine (after cloning):**
```
.mcp.json                        # YOUR client config - copy deploy/mcp.json.example and fill in
deploy/image-mcp.env            # YOUR secrets - copy deploy/image-mcp.env.example, set the token
.venv/                           # python env - create with: python3 -m venv .venv (+ pip install -r server/requirements.txt)
outputs/                         # created automatically later by: server/image_server.py --smoke
```
The exact steps live in the README (Setup → Repo layout note). Git tracking
policy (.gitignore) is not part of this spec.
