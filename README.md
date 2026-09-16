# Pictura MCP

A **local, GPU-backed image generation MCP server** (codename `pictura-mcp`) exposed through the standard Model Context Protocol. It works with **any
MCP client** — Claude Desktop, Cursor, VS Code, Windsurf, Claude Code, etc.

```
any MCP client ──(stdio | streamable HTTP | SSE)──▶ pictura_server.py (Python)
                                                          │
                                   diffusers (SDXL) ───────┴─▶ local GPU
```

- **Server**: `server/pictura_server.py` — a standard MCP server (stdio / HTTP /
  SSE) running Stable Diffusion via Hugging Face `diffusers`.
- **Client**: whatever MCP client you already use. Examples for several clients
  are in § [Client configuration](#client-configuration).

## Features / tools

| Tool | Description |
|---|---|
| `generate_image` | text → image (SDXL default) |
| `edit_image` | image → image (img2img / edit from a prompt) |
| `list_loras` | allowlisted LoRA ids (for `lora`) |
| `list_control_types` | abstract ControlNet types (for `control_type`) |
| `server_status` | model / device / VRAM info |

Both generation tools accept optional **LoRA** adapters, and `edit_image`
additionally supports **ControlNet** via an abstract `control_type` (see
[LoRA & ControlNet](#lora--controlnet)). Parameter meaning and value formats are
embedded in each tool's input schema (visible to MCP clients); `list_loras` /
`list_control_types` return the valid values, and ControlNet model identifiers
stay server-side.

Stateless by design: images are returned inline as base64 and **never written to
the server disk**; local and remote behavior are identical.

## Setup

```bash
# 1) Python venv + dependencies (torch is the CUDA build, ~3GB)
python3 -m venv .venv
./.venv/bin/pip install -r server/requirements.txt

# 2) Smoke test (downloads the model, ~5GB, on first run)
./.venv/bin/python server/pictura_server.py --smoke
# -> OK if outputs/smoke_test.png is created

# 3) Connect from your MCP client (see below)
```

### CUDA driver version

The default install above pulls the latest torch from PyPI, whose wheels are
built for **CUDA 13** (`cu130`) — this needs a CUDA-13-capable driver
(**≥ 580**). Check yours with `nvidia-smi` (top-right corner).

If your driver only supports up to **CUDA 12.8** (e.g. 570.x drivers), install
the `cu128` torch build instead — same code, older CUDA runtime:

```bash
python3 -m venv .venv
./.venv/bin/pip install "torch==2.11.0" "torchvision" \
    --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/pip install -r server/requirements.txt   # keeps 2.11.0+cu128
```

(`server/requirements-cu128.txt` documents this variant; cu128 wheels exist up
to torch 2.11.x, and all other deps only require torch≥2.6.) Verify with
`./.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
— expect e.g. `2.11.0+cu128` and `True`.

> **Repo layout note (gitignored files).** The committed repository ships
> **templates** (`deploy/mcp.json.example`, `deploy/pictura-mcp.env.example`,
> `deploy/pictura-mcp.service`). Local files that are **gitignored** and created
> per machine: your real client config (e.g. `.mcp.json` — copy from
> `deploy/mcp.json.example`, replace `<PROJECT_ROOT>`), `deploy/pictura-mcp.env`
> (secrets — never commit), plus `.venv/` and `outputs/`.

## Client configuration

Every MCP client stores server definitions in the same shape
(`mcpServers` → name → `command`/`args`/`env`). Point it at the server:

```jsonc
{
  "mcpServers": {
    "generate-image": {
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["<PROJECT_ROOT>/server/pictura_server.py"],
      "env": { "IMAGE_MODEL": "stabilityai/stable-diffusion-xl-base-1.0" }
    }
  }
}
```

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
  --token my-secret-token
```

- `--transport http` (endpoint `/mcp`) or `--transport sse` (endpoint `/sse`);
  `--host 127.0.0.1` is the safe default — use `0.0.0.0` for remote clients
- `--token <token>` (or env `PICTURA_MCP_TOKEN`) requires
  `Authorization: Bearer <token>` on every request; **always set it when the
  server is reachable beyond localhost**
- `--max-body-mb <MB>` (default 16) caps the HTTP request body; base64 images
  arrive in the body
- The server is **stateless**: no image files are written on the server in any
  mode; clients receive base64 and save where they like

Remote client config (`deploy/mcp.remote.json.example`):

```json
{
  "mcpServers": {
    "generate-image": {
      "url": "http://<SERVER_HOST_OR_IP>:8000/mcp",
      "headers": { "Authorization": "Bearer <TOKEN>" }
    }
  }
}
```

### systemd (recommended for long-running / remote)

There are two ways to run it as a service.

**A) Dedicated service account (system scope, recommended):**

```bash
# As root - creates the service user, prepares dirs, renders & installs the unit
sudo deploy/install-systemd.sh /absolute/path/to/this/repo pictura-mcp 8000
#   then edit deploy/pictura-mcp.env (token, model, IMAGE_CUDA_DEVICE,
#   IMAGE_LOG_FILE, IMAGE_MODEL_CACHE_DIR) and: sudo systemctl restart pictura-mcp
sudo systemctl status pictura-mcp
```

This runs under the unprivileged `pictura-mcp` system account with hardening
(`NoNewPrivileges`, `ProtectSystem`, `PrivateTmp`, …). The model cache and log
paths in `deploy/pictura-mcp.env` are created/owned by that account (**log file is
0640, owner = service account, group = service account**). Non-root operators
read the log by joining the group once: `sudo usermod -aG pictura-mcp <username>`
(then log out/in). If the service crashes at startup, remove
`MemoryDenyWriteExecute=true` from the unit (torch sometimes conflicts) and
`systemctl daemon-reload && restart`.

**B) Current user (user scope, quick):**

```bash
cp deploy/pictura-mcp.env.example deploy/pictura-mcp.env   # set PICTURA_MCP_TOKEN
chmod 600 deploy/pictura-mcp.env
mkdir -p ~/.config/systemd/user
sed 's#<PROJECT_ROOT>#/absolute/path/to/this/repo#' \
  deploy/pictura-mcp.service > ~/.config/systemd/user/pictura-mcp.service
# remove the User= / Group= lines for a user unit
systemctl --user daemon-reload && systemctl --user enable --now pictura-mcp
systemctl --user status pictura-mcp
```

Log rotation: `deploy/logrotate.example` (copytruncate, or SIGHUP postrotate).

> Verified end-to-end: `tools/list` returns `generate_image`, `edit_image`,
> `server_status`; missing/wrong tokens get 401; clients connect over stdio and
> over HTTP.

## Switching models (env `IMAGE_MODEL`)

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` (default) | ~30s @1024, 30 steps | **High** | fp16-safe VAE + CPU offload on lower VRAM |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | Fast (~3s @512, 25 steps) | Standard | lightweight |
| `stabilityai/sd-turbo` | Very fast (4 steps) | Good | use `num_inference_steps=4, guidance_scale=1` |
| `black-forest-labs/FLUX.1-schnell` | Slow | High | extra FLUX setup; img2img unsupported |

Set `IMAGE_MODEL` in your client's server `env` (or the systemd env file), then
restart/reconnect the client. Defaults are model-aware: **1024×1024 / 30 steps**
for SDXL. On lower-VRAM cards the SDXL weights auto-fall back to CPU offload; CUDA-OOM at
runtime also auto-offloads and retries.

### Supported model families

| Family | txt2img | img2img | LoRA | ControlNet | Notes |
|---|---|---|---|---|---|
| **SDXL** (`*xl*`) | ✅ | ✅ | ✅ | ✅ | **first-class**; default; abstract `control_type` (`canny`/`depth`/`openpose`) |
| **SD 1.5** (`stable-diffusion-v1-5/*`) | ✅ | ✅ | ✅ (allowlist override) | ❌ (SDXL-only) | 512×512 default; lightweight |
| **FLUX** (`*flux*`) | ✅ | ❌ | ⚠️ (extra setup) | ❌ | needs extra FLUX setup |

### SD 1.5 usage example

ControlNet is SDXL-only, so switch to an SD1.5 checkpoint for fast/lightweight
*text-to-image* (and img2img). Because the **built-in LoRA allowlist is SDXL**, set
`IMAGE_LORA_ALLOWLIST=*` (any id) or your own SD1.5 LoRA ids. Example client `env`:

```jsonc
{
  "mcpServers": {
    "generate-image": {
      "command": "<PROJECT_ROOT>/.venv/bin/python",
      "args": ["<PROJECT_ROOT>/server/pictura_server.py"],
      "env": {
        "IMAGE_MODEL": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "IMAGE_LORA_ALLOWLIST": "*"          // or e.g. "some/org/sd15-style-lora"
      }
    }
  }
}
```

```bash
# one-off run for SD1.5
IMAGE_MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5 \
IMAGE_LORA_ALLOWLIST='*' \
./.venv/bin/python server/pictura_server.py --transport http --token <TOKEN>
```

- Defaults become **512×512 / 30 steps**, generation ≈ 3 s @512/25.
- Calling `control_type` on a non-SDXL model returns a clear error (no bad loads).

## Image editing (img2img)

`edit_image(prompt, image, ...)` transforms an existing image:

- **`image`**: host-local file path, `file://` URI, or `data:image/...;base64,...`
  URI (portable across machines)
- **`strength`** (0..1, default 0.6): higher = larger change
- **`width`/`height`** (0 = keep source size; clamp ≤1024, multiple of 8)

Reuses loaded weights via `AutoPipelineForImage2Image.from_pipe` (no second
model copy). `--smoke` also exercises the img2img path.

## Input size & VRAM notes

- Base64 image input travels in the request body → governed by
  `--max-body-mb` (default **16 MB ≈ 12 MB image**), ample headroom over the
  ~1.6 MB outputs and typical camera JPEGs. Raise it only if you need to pass
  very large source images.
- fp16 + attention/VAE slicing + proactive CPU offload + OOM auto-retry are all
  baked in for low-VRAM cards.
- GPU selection: set `IMAGE_CUDA_DEVICE` (e.g. `0` or `0,1`) to restrict which
  CUDA GPU(s) the server uses (`CUDA_VISIBLE_DEVICES`).
- Log destination: set `IMAGE_LOG_FILE` (or `--log-file <path>`) to append the
  `[pictura-mcp]` log to a file instead of stderr/journald (handy for systemd).
  Logrotate-ready: the server reopens its log file on `SIGHUP`, and a
  `copytruncate`-based config is provided in `deploy/logrotate.example`.
  **Privacy: user prompts and tool arguments are never written to any log.**

## LoRA & ControlNet

**LoRA** (both `generate_image` and `edit_image`): pass `lora` as comma-separated
`huggingface/repo:weight` entries (weight defaults to 1.0). Adapters download
from the HF cache on first use. Example:

```text
lora = "nerijs/pixel-art-xl:0.8,CiroN2022/toy-face:0.6"
```

**ControlNet** (`edit_image` only): pass `control_type` — an abstract type
applied to the source image (`canny`, `depth`, `openpose`). The server runs the
preprocessor in-memory and picks/hides the backing model; `control_scale`
(0.4–1.0) tunes the strength. Call `list_control_types` for the valid types.
Example:

```text
control_type = "canny"   # auto-extract edges from the source, then ControlNet
control_scale = 0.9
```

Requires the `peft` dependency (listed in `server/requirements.txt`).

**Allowlist & downloads**
- Client-supplied `lora` ids are restricted to a **built-in default allowlist**,
  extendable via `IMAGE_LORA_ALLOWLIST` (comma-separated overrides;
  `*` = allow any bare `org/repo` id). URLs, local paths and path traversal are
  always rejected, and weights load safetensors-only.
- **ControlNet ids are server-side and hidden** — clients only choose an abstract
  `control_type`; the backing model is resolved from an internal allowlist
  (`IMAGE_CONTROLNET_ALLOWLIST`).
- Allowlisted models are **pre-downloaded at service startup** into the model
  cache, so tool calls don't pay the download cost. Set `IMAGE_SKIP_PREFETCH=1`
  to disable.
- Download location: `IMAGE_MODEL_CACHE_DIR` (default: the Hugging Face cache).

## Docs

- `SPEC.md` — full technical specification
- `COMPARISON.md` — feature comparison vs other image-gen MCPs
- `skills/` — distributable **Agent Skill** (`pictura-mcp`) for end-user agents (see `skills/README.md`)
- `deploy/` — configuration templates + systemd unit
