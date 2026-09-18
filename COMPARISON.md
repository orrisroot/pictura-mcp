# Feature comparison — our image-gen MCP vs established MCPs

Comparison sources: this repo (`SPEC.md`) + official docs surfaced via context7
(artokun/comfyui-mcp, wavespeedai/mcp-server, am0y/mcp-fal, Stability AI platform).

## 1. Comparison at a glance

| Dimension | **This project** (`pictura_server.py`) | **ComfyUI MCP** (`comfyui-mcp`) | **Stability AI** (official MCP, API) | **WaveSpeed MCP** | **fal.ai MCP** |
|---|---|---|---|---|---|
| Generation engine | local **diffusers** (SDXL default) | local **ComfyUI** (must run separately) | Stability **cloud API** (local-core option exists) | WaveSpeed **cloud API** | fal.ai **cloud API** |
| Self-contained | ✅ yes (no external app) | ❌ requires ComfyUI + Python env + models | ❌ needs Stability account | ❌ needs API key | ❌ needs API key |
| Runs on your own GPU (incl. low-VRAM consumer cards) | ✅ | ✅ (via ComfyUI) | ⚠️ (local-core option, tuned for datacenter GPUs) | ❌ (cloud) | ❌ (cloud) |
| CPU/GPU memory tuning built-in | ✅ fp16, slicing, **auto CPU offload**, OOM retry | ⚠️ depends on your ComfyUI setup | n/a | n/a | n/a |
| Cost | **0** (your electricity only) | **0** (local) | API billing | API billing | API billing |
| API key required | ❌ | ❌ | ✅ | ✅ (`WAVESPEED_API_KEY`) | ✅ (`FAL_KEY`) |
| Privacy (image data) | **stays on your host** | stays on your host | leaves host | leaves host | leaves host |
| Text-to-image | ✅ | ✅ | ✅ | ✅ | ✅ |
| Image-to-image / editing | ✅ (in-memory, incl. data-URI input) | ✅ (rich) | ✅ | ✅ | ✅ |
| ControlNet / LoRA / custom checkpoints | ⚠️ code-level (not exposed as tools) | ✅ (first-class, big ecosystem) | some (platform models) | ✅ model catalog | ✅ 600+ models |
| Video / audio / 3D | ❌ | ✅ | ✅ (platform) | ✅ | ✅ |
| Workflow authoring / graph editing | ❌ | ✅ (node workflows, natural language) | ❌ | ❌ | ❌ |
| Model switching | ✅ single env var (`PICTURA_MODEL`) | ✅ full model manager | platform catalog | `list_models` | `models` tool |
| Transports | **stdio + streamable HTTP + SSE** | stdio, `--http`, `--comfyui-url` | stdio / remote API | stdio (Docker available) | stdio |
| Remote / headless | ✅ + systemd unit | ✅ (LAN/VPS) | hosted | hosted | hosted |
| Auth on your server | ✅ API key (PICTURE_API_KEY header) | ⚠️ not emphasized (LAN) | platform auth | platform API key | platform API key |
| Server stateless (no files kept on host) | ✅ always (stdio: inline base64 / http-sse: short-lived download URL, RAM cache) | ⚠️ writes files/workflows | n/a | `local` output mode writes to disk | n/a (URLs on CDN) |
| Image input size cap | **16 MB body** (~12 MB img) default, tunable via `PICTURA_MAX_BODY_MB` / `--max-body-mb` | depends on upload | platform | URL/base64 support | CDN upload flow |
| Ecosystem / maturity | self-maintained, small | very large (Civitai workflows, plugins) | vendor, large | growing | large (600+ models) |

## 2. Where this project wins

- **Zero-cost, zero-account, fully private**: everything runs on your own GPU;
  no API key, no money, no image ever leaves the machine. All cloud MCPs
  (Stability / WaveSpeed / fal) charge per call and receive your images.
- **Self-contained**: no ComfyUI / reverse proxy / external engine to install —
  one Python server + Hugging Face cache.
- **Headless + stateless**: purpose-built for agent use; never writes to
  disk, returns base64 inline, identical behavior local or remote, and can run
  as a single systemd service behind an API key.
- **Low-VRAM-friendly engineering**: fp16, slicing, proactive CPU offload and OOM
  auto-retry are built in and default-on for SDXL at 1024 px.
- **Decent image input**: 16 MB request body by default (≈12 MB image) — well
  above Claude's ≈5 MB per-image inline limit and headroom over the ~1.6 MB
  outputs; raise `--max-body-mb` / `PICTURA_MAX_BODY_MB` only if you really
  pass very large sources.

## 3. Where established MCPs win

- **ComfyUI MCP**: the feature king for local generation — ControlNet, LoRAs,
  arbitrary workflows, video/audio, graph editing, model/plugin management, plus
  a support ecosystem (Civitai). If you want ComfyUI anyway, its MCP is a great
  control plane. Cost: a heavier runtime to install and maintain.
- **Cloud MCPs (Stability / WaveSpeed / fal)**: instant access to frontier models
  (FLUX, SD3, video) with no GPU ownership, auto-scaling, and huge model
  catalogs — for users who trade privacy/cost for convenience.

## 4. Recommended evolution path for this project

If the comparison favors gaps we care about, the cheapest wins to add here:

1. **`http(s)://` image input** in `edit_image` (fetch server-side) → mirrors
   cloud-MCP workflows.
2. **Batch / multi-seed generate** tool.
3. **Video gen** would require a different model family (e.g. Wan/LTX) — larger
   scope; not recommended for a low-VRAM card.
