---
name: pictura-mcp
description: >-
  Generate and edit images with Pictura MCP, a local GPU Stable Diffusion
  server (text-to-image, image editing, LoRA, ControlNet). Use whenever the
  agent is asked to create, edit, transform, or stylize images through its
  image tools, including saving the returned images to disk and following
  model-family limits.
version: 0.1.0
license: MIT
---

# Pictura MCP

Pictura MCP is a **local, stateless** image-generation MCP server (Stable
Diffusion — SDXL or SD3.5 — or Qwen-Image 2.1, active family from the local
model config). Images
come back to you inline (stdio) or as a
short-lived download URL (http/sse); the server never writes files, so **you
save them yourself**.

## Tools

The tool names may be namespaced depending on the client (e.g. pi shows them
as `mcp__generate_image`). The canonical names are:

| Tool | Purpose |
|---|---|
| `generate_image` | text → image |
| `edit_image` | image → image (edit) |
| `upload_image` | local file → one-time upload reservation (http/sse; stdio: use local path) |
| `list_loras` | valid LoRA ids for the `lora` parameter |
| `list_control_types` | valid abstract ControlNet types for `control_type` (only present for families with ControlNet) |
| `server_status` | model + image-size policy (stdio adds device / VRAM etc.) |

## Saving images (important)

The server is stateless: **nothing is written to any disk**. How the image
arrives depends on the transport you are connected over (the result note says
which):

- **stdio (local)**: the image is returned inline as base64 `ImageContent`
  (data field). Capture it, decode it into a PNG, and write it to a sensible
  path (e.g. `./outputs/YYYYMMDD_prompt.png`; create the directory first).
- **http/sse (remote)**: the result note contains a **short-lived download
  URL** (`.../images/<id>`, valid ~10 min). Fetch it (curl / an HTTP tool) and
  save the bytes to a file, e.g. `curl -sSf <url> -o ./outputs/xxx.png`.

Then report the saved path to the user. If you cannot save, tell the user the
image is available (inline / at the URL) but not yet stored.

> Whatever the mode, never tell the user a file was written — only the paths
> you actually created.

## Verify before you use

When unsure which LoRA/ControlNet values are valid, **call the discovery tools**
first (`list_loras`, and `list_control_types` when it is offered) — only
supported ids are accepted, and arbitrary URLs or file paths are rejected. If
the active family has no ControlNet support, `list_control_types` is not
offered and `edit_image` has no `control_type` parameter — do not pass one.

## generate_image

```
generate_image(
  prompt,                       # required; English works best
  negative_prompt = "",         # "blurry, low quality" etc.
  width = 0, height = 0,        # 0 = model-family default; snapped to a native bucket
  num_inference_steps = 0,      # 0 = model-family default (SDXL 30 / SD3.5 40)
  guidance_scale = 0,           # 0 = model-family default (SDXL 7.0 / SD3.5 4.5)
  seed = -1,                    # -1 = random
  lora = ""                     # "huggingface/repo:weight" (repeatable with commas)
)
```

## edit_image (img2img)

```
edit_image(
  prompt,                       # required
  image,                        # required: http(s) URL, or file path (stdio/local)
  negative_prompt = "",
  strength = 0.6,               # 0..1; higher = bigger change (SDXL/SD3.5; not a Qwen param)
  width = 0, height = 0,        # 0 = keep source size; else snapped to a native bucket
  num_inference_steps = 0,      # 0 = model-family default
  guidance_scale = 0,           # 0 = model-family default (Qwen: >1 + negative = CFG)
  seed = -1,
  lora = "",
  control_type = "",            # e.g. "canny"; ONLY exists for families with ControlNet
  control_scale = 1.0           # ~0.4-1.0 (same condition as control_type)
)
```

> For the `qwen-image-2.1` family (no ControlNet, no strength):
> `edit_image`'s schema omits `control_type`, `control_scale` **and `strength`**
> entirely — do not pass them. The unified pipeline always runs the full step
> count on the source-conditioned edit.

**`image` — how to point at the source image:**

- **A server image URL** `http://<host>/images/<id>` is best: it comes out of
  `generate_image` / `edit_image` results (http/sse mode) or from a
  `POST /images/upload` response, and resolves from the in-memory cache.
- **An external `http(s)://` URL** is also accepted; the server fetches it
  (private/loopback addresses are refused).
- **On a local stdio run** you may pass a **host file path** or `file://` URI.
- **Uploading a local file over http/sse**: call the `upload_image` tool to
  reserve an upload — it returns a **one-time token** and a ready-to-run
  `curl` line; then POST the bytes (raw or multipart `file` field) to the
  returned URL with the `X-UPLOAD-TOKEN` header. The response is
  `{"image": ".../images/<id>", ...}` — pass that URL to `edit_image`. (Direct
  `PICTURA_API_KEY`-header uploads also work.)
- **Downloading results over http/sse**: the download URLs are short-lived
  capability links; fetch and save them.
- To chain an edit onto a generation: use the download URL from the
  `generate_image` result, or save the returned image locally and pass its
  path (stdio).

The tool schema shows which forms apply to the current run.

When it exists, `control_type` preprocesses the source image internally
(e.g. canny/depth/pose) and steers the edit; the valid types and their backing
models are configured in `server/model.json` (`control_types`) — call
`list_control_types` for the current set. When the tool is absent, the active
family has no ControlNet support and `edit_image` accepts no `control_type`.

## Prompting conventions

- Be specific: subject + style + lighting/composition (e.g.
  "a fluffy corgi in a spacesuit, neon synthwave, dramatic lighting").
- Use `negative_prompt` for common artifacts ("blurry, low quality, deformed").
- LoRA ids look like `org/repo` and take an optional weight (`:0.8`); verify
  with `list_loras`.

## Model families

- Supported families: **SDXL** (`sdxl`), **SD3.5** (`sd35-medium` /
  `sd35-large`), and **Qwen-Image 2.1** (`qwen-image-2.1`); the active model
  comes from `server/model.json` (`model`).
  - SDXL / SD3.5: txt2img, img2img, LoRA and ControlNet are all supported.
  - Qwen-Image 2.1: txt2img, img2img (image-conditioned editing on the same
    unified pipeline) and LoRA are supported; **ControlNet is not**.
  - Qwen-Image 2.1 defaults: guidance 1.0 = no classifier-free guidance; pass
    `guidance_scale` > 1 with a `negative_prompt` to enable CFG.
- Unsupported families are rejected at startup.

## Notes

- First calls load the local models; expect the first generation to be slow
  (the pipeline builds on first use).
- **`edit_image` source input**: pass an `http(s)://` URL — a server image URL
  from `generate_image` / `edit_image` / `POST /images/upload` (resolved from
  the in-memory cache) or an external image URL (fetched, SSRF-guarded). On a
  local stdio run a host file path / `file://` URI is also accepted; over
  http/sse the server never reads host files. The tool schema shows which
  mode applies.
- Sizes: any requested width/height is snapped to the nearest native training
  bucket of the active model family (multiples of 8) — matching a bucket keeps
  quality; off-bucket sizes (e.g. 512×512) cause tiled/duplicated patterns.
  With `width=0`/`height=0` the source size is snapped too, so the output
  aspect can differ slightly from a non-bucket source.
- **Privacy**: never write the user's prompt into log files, notes, or other
  persistent text.
- On errors, report the returned error text verbatim to the user.
