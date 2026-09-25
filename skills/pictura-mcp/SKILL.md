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
Diffusion, SDXL by default). Images come back to you inline (stdio) or as a
short-lived download URL (http/sse); the server never writes files, so **you
save them yourself**.

## Tools

The tool names may be namespaced depending on the client (e.g. pi shows them
as `mcp__generate_image`). The canonical names are:

| Tool | Purpose |
|---|---|
| `generate_image` | text → image |
| `edit_image` | image → image (edit) |
| `list_loras` | valid LoRA ids for the `lora` parameter |
| `list_control_types` | valid abstract ControlNet types for `control_type` |
| `server_status` | model / device / VRAM info |

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
first (`list_loras`, `list_control_types`) — only allowlisted ids are accepted,
and arbitrary URLs or file paths are rejected.

## generate_image

```
generate_image(
  prompt,                       # required; English works best
  negative_prompt = "",         # "blurry, low quality" etc.
  width = 1024, height = 1024,  # SDXL default; snapped to an SDXL ~1MP bucket
  num_inference_steps = 30,     # clamps to [10,100]
  guidance_scale = 7.5,
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
  strength = 0.6,               # 0..1; higher = bigger change
  width = 0, height = 0,        # 0 = keep source size; else snapped to an SDXL ~1MP bucket
  num_inference_steps = 25,
  guidance_scale = 7.5,
  seed = -1,
  lora = "",
  control_type = "",            # e.g. "canny"; abstract, discover via list_control_types
  control_scale = 1.0           # ~0.4-1.0
)
```

**`image` — how to point at the source image:**

- **A server image URL** `http://<host>/images/<id>` is best: it comes out of
  `generate_image` / `edit_image` results (http/sse mode) or from a
  `POST /images/upload` response, and resolves from the in-memory cache.
- **An external `http(s)://` URL** is also accepted; the server fetches it
  (private/loopback addresses are refused).
- **On a local stdio run** you may pass a **host file path** or `file://` URI.
- **Uploading a local file over http/sse**: `POST /images/upload` with the
  image bytes in the body returns `{"image": ".../images/<id>", ...}` — pass
  that URL to `edit_image`.
- To chain an edit onto a generation: use the download URL from the
  `generate_image` result, or save the returned image locally and pass its
  path (stdio).

The tool schema shows which forms apply to the current run.

`control_type` preprocesses the source image internally (e.g. canny/depth/pose)
and steers the edit; it is **SDXL-only**.

## Prompting conventions

- Be specific: subject + style + lighting/composition (e.g.
  "a fluffy corgi in a spacesuit, neon synthwave, dramatic lighting").
- Use `negative_prompt` for common artifacts ("blurry, low quality, deformed").
- LoRA ids look like `org/repo` and take an optional weight (`:0.8`); verify
  with `list_loras`.

## Model family (SDXL only)

- The server supports the **SDXL family only** (`PICTURA_MODEL` must be an SDXL
  checkpoint; the default is
  `stabilityai/stable-diffusion-xl-base-1.0`). txt2img, img2img, LoRA and
  ControlNet are all supported. Non-SDXL models are rejected at startup.

## Notes

- First calls may download/load models and prefetch; expect them to be slow.
- **`edit_image` source input**: pass an `http(s)://` URL — a server image URL
  from `generate_image` / `edit_image` / `POST /images/upload` (resolved from
  the in-memory cache) or an external image URL (fetched, SSRF-guarded). On a
  local stdio run a host file path / `file://` URI is also accepted; over
  http/sse the server never reads host files. The tool schema shows which
  mode applies.
- Sizes: any requested width/height is snapped to the nearest SDXL training
  bucket (~1MP, multiples of 8) — matching a bucket keeps quality; off-bucket
  sizes (e.g. 512×512) cause tiled/duplicated patterns.
  With `width=0`/`height=0` the source size is snapped too, so the output
  aspect can differ slightly from a non-bucket source.
- **Privacy**: never write the user's prompt into log files, notes, or other
  persistent text.
- On errors, report the returned error text verbatim to the user.
