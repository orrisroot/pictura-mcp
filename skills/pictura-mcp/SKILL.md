---
name: pictura-mcp
description: >-
  Generate and edit images with Pictura MCP, a local GPU image server
  (text-to-image, image editing, LoRA, ControlNet where the model family
  provides it). Use whenever the agent is asked to create, edit, transform,
  or stylize images through its image tools, including saving the returned
  images to disk and following model-family limits.
version: 0.1.0
license: MIT
---

# Pictura MCP

Pictura MCP is a **local, stateless** image-generation MCP server
(SDXL / SD3.5 / Qwen-Image 2.1 — the active family comes from the local
model config). The server never writes files, so **you save the images
yourself**: inline base64 (stdio) or a short-lived download URL (http/sse).

## Tools

| Tool | Purpose |
|---|---|
| `generate_image` | text → image |
| `edit_image` | image → image (edit) |
| `upload_image` | local file → one-time upload reservation (http/sse; stdio: use local path) |
| `list_loras` | valid LoRA ids for the `lora` parameter |
| `list_control_types` | valid ControlNet types for `control_type` (only present when the family supports ControlNet) |
| `server_status` | model + image-size policy (stdio adds device / VRAM etc.) |

## Saving images (important)

Nothing is written to any disk. Save the image yourself:

- **stdio (local)**: base64 `ImageContent` in the result — decode it into a
  PNG at a sensible path (e.g. `./outputs/YYYYMMDD_prompt.png`).
- **http/sse (remote)**: a short-lived download URL (`.../images/<id>`, ~10
  min) — `curl -sSf <url> -o ./outputs/xxx.png`.

Then report the saved path to the user. Never say a file was written unless
you actually created it.

## Verify before you use

Call the discovery tools first when unsure: `list_loras`, and
`list_control_types` when offered — only supported ids are accepted; URLs and
arbitrary paths are always rejected. If `list_control_types` is not offered,
the active family has no ControlNet and `edit_image` has no `control_type`
parameter — do not pass one.

## generate_image

```
generate_image(
  prompt,                       # required; English works best
  negative_prompt = "",         # "blurry, low quality" etc.
  width = 0, height = 0,        # 0 = model-family default; snapped to a native bucket
  num_inference_steps = 0,      # 0 = family default (SDXL 30 / SD3.5 40 / Qwen 40)
  guidance_scale = 0,           # 0 = family default (SDXL 7.0 / SD3.5 4.5 / Qwen 1.0)
  seed = -1,                    # -1 = random
  lora = ""                     # "huggingface/repo:weight" (comma-separable)
)
```

## edit_image (img2img)

```
edit_image(
  prompt,                       # required
  image,                        # required: http(s) URL, or file path (stdio/local)
  negative_prompt = "",
  strength = 0.6,               # 0..1; higher = bigger change (SDXL/SD3.5 only)
  width = 0, height = 0,        # 0 = keep source size; else snapped to a native bucket
  num_inference_steps = 0,      # 0 = family default
  guidance_scale = 0,           # 0 = family default (Qwen: >1 + negative = CFG)
  seed = -1,
  lora = "",
  control_type = "",            # "canny" etc.; only exists for ControlNet families
  control_scale = 1.0           # ~0.4-1.0 (same condition as control_type)
)
```

> The `qwen-image-2.1` family: `edit_image` omits `strength`,
> `control_type` and `control_scale` entirely — the edit always runs the
> full step count on the source-conditioned pipeline.

**`image` — how to point at the source image:**

- **A server image URL** `http://<host>/images/<id>` from `generate_image` /
  `edit_image` results or a `POST /images/upload` response.
- **An external `http(s)://` URL** (fetched server-side; private/loopback
  refused).
- **A host file path** or `file://` URI — local stdio runs only.
- **Uploading a local file over http/sse**: call `upload_image`, then POST
  the bytes with the returned one-time `X-UPLOAD-TOKEN`; pass the resulting
  `/images/<id>` URL to `edit_image`.

To chain an edit onto a generation: use the download URL from the
`generate_image` result, or save the image locally and pass its path (stdio).
The tool schema shows which forms apply to the current run.

## Prompting conventions

- Be specific: subject + style + lighting/composition.
- Use `negative_prompt` for common artifacts ("blurry, low quality,
  deformed").
- LoRA ids look like `org/repo` and take an optional weight (`:0.8`); check
  `list_loras` first.

## Model families

| Family | txt2img | img2img | LoRA | ControlNet | Defaults |
|---|---|---|---|---|---|
| `sdxl` | ✓ | ✓ | ✓ | ✓ (canny/depth/openpose) | 1024², 30 steps, guidance 7.0 |
| `sd35-medium` / `sd35-large` | ✓ | ✓ | ✓ | ✓ (canny/depth) | 1024², 40 steps, guidance 4.5 |
| `qwen-image-2.1` | ✓ | ✓ (unified pipeline) | ✓ | — | 1024², 40 steps, guidance 1.0 (pass `>1` + negative prompt for CFG) |

Unsupported families are rejected at startup.

## Notes

- First calls load the local models; expect the first generation to be slow.
- Sizes snap to the nearest native training bucket of the active family —
  matching a bucket keeps quality; off-bucket sizes (e.g. 512×512) cause
  tiled/duplicated patterns, and `width=0`/`height=0` snaps the source size
  too (the output aspect can differ slightly from a non-bucket source).
- **Privacy**: never write the user's prompt into log files, notes, or other
  persistent text.
- On errors, report the returned error text verbatim to the user.
