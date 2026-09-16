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
Diffusion, SDXL by default). The images come back to you inline; the server
never writes files, so **you save them yourself**.

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

The server is stateless: every tool returns the image as inline base64 and
**nothing is written to any disk**. You (the agent) are responsible for saving:

1. Capture the returned image data.
2. Write it to a sensible path (e.g. `./outputs/YYYYMMDD_prompt.png`); create the directory first if needed.
3. Report the saved path to the user. If you cannot save, tell the user the image is only available inline.

## Verify before you use

When unsure which LoRA/ControlNet values are valid, **call the discovery tools**
first (`list_loras`, `list_control_types`) — only allowlisted ids are accepted,
and arbitrary URLs or file paths are rejected.

## generate_image

```
generate_image(
  prompt,                       # required; English works best
  negative_prompt = "",         # "blurry, low quality" etc.
  width = 1024, height = 1024,  # SDXL default; clamps to [256,1024], multiple of 8
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
  image,                        # required: file path, file://, or data:image URI
  negative_prompt = "",
  strength = 0.6,               # 0..1; higher = bigger change
  width = 0, height = 0,        # 0 = keep source size, clamp ≤1024
  num_inference_steps = 25,
  guidance_scale = 7.5,
  seed = -1,
  lora = "",
  control_type = "",            # e.g. "canny"; abstract, discover via list_control_types
  control_scale = 1.0           # ~0.4-1.0
)
```

`control_type` preprocesses the source image internally (e.g. canny/depth/pose)
and steers the edit; it is **SDXL-only**.

## Prompting conventions

- Be specific: subject + style + lighting/composition (e.g.
  "a fluffy corgi in a spacesuit, neon synthwave, dramatic lighting").
- Use `negative_prompt` for common artifacts ("blurry, low quality, deformed").
- LoRA ids look like `org/repo` and take an optional weight (`:0.8`); verify
  with `list_loras`.

## Model-family constraints

- **SDXL** is first-class: txt2img, img2img, LoRA, ControlNet all supported.
- **SD 1.5** (`IMAGE_MODEL=stable-diffusion-v1-5/*`): txt2img + img2img + LoRA
  (with allowlist override); `control_type` returns a clear error (SDXL-only).
- **FLUX** (`*flux*`): txt2img only.

## Notes

- First calls may download/load models and prefetch; expect them to be slow.
- Sizes must be multiples of 8 within [256, 1024].
- **Privacy**: never write the user's prompt into log files, notes, or other
  persistent text.
- On errors, report the returned error text verbatim to the user.
