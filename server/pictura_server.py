"""
Image generation MCP server (MCP 2.x).

Runs a Stable Diffusion pipeline (Hugging Face `diffusers`) and exposes it as
MCP tools over stdio. Uses the local GPU (CUDA) when available.

Configuration (environment variables):
    IMAGE_MODEL       Hugging Face model id (default: stabilityai/stable-diffusion-xl-base-1.0)
                      Other options: stable-diffusion-v1-5/stable-diffusion-v1-5 (SD1.5),
                                     stabilityai/sd-turbo, black-forest-labs/FLUX.1-schnell
    IMAGE_VAE         Optional VAE model id to attach (e.g. for SDXL fp16 fixes)
    IMAGE_DEVICE      cuda | cpu (default: cuda if available else cpu)
    IMAGE_CUDA_DEVICE          restrict CUDA GPUs, e.g. "0" or "0,1" (=> CUDA_VISIBLE_DEVICES)
    IMAGE_MODEL_CACHE_DIR        model download/cache directory (default: HF cache)
    IMAGE_LORA_ALLOWLIST          override LoRA allowlist (comma-separated; "*" = any)
    IMAGE_CONTROLNET_ALLOWLIST    override ControlNet allowlist (comma-separated; "*" = any)
    IMAGE_SKIP_PREFETCH=1         skip pre-downloading allowlisted models at startup
    IMAGE_HOST                    bind address for http/sse (default 127.0.0.1)
    IMAGE_PORT                    TCP port for http/sse (default 8000)
    IMAGE_LOG_FILE                append [pictura-mcp] logs to this file (default: stderr)

Client-supplied `lora` ids are restricted to a built-in default allowlist
(override via IMAGE_LORA_ALLOWLIST); URLs/local paths are rejected and weights
load safetensors-only. ControlNet is exposed as an abstract `control_type`
(e.g. 'canny'); the backing model/preprocessor stay server-side and are picked
from an internal allowlist (IMAGE_CONTROLNET_ALLOWLIST). Allowlisted models are
pre-downloaded into the cache at service startup.

Generated images are never written to disk on the server: they are always
returned inline as base64 and the client (e.g. the coding agent) is responsible
for saving them — local and remote operation is identical.

Transports / remote access (CLI):
    --transport stdio|http|sse   default stdio (spawned by the MCP client)
    --host <host>                bind address for http/sse (default: $IMAGE_HOST or 127.0.0.1)
    --port <port>                TCP port for http/sse (default: $IMAGE_PORT or 8000)
    --max-body-mb <MB>           max HTTP request body for http/sse (default 16)
                                 Img2img base64 image input is sent in the body;
                                 16 MB body ≈ 12 MB image, ample for typical
                                 ~1.6 MB outputs and camera JPEGs. Raise only if
                                 you really need to pass very large images.
    --token <token>              require "Authorization: Bearer <token>" on http/sse
    --log-file <path>            append [pictura-mcp] logs to a file (default: stderr)

Run:
    ./.venv/bin/python server/pictura_server.py                     # stdio MCP server
    ./.venv/bin/python server/pictura_server.py --transport http \
        --host 0.0.0.0 --port 8000 --token sekrit               # remote HTTP server
    ./.venv/bin/python server/pictura_server.py --smoke            # self-test (no MCP)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer, Context
from mcp.types import ImageContent, TextContent
from pydantic import Field

# Keep MCP stdio clean: quiet the noisy HTTP/cache logging.
# Also cap framework loggers so tool arguments are never emitted at INFO/DEBUG.
for _name in ("httpx", "huggingface_hub", "mcp", "uvicorn", "starlette", "asyncio"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# PRIVACY: never include user prompts or tool arguments in logs.
# _log(msg) must only ever receive server-internal status text (no request data).
# See README/SPEC for the privacy guarantee.

# ---- log destination ------------------------------------------------------
# IMAGE_LOG_FILE / --log-file redirect the [pictura-mcp] log to a file (append),
# reopened on SIGHUP so logrotate (postrotate kill -HUP) keeps working.
_LOG_FH = None
_LOG_PATH: str | None = None


def _set_log_file(path: str | None) -> None:
    global _LOG_FH, _LOG_PATH
    _LOG_PATH = path
    if not path:
        return
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        if _LOG_FH is not None:
            try:
                _LOG_FH.close()
            except OSError:
                pass
        # owner rw / group r (readers join the service group); others none.
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        _LOG_FH = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        _log(f"log file: {p} (0640, group r)")
    except OSError as e:
        _LOG_FH = None
        print(f"[pictura-mcp] cannot open log file {path}: {e}", file=sys.stderr, flush=True)


def _reopen_log() -> None:
    """Reopen the log file (SIGHUP handler) after logrotate moved it."""
    if _LOG_PATH:
        _set_log_file(_LOG_PATH)


def _log(msg: str) -> None:
    # Progress/debug. Writes to the configured log file, else stderr (never to
    # the MCP stdio channel, which is reserved for protocol messages).
    line = f"[pictura-mcp] {msg}"
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            return
        except OSError:
            pass
    print(line, file=sys.stderr, flush=True)


if os.environ.get("IMAGE_LOG_FILE"):
    _set_log_file(os.environ["IMAGE_LOG_FILE"])

# Reduce CUDA allocator fragmentation (helps under CPU offload / ControlNet).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_ID = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_ID: str | None = os.environ.get("IMAGE_VAE") or None
DEVICE = os.environ.get("IMAGE_DEVICE", "cuda").lower()

# Restrict which CUDA GPU(s) are used, e.g. IMAGE_CUDA_DEVICE=0 or 0,1.
# Applied via the standard CUDA_VISIBLE_DEVICES before torch initializes CUDA.
_cuda_visible = os.environ.get("IMAGE_CUDA_DEVICE")
if _cuda_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible
    _log(f"CUDA_VISIBLE_DEVICES -> {_cuda_visible}")

# Optional model download/cache directory (overrides the default HF cache).
CACHE_DIR: str | None = os.environ.get("IMAGE_MODEL_CACHE_DIR") or None
if CACHE_DIR:
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(CACHE_DIR) / "hub"))

# Default allowlists for client-supplied LoRA / ControlNet ids.
# Override with IMAGE_LORA_ALLOWLIST / IMAGE_CONTROLNET_ALLOWLIST
# (comma-separated; "*" = allow any bare 'org/repo' id). ControlNet ids are
# internal (hidden from clients) - clients only see abstract control types.
DEFAULT_LORA_ALLOWLIST = ["nerijs/pixel-art-xl", "CiroN2022/toy-face"]
DEFAULT_CONTROLNET_ALLOWLIST = [
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0",
    "xinsir/controlnet-openpose-sdxl-1.0",
]

# Abstract control types -> internal preprocessor + ControlNet model.
# Clients pass only the *type* (e.g. "canny"); the model id stays server-side.
_CONTROL_TYPES: dict[str, dict] = {
    "canny": {
        "desc": "Canny edge lines (keeps linear structure / line art)",
        "pre": "opencv_canny",
        "model": "diffusers/controlnet-canny-sdxl-1.0",
        "prep_model": None,
        "supported": True,
    },
    "depth": {
        "desc": "Depth map (spatial layout)",
        "pre": "dpt",
        "model": "diffusers/controlnet-depth-sdxl-1.0",
        "prep_model": "Intel/dpt-hybrid-midas",
        "supported": True,
    },
    "openpose": {
        "desc": "Skeleton / pose",
        "pre": "yolo_pose",
        "model": "xinsir/controlnet-openpose-sdxl-1.0",
        "prep_model": "yolov8n-pose",
        "supported": True,
    },
}

# Model-aware default resolution / steps.
IS_XL = "xl" in MODEL_ID.lower()
STEPS_DEFAULT = 30
W_DEFAULT, H_DEFAULT = (1024, 1024) if IS_XL else (512, 512)

_pipe = None
_pipe_info: dict = {}


# --------------------------------------------------------------------------
# Pipeline loading (lazy, once)
# --------------------------------------------------------------------------

def _load_pipeline():
    """Load (or reuse) the diffusion pipeline with memory optimizations for
    a single consumer GPU (works on small-VRAM cards too)."""
    global _pipe, _pipe_info
    if _pipe is not None:
        return _pipe, _pipe_info

    import torch
    from diffusers import (
        AutoencoderKL,
        DiffusionPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    device = DEVICE if torch.cuda.is_available() else "cpu"
    if DEVICE != "cpu" and not torch.cuda.is_available():
        _log(f"CUDA not available, falling back to CPU (device={device})")

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs: dict = {"dtype": dtype}

    # Some models must be loaded with a specific pipeline class / extra bits.
    model_lower = MODEL_ID.lower()
    loader = None
    if "xl" in model_lower:
        loader = StableDiffusionXLPipeline
    elif "flux" in model_lower:
        from diffusers import FluxPipeline

        loader = FluxPipeline
        model_kwargs["dtype"] = dtype
    else:
        loader = StableDiffusionPipeline

    _log(f"Loading model {MODEL_ID} ... (this may take a while on first run)")
    t0 = time.time()

    try:
        if "xl" in model_lower:
            if VAE_ID:
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(VAE_ID, dtype=dtype)
                _pipe = loader.from_pretrained(MODEL_ID, vae=vae, **model_kwargs)
            else:
                _pipe = loader.from_pretrained(MODEL_ID, **model_kwargs)
        else:
            _pipe = loader.from_pretrained(MODEL_ID, **model_kwargs)
    except Exception:
        # Retry with lower precision headroom (some repos are fp32-safetensors only).
        _log("Initial load failed, retrying with default precision...")
        model_kwargs.pop("dtype", None)
        model_kwargs.pop("vae", None)
        _pipe = loader.from_pretrained(MODEL_ID, **model_kwargs)

    # ---- memory optimizations for low-VRAM cards -----------------------------
    if "flux" not in model_lower:
        try:
            _pipe.enable_attention_slicing()
        except Exception as e:  # noqa: BLE001
            _log(f"attention slicing skipped: {e}")
        try:
            _pipe.enable_vae_slicing()
        except Exception:  # noqa: BLE001
            pass
        try:
            _pipe.enable_vae_tiling()
        except Exception:  # noqa: BLE001
            pass

    # ---- fp16 VAE NaN / black-image guard for SDXL ---------------------------
    # Done BEFORE device placement so the swapped VAE is included by the
    # offload hooks or the .to(device) move below.
    if "xl" in model_lower and dtype == torch.float16:
        if not VAE_ID:
            # Use the fp16-safe VAE (works in fp16 directly, no fp32 upcast or
            # vae-tiling precision conflicts).
            from diffusers import AutoencoderKL

            _log("SDXL: using fp16-safe VAE madebyollin/sdxl-vae-fp16-fix")
            _pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=dtype)
        else:
            # User-explicit VAE: keep decode in fp32 to avoid NaN/black output.
            try:
                _pipe.upcast_vae()
                _log("SDXL: upcast_vae() enabled to avoid fp16 black-image artifacts")
            except Exception as e:  # noqa: BLE001
                _log(f"SDXL upcast_vae skipped: {e}")

    offloaded = False
    weights_gb = 0.0
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        weights_gb = _estimate_weights_bytes(_pipe) / 1e9
        # SDXL in fp16 OOMs around ~768px+ with weights resident on low-VRAM
        # cards, so spill
        # to CPU up front when the default resolution needs the headroom.
        proactive_offload = IS_XL and (W_DEFAULT >= 768 or H_DEFAULT >= 768)
        if proactive_offload or weights_gb > 0.8 * vram_gb:
            _log(
                f"(proactive_offload={proactive_offload}, weights {weights_gb:.1f}/"
                f"{vram_gb:.1f} GB) -> enabling model CPU offload"
            )
            try:
                _pipe.enable_model_cpu_offload()
                offloaded = True
            except Exception as e:  # noqa: BLE001
                _log(f"model cpu offload skipped: {e} -> .to({device})")
                _pipe.to(device)
        else:
            _log(f"weights {weights_gb:.1f} GB fit in VRAM {vram_gb:.1f} GB -> keeping on GPU")
            _pipe.to(device)
    else:
        _pipe.to(device)

    _pipe.safety_checker = None
    try:
        _pipe._safety_checker = None  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass

    _pipe_info = {
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "offload": offloaded,
        "weights_gb": round(weights_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "load_seconds": round(time.time() - t0, 1),
    }
    _log(f"Model ready ({_pipe_info})")
    return _pipe, _pipe_info


def _estimate_weights_bytes(pipe) -> int:
    """Rough size in bytes of the model weights currently held in RAM."""
    total = 0
    for component in getattr(pipe, "components", {}).values():
        if hasattr(component, "parameters"):
            for p in component.parameters():
                total += p.numel() * p.element_size()
    if total == 0:  # fallback: iterate modules
        for module in pipe.modules():
            for p in module.parameters(recurse=False):
                total += p.numel() * p.element_size()
    return total


def _generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    on_step=None,
    lora_spec: str = "",
):
    import torch

    pipe, _ = _load_pipeline()
    _apply_loras(pipe, lora_spec)

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    common = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    if pipe.__class__.__name__ in ("FluxPipeline", "FluxPriorityPipeline"):
        # FLUX does not use a negative prompt.
        pass
    else:
        common["negative_prompt"] = negative_prompt

    is_flux = pipe.__class__.__name__ in ("FluxPipeline", "FluxPriorityPipeline")
    if is_flux:
        # FLUX does not use a negative prompt and has no per-step callback.
        images = pipe(**common).images
    else:

        def _cb(_pipe, step: int, _ts, _kwargs):
            if on_step:
                on_step(step)
            return _kwargs

        images = _oom_retry(
            pipe,
            **common,
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=[],
        ).images
    return images[0], seed


# --------------------------------------------------------------------------
# img2img
# --------------------------------------------------------------------------

_i2i_pipe = None


def _load_i2i_pipeline():
    """Build an img2img pipeline that reuses the loaded txt2img components
    (shared weights -> no extra VRAM for the base model)."""
    global _i2i_pipe
    if _i2i_pipe is not None:
        return _i2i_pipe
    from diffusers import AutoPipelineForImage2Image

    pipe, info = _load_pipeline()
    if pipe.__class__.__name__ in ("FluxPipeline", "FluxPriorityPipeline"):
        raise RuntimeError("img2img is not supported for FLUX in this build")
    _i2i_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    # With model CPU offload the hooks handle placement; otherwise move to GPU.
    if not info.get("offload"):
        _i2i_pipe.to(info.get("device") or DEVICE)
    try:
        _i2i_pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        _i2i_pipe.enable_vae_slicing()
        _i2i_pipe.enable_vae_tiling()
    except Exception:
        pass
    _log("img2img pipeline ready (shared weights)")
    return _i2i_pipe


def _load_source_image(image_src: str):
    """Accept a local file path, a file:// URI, or a data:image/...;base64,... URI."""
    from PIL import Image as PILImage

    src = image_src.strip()
    if src.startswith("data:"):
        _meta, b64 = src.split(",", 1)
        return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if src.startswith("file://"):
        src = src[len("file://"):]
    try:
        return PILImage.open(src).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"image not found: {src} (pass a local file path or a data: URI)"
        )


def _i2i_dims(source_w: int, source_h: int, width: int, height: int):
    """Resolve target dimensions: explicit width/height, else source size,
    clamped to [256, 1024] and rounded to multiples of 8."""
    w = width or source_w
    h = height or source_h
    scale = min(1.0, 1024.0 / max(w, h))
    w = max(256, min(1024, int(round(w * scale / 8) * 8)))
    h = max(256, min(1024, int(round(h * scale / 8) * 8)))
    return w, h


def _oom_retry(pipe, **kwargs):
    """Call the pipeline; on CUDA OOM, spill weights to CPU and retry once."""
    import torch

    try:
        return pipe(**kwargs)
    except torch.cuda.OutOfMemoryError:
        _log("CUDA OOM -> enabling model CPU offload and retrying")
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:  # noqa: BLE001
            _log(f"offload failed ({e}), retrying as-is")
        return pipe(**kwargs)


# --------------------------------------------------------------------------
# LoRA / ControlNet support
# --------------------------------------------------------------------------

import re as _re

_REPO_ID_RE = _re.compile(
    r"^(?!\.)[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*/[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*$"
)


def _allowlist(env: str, default) -> set | None:
    """Resolve an allowlist: env override, else the built-in default.
    "*" disables the allowlist (any bare 'org/repo' id passes syntax check)."""
    raw = os.environ.get(env, "").strip()
    if raw == "*":
        return None
    if raw:
        return {x.strip() for x in raw.split(",") if x.strip()}
    return set(default)


def _check_model_id(kind: str, mid: str, allowlist_env: str, default) -> str:
    """Validate a client-supplied model id.

    Only bare HuggingFace 'org/repo' ids are accepted - URLs, absolute paths,
    relative paths and traversal are rejected (the ids feed from_pretrained /
    load_lora_weights, which would otherwise fetch from arbitrary URLs or load
    arbitrary host paths - SSRF / pickle-deserialization risks).
    The allowlist (env override else the built-in default; "*" = any id) is
    enforced, and weights load safetensors-only.
    """
    mid = mid.strip()
    if not mid:
        raise ValueError(f"{kind}: empty model id")
    allowed = _allowlist(allowlist_env, default)
    if allowed is not None and mid not in allowed:
        raise ValueError(
            f"{kind} '{mid}' is not in the allowlist (see {allowlist_env} / built-in defaults)"
        )
    if not _REPO_ID_RE.match(mid):
        raise ValueError(
            f"{kind} '{mid}' must be a bare 'org/repo' Hugging Face id "
            "(URLs and local file paths are not allowed)"
        )
    return mid


def _prefetch_allowlisted() -> None:
    """Pre-download the allowlisted LoRA / ControlNet repos into the cache dir.

    Called once at service startup so tool calls do not pay the download cost.
    Failures are logged and non-fatal (models can still be fetched lazily).
    """
    from huggingface_hub import snapshot_download

    ids = list(_allowlist("IMAGE_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST) or DEFAULT_LORA_ALLOWLIST)
    ids += [info["model"] for info in _CONTROL_TYPES.values() if info.get("supported")]
    for mid in dict.fromkeys(ids):
        _log(f"prefetching {mid} ...")
        try:
            snapshot_download(
                repo_id=mid,
                cache_dir=CACHE_DIR,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "*.model"],
            )
            _log(f"prefetched {mid}")
        except Exception as e:  # noqa: BLE001
            _log(f"prefetch failed for {mid}: {e}")
    # preprocessor weights for supported control types
    prep = _CONTROL_TYPES["depth"].get("prep_model")
    if prep:
        _log(f"prefetching preprocessor {prep} ...")
        try:
            snapshot_download(repo_id=prep, cache_dir=CACHE_DIR, allow_patterns=["*.bin", "*.json"])
            _log(f"prefetched preprocessor {prep}")
        except Exception as e:  # noqa: BLE001
            _log(f"prefetch failed for preprocessor {prep}: {e}")
    try:
        _download_yolo_pose()
    except Exception as e:  # noqa: BLE001
        _log(f"prefetch failed for yolov8n-pose: {e}")


def _parse_lora(spec: str):
    """Parse 'huggingface/repo:weight,huggingface/repo2:0.5' -> [(id, weight)]."""
    out = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk and chunk.rsplit(":", 1)[1].replace(".", "", 1).lstrip("-").isdigit():
            mid, weight = chunk.rsplit(":", 1)
            out.append((mid.strip(), float(weight)))
        else:
            out.append((chunk, 1.0))
    return out


def _apply_loras(pipe, spec: str) -> None:
    """Load/apply (or clear) LoRA adapters on a pipeline.

    The spec is comma-separated 'huggingface/model:weight' entries (weight
    defaults to 1.0). Adapters are downloaded from the HF cache.
    """
    entries = _parse_lora(spec)
    if not entries:
        try:
            pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001
            pass
        return
    names, weights = [], []
    for i, (mid, weight) in enumerate(entries):
        name = f"lora{i}"
        _check_model_id("LoRA", mid, "IMAGE_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST)
        pipe.load_lora_weights(
            mid, adapter_name=name, use_safetensors=True, cache_dir=CACHE_DIR
        )
        names.append(name)
        weights.append(weight)
    pipe.set_adapters(names, adapter_weights=weights)
    _log(f"LoRA applied: {[(m.rsplit('/', 1)[-1], w) for m, w in entries]}")


def _resolve_control(ctype: str) -> dict:
    """Resolve an abstract control_type to its internal preprocessor/model.
    Model identifiers stay server-side (not exposed to clients)."""
    ctype = (ctype or "").strip().lower()
    if not ctype:
        raise ValueError("control_type is empty")
    if not IS_XL:
        raise ValueError(
            "ControlNet (control_type) is only supported with the SDXL model family "
            "(IMAGE_MODEL must contain 'xl'); remove control_type or switch to an SDXL model"
        )
    info = _CONTROL_TYPES.get(ctype)
    if info is None:
        raise ValueError(f"unknown control_type '{ctype}'; available: {list(_CONTROL_TYPES)}")
    if not info.get("supported"):
        raise ValueError(
            f"control_type '{ctype}' is not yet supported by the installed preprocessor"
        )
    _check_model_id(
        "ControlNet", info["model"], "IMAGE_CONTROLNET_ALLOWLIST", DEFAULT_CONTROLNET_ALLOWLIST
    )
    return info


def _preprocess_control(image, ctype: str):
    """In-memory: create the condition map for a supported control type.
    Returns a PIL image; nothing is written to disk or kept as state."""
    from PIL import Image as PILImage

    if ctype == "canny":
        import cv2
        import numpy as np

        gray = np.asarray(image.convert("L"))
        edges = cv2.Canny(gray, 60, 150)
        return PILImage.fromarray((edges > 0).astype("uint8") * 255).convert("RGB")

    if ctype == "depth":
        return _preprocess_depth(image)

    if ctype == "openpose":
        return _preprocess_openpose(image)

    raise ValueError(f"no preprocessor for control_type '{ctype}'")


_dpt_processor = None
_dpt_model = None


def _preprocess_depth(image):
    """Depth map (near = white) via DPT on the model cache."""
    import torch
    from PIL import Image as PILImage
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    global _dpt_processor, _dpt_model
    mid = _CONTROL_TYPES["depth"]["prep_model"]
    if _dpt_model is None:
        _log(f"Loading depth preprocessor {mid} ...")
        _dpt_processor = DPTImageProcessor.from_pretrained(mid, cache_dir=CACHE_DIR)
        _dpt_model = DPTForDepthEstimation.from_pretrained(mid, cache_dir=CACHE_DIR)
        _dpt_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    enc = _dpt_processor(images=image, return_tensors="pt")
    enc = {k: v.to(_dpt_model.device) for k, v in enc.items()}
    with torch.no_grad():
        depth = _dpt_model(**enc).predicted_depth.float().squeeze(0).cpu()  # (H,W)
    dmin, dmax = depth.min(), depth.max()
    norm = (depth - dmin) / (dmax - dmin + 1e-6)
    gray = (255.0 * (1.0 - norm)).to("cpu").byte().numpy()  # near = white
    return PILImage.fromarray(gray).convert("RGB")


_OPENPOSE_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
_OPENPOSE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 0, 255), (0, 255, 128), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128),
]


def _yolo_pose_path() -> Path:
    base = Path(CACHE_DIR) if CACHE_DIR else Path.home() / ".cache"
    return base / "yolov8n-pose.pt"


def _download_yolo_pose() -> None:
    import urllib.request

    p = _yolo_pose_path()
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
    _log(f"Downloading {p.name} ...")
    urllib.request.urlretrieve(url, p)


_yolo_model = None


def _preprocess_openpose(image):
    """Skeleton / keypoint map via YOLOv8-pose on the model cache."""
    import numpy as np
    from PIL import Image as PILImage

    global _yolo_model
    if _yolo_model is None:
        _download_yolo_pose()
        from ultralytics import YOLO

        _log("Loading openpose preprocessor (yolov8n-pose) ...")
        _yolo_model = YOLO(str(_yolo_pose_path()))
    results = _yolo_model(np.asarray(image.convert("RGB")), verbose=False)
    canvas = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    import cv2

    for res in results:
        kps = res.keypoints
        if kps is None or kps.data is None or len(kps) == 0:
            continue
        pts = kps.data[0].cpu().numpy()  # (17,3) x,y,conf
        for e_idx, (a, b) in enumerate(_OPENPOSE_EDGES):
            if pts[a, 2] > 0.3 and pts[b, 2] > 0.3:
                p1 = (int(pts[a, 0]), int(pts[a, 1]))
                p2 = (int(pts[b, 0]), int(pts[b, 1]))
                cv2.line(canvas, p1, p2, _OPENPOSE_COLORS[e_idx % len(_OPENPOSE_COLORS)], 2)
        for i in range(len(pts)):
            if pts[i, 2] > 0.3:
                cv2.circle(canvas, (int(pts[i, 0]), int(pts[i, 1])), 3, (255, 255, 255), -1)
    return PILImage.fromarray(canvas).convert("RGB")


_cn_model = None
_cn_pipe = None
_cn_info: dict = {}


def _load_controlnet_pipeline(model_id: str):
    """Build a controlnet img2img pipeline (independent, lazily loaded)."""
    global _cn_model, _cn_pipe, _cn_info
    import torch
    from diffusers import ControlNetModel

    if _cn_pipe is not None and _cn_info.get("model") == model_id:
        return _cn_pipe

    device = DEVICE if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _check_model_id("ControlNet", model_id, "IMAGE_CONTROLNET_ALLOWLIST", DEFAULT_CONTROLNET_ALLOWLIST)
    _log(f"Loading ControlNet model {model_id} ...")
    _cn_model = ControlNetModel.from_pretrained(
        model_id, dtype=dtype, use_safetensors=True, cache_dir=CACHE_DIR
    )

    if "xl" in MODEL_ID.lower():
        from diffusers import AutoencoderKL, StableDiffusionXLControlNetImg2ImgPipeline as cn_cls

        vae_kwargs = {
            "vae": AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=dtype)
        }
    else:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline as cn_cls

        vae_kwargs = {}

    _cn_pipe = cn_cls.from_pretrained(MODEL_ID, controlnet=_cn_model, dtype=dtype, **vae_kwargs)
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(_cn_pipe, fn)()
        except Exception:  # noqa: BLE001
            pass

    weights_gb = _estimate_weights_bytes(_cn_pipe) / 1e9
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    proactive = IS_XL and (W_DEFAULT >= 768 or H_DEFAULT >= 768)
    if device == "cuda" and (proactive or weights_gb > 0.8 * vram_gb):
        _cn_pipe.enable_model_cpu_offload()
    else:
        _cn_pipe.to(device)
    _cn_info = {"model": model_id}
    _log(f"ControlNet img2img pipeline ready ({weights_gb:.1f} GB weights)")
    return _cn_pipe


def _image2image(
    prompt: str,
    negative_prompt: str,
    image,
    strength: float,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    on_step=None,
    lora_spec: str = "",
    control_type: str = "",
    control_scale: float = 1.0,
):
    import torch
    from PIL import Image as PILImage

    source = image if isinstance(image, PILImage.Image) else _load_source_image(image)
    w, h = _i2i_dims(*source.size, width, height)
    if (w, h) != source.size:
        source = source.resize((w, h), PILImage.LANCZOS)

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eff_steps = max(1, int(steps * strength))

    def _cb(_p, step, _ts, _kwargs):
        if on_step:
            on_step(min(step, eff_steps))
        return _kwargs

    common = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=[],
    )

    if control_type:
        ctl = _resolve_control(control_type)
        pipe = _load_controlnet_pipeline(ctl["model"])
        _apply_loras(pipe, lora_spec)
        control_img = _preprocess_control(source, control_type)
        control_img = control_img.resize((w, h), PILImage.LANCZOS)
        common["control_image"] = control_img
        common["controlnet_conditioning_scale"] = control_scale
    else:
        pipe = _load_i2i_pipeline()
        _apply_loras(pipe, lora_spec)

    result = _oom_retry(pipe, **common)
    return result.images[0], seed, eff_steps


def _finalize_result(image, actual_seed: int, t0: float, prefix: str = "img") -> tuple:
    """Encode a generated image inline. Returns (b64, elapsed_seconds, name).

    The server is stateless: nothing is written to disk (identical in local and
    remote modes); the client saves the returned base64 where it wants.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    name = f"{prefix}_{int(time.time())}_{actual_seed}.png"
    return b64, round(time.time() - t0, 1), name


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

def _build_server():
    server = MCPServer(
        name="generate-image",
        version="0.1.0",
        title="Image Generation (Stable Diffusion)",
        instructions=(
            "Generate images with a local Stable Diffusion pipeline running on the "
            "host GPU. The tool returns the image plus the saved file path."
        ),
    )

    @server.tool(
        name="generate_image",
        title="Generate Image",
        description=(
            "Generate an image from a text prompt using the local Stable Diffusion "
            "model. Returns the generated image and the saved file path."
        ),
    )
    async def generate_image(
        prompt: str,
        negative_prompt: str = "",
        width: int = W_DEFAULT,
        height: int = H_DEFAULT,
        num_inference_steps: int = STEPS_DEFAULT,
        guidance_scale: float = 7.5,
        seed: int = -1,
        lora: Annotated[
            str,
            Field(
                description=(
                    "Apply LoRA adapter(s): comma-separated 'huggingface/org:weight' "
                    "entries (weight defaults to 1.0). Only allowlisted ids work; "
                    "call list_loras to get valid ids. Example: 'nerijs/pixel-art-xl:0.8'"
                ),
            ),
        ] = "",
        ctx: Context = None,
    ) -> list:
        """Parameters:
        - prompt: what to draw (English works best; be specific).
        - negative_prompt: things to avoid (e.g. "blurry, low quality").
        - width/height: image size in pixels (multiple of 8, 256..1024).
        - num_inference_steps: 20-30 typical for SD1.5, 25-40 for SDXL.
        - guidance_scale: how closely to follow the prompt (1..15, ~7.5 default).
        - seed: fixed seed for reproducibility, -1 = random.
        - lora: apply LoRA adapter(s): 'huggingface/repo:weight' (default 1.0),
          comma-separated; empty = none. Call list_loras for valid ids.
        Note: the server never writes files; the image is returned inline and the
        client saves it where it wants (identical in local and remote modes).
        """
        width = max(256, min(1024, (round(width / 8) * 8)))
        height = max(256, min(1024, (round(height / 8) * 8)))
        steps = max(10, min(100, num_inference_steps))

        import asyncio

        loop = asyncio.get_running_loop()

        def on_step(step: int):
            # Progress is posted from the worker thread back to the event loop.
            if ctx is not None:
                asyncio.run_coroutine_threadsafe(ctx.report_progress(step, steps), loop)

        t0 = time.time()
        try:
            image, actual_seed = await asyncio.to_thread(
                _generate,
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance_scale,
                seed,
                on_step,
                lora,
            )
        except Exception as e:
            return [
                TextContent(type="text", text=f"image generation failed: {e}"),
            ]

        b64, elapsed, name = _finalize_result(image, actual_seed, t0, prefix="img")
        note = (
            f"No file written on server (stateless, same as remote); image returned "
            f"inline - save it client-side (e.g. as {name}). seed={actual_seed}, {elapsed}s"
        )
        return [
            ImageContent(type="image", data=b64, mimeType="image/png"),
            TextContent(type="text", text=note),
        ]

    @server.tool(
        name="edit_image",
        title="Edit Image (img2img)",
        description=(
            "Transform an existing image using a text prompt (img2img). Pass the "
            "source as a local file path (server host), a file:// URI, or a "
            "data:image/...;base64,... URI. Returns the edited image inline "
            "(base64); nothing is written to the server disk."
        ),
    )
    async def edit_image(
        prompt: str,
        image: str,
        negative_prompt: str = "",
        strength: float = 0.6,
        width: int = 0,
        height: int = 0,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = -1,
        lora: Annotated[
            str,
            Field(
                description=(
                    "Apply LoRA adapter(s): comma-separated 'huggingface/org:weight' "
                    "entries (weight defaults to 1.0). Only allowlisted ids work; "
                    "call list_loras to get valid ids."
                ),
            ),
        ] = "",
        control_type: Annotated[
            str,
            Field(
                description=(
                    "ControlNet control type applied to the source image (abstract; "
                    "the server picks and hides the model). Currently supported: "
                    + (", ".join(c for c, i in _CONTROL_TYPES.items() if i.get("supported")) or "-")
                    + ". Do a live lookup via list_control_types (the set can change "
                    "server-side). Empty disables ControlNet."
                ),
            ),
        ] = "",
        control_scale: Annotated[
            float,
            Field(
                description="ControlNet conditioning strength, typical range 0.4-1.0 (default 1.0).",
            ),
        ] = 1.0,
        ctx: Context = None,
    ) -> list:
        """Parameters:
        - prompt: how to transform the image (English works best).
        - image: source image - local file path, file:// URI, or data:image...;base64, URI.
        - negative_prompt: things to avoid.
        - strength: 0..1 how strongly to transform (higher = more change; 0.6 default).
        - width/height: target size in pixels (0 = keep source size, clamped to <=1024).
        - num_inference_steps / guidance_scale / seed: same as generate_image.
        - lora: apply LoRA adapter(s): 'huggingface/repo:weight' (default 1.0),
          comma-separated; empty = none. Call list_loras for valid ids.
        - control_type: ControlNet type applied to the source image (abstract; the
          server picks and hides the model). Currently supported: "
        + (", ".join(c for c, i in _CONTROL_TYPES.items() if i.get("supported")))
        + ". Call list_control_types for the live list (it can change). Empty disables.
        - control_scale: ControlNet conditioning strength (~0.4-1.0).
        Note: the server never writes files; the image is returned inline and the
        client saves it where it wants (identical in local and remote modes).
        """
        steps = max(10, min(100, num_inference_steps))
        strength = max(0.01, min(1.0, strength))
        width = max(0, min(1024, (round(width / 8) * 8) if width else 0))
        height = max(0, min(1024, (round(height / 8) * 8) if height else 0))

        import asyncio

        loop = asyncio.get_running_loop()
        eff_steps = int(steps * strength)

        def on_step(step: int):
            if ctx is not None:
                asyncio.run_coroutine_threadsafe(ctx.report_progress(step, eff_steps), loop)

        t0 = time.time()
        try:
            edited, actual_seed, _eff = await asyncio.to_thread(
                _image2image,
                prompt,
                negative_prompt,
                image,
                strength,
                width,
                height,
                steps,
                guidance_scale,
                seed,
                on_step,
                lora,
                control_type,
                control_scale,
            )
        except Exception as e:
            return [
                TextContent(type="text", text=f"image edit failed: {e}"),
            ]

        b64, elapsed, name = _finalize_result(edited, actual_seed, t0, prefix="img2img")
        note = (
            f"No file written on server (stateless, same as remote); image returned "
            f"inline - save it client-side (e.g. as {name}). "
            f"seed={actual_seed}, strength={strength}, {elapsed}s"
        )
        return [
            ImageContent(type="image", data=b64, mimeType="image/png"),
            TextContent(type="text", text=note),
        ]

    @server.tool(
        name="server_status",
        title="Server Status",
        description="Report the loaded image model, device, dtype and VRAM usage.",
    )
    async def server_status() -> str:
        info = _pipe_info or _load_pipeline()[1]
        return (
            f"model={info.get('model')}\n"
            f"device={info.get('device')}\n"
            f"dtype={info.get('dtype')}\n"
            f"vram_gb={info.get('vram_gb')}\n"
            f"load_seconds={info.get('load_seconds')}"
        )

    @server.tool(
        name="list_loras",
        title="List LoRA ids",
        description=(
            "Return the allowlisted LoRA ids valid for the 'lora' parameter of "
            "generate_image / edit_image."
        ),
    )
    async def list_loras() -> str:
        lora_ids = _allowlist("IMAGE_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST)
        shown = ", ".join(sorted(lora_ids)) if lora_ids else "(any 'org/repo' - allowlist disabled)"
        return (
            "LoRA ids - pass to 'lora' as 'org/repo:weight' (weight defaults to 1.0):\n"
            + "  " + shown
        )

    @server.tool(
        name="list_control_types",
        title="List ControlNet Types",
        description=(
            "Return the abstract ControlNet control types valid for the "
            "'control_type' parameter of edit_image (no model identifiers exposed)."
        ),
    )
    async def list_control_types() -> str:
        lines = []
        for ctype, info in _CONTROL_TYPES.items():
            note = info["desc"] if info.get("supported") else "(not supported yet)"
            lines.append(f"- {ctype}: {note}")
        return "Control types for edit_image 'control_type':\n" + "\n".join(lines)

    return server


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _smoke_test() -> int:
    """Generate one tiny image directly (no MCP) to verify the pipeline works."""
    import logging

    logging.basicConfig(level=logging.INFO)
    smoke_dir = Path(__file__).resolve().parent.parent / "outputs"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Smoke test: model={MODEL_ID}, device={DEVICE}")
    try:
        image, seed = _generate(
            prompt="a red apple on a wooden table, studio lighting",
            negative_prompt="blurry, low quality",
            width=256,
            height=256,
            steps=2,
            guidance=7.5,
            seed=12345,
        )
        out = smoke_dir / "smoke_test.png"
        image.save(out)
        _log(f"Smoke test OK: saved {out} (seed={seed}), info={_pipe_info}")

        # img2img check: reuse the generated image as a source.
        try:
            edited, seed2, _ = _image2image(
                prompt="turn it into an oil painting, warm colors",
                negative_prompt="blurry",
                image=str(out),
                strength=0.5,
                width=256,
                height=256,
                steps=2,
                guidance=7.5,
                seed=999,
            )
            out2 = smoke_dir / "smoke_test_img2img.png"
            edited.save(out2)
            _log(f"Smoke test img2img OK: saved {out2} (seed={seed2})")
        except Exception as e:  # noqa: BLE001
            _log(f"Smoke test img2img FAILED: {type(e).__name__}: {e}")
            return 1
        return 0
    except Exception as e:  # noqa: BLE001
        _log(f"Smoke test FAILED: {type(e).__name__}: {e}")
        return 1


def main() -> int:  # noqa: C901
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Image generation MCP server")
    parser.add_argument("--smoke", action="store_true", help="self-test without MCP")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default=None, help="bind address for http/sse"
                        " (default: $IMAGE_HOST or 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port for http/sse (default: $IMAGE_PORT or 8000)",
    )
    parser.add_argument(
        "--max-body-mb",
        type=int,
        default=int(os.environ.get("IMAGE_MAX_BODY_MB", "16")),
        help="max HTTP request body size in MB for http/sse (default 16)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="require Bearer token on http/sse (falls back to $PICTURA_MCP_TOKEN)",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("IMAGE_LOG_FILE"),
        help="append [pictura-mcp] logs to a file (default: stderr / journald)",
    )
    args = parser.parse_args()
    token = args.token or os.environ.get("PICTURA_MCP_TOKEN")
    http_port = args.port or int(os.environ.get("IMAGE_PORT", "8000"))
    http_host = args.host or os.environ.get("IMAGE_HOST", "127.0.0.1")
    _set_log_file(args.log_file)
    # logrotate support: reopen the configured log file on SIGHUP.
    try:
        import signal

        signal.signal(signal.SIGHUP, lambda *_: _reopen_log())
    except (AttributeError, ValueError):  # noqa: BLE001 - no SIGHUP / not main thread
        pass

    if args.smoke:
        return _smoke_test()

    server = _build_server()
    # Warm the model cache: pre-download allowlisted LoRA / ControlNet repos.
    if not os.environ.get("IMAGE_SKIP_PREFETCH") == "1":
        _prefetch_allowlisted()
    try:
        if args.transport == "stdio":
            asyncio.run(server.run_stdio_async())
        else:
            return _run_http_server(
                server,
                args.transport,
                http_host,
                http_port,
                token,
                max_body_bytes=args.max_body_mb * 1024 * 1024,
            )
    except KeyboardInterrupt:
        return 0
    return 0


def _run_http_server(
    server,
    transport: str,
    host: str,
    port: int,
    token: str | None,
    max_body_bytes: int = 16 * 1024 * 1024,
) -> int:
    """Serve the MCP server over HTTP(S) so remote clients can connect."""
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, Response

    if transport == "sse":
        app = server.sse_app(
            host=host, max_request_body_size=max_body_bytes
        )
        path = "/sse"
    else:
        app = server.streamable_http_app(
            host=host, json_response=True, max_request_body_size=max_body_bytes
        )
        path = "/mcp"

    if token:
        async def _require_token(request, call_next):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=_require_token)
        _log(
            f"MCP {transport} server: http://{host}:{port}{path} "
            f"(auth: Bearer token, max body {max_body_bytes // (1024 * 1024)} MB)"
        )
    else:
        _log(
            f"MCP {transport} server: http://{host}:{port}{path} "
            f"(NO auth - only for trusted LAN, max body {max_body_bytes // (1024 * 1024)} MB)"
        )

    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
