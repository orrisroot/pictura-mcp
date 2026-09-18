"""
Image generation MCP server (MCP 2.x).

Runs an SDXL (Stable Diffusion XL) pipeline (Hugging Face `diffusers`) and
exposes it as MCP tools over stdio. Uses the local GPU (CUDA) when available.
SDXL is the supported model family: any SDXL checkpoint (including base or
finetune variants) can be selected via `PICTURA_MODEL`.

Configuration (environment variables):
    PICTURA_MODEL       Hugging Face model id, SDXL family
                      (default: stabilityai/stable-diffusion-xl-base-1.0;
                      any SDXL checkpoint/finetune id works)
    PICTURA_VAE         Optional VAE model id to attach (e.g. for SDXL fp16 fixes)
    PICTURA_DEVICE      cuda | cpu (default: cuda if available else cpu)
    PICTURA_CUDA_DEVICE          restrict CUDA GPUs, e.g. "0" or "0,1" (=> CUDA_VISIBLE_DEVICES)
    PICTURA_MODEL_CACHE_DIR        model download/cache directory (default: HF cache)
    PICTURA_LORA_ALLOWLIST          override LoRA allowlist (comma-separated; "*" = any)
    PICTURA_CONTROLNET_ALLOWLIST    override ControlNet allowlist (comma-separated; "*" = any)
    PICTURA_SKIP_PREFETCH=1         skip pre-downloading allowlisted models at startup
    PICTURA_MAX_CONCURRENT          concurrent rendering slots (default "auto": sized
                                  from measured free VRAM; integer pins the pool;
                                  1 = strictly serial). Each slot is an independent
                                  pipeline instance (own LoRA/sched/offload state).
    PICTURA_HOST                    bind address for http/sse (default 127.0.0.1)
    PICTURA_PORT                    TCP port for http/sse (default 8000)
    PICTURA_LOG_FILE                append [pictura-mcp] logs to a file (default: stderr)

URL image return (http/sse only):
    PICTURA_PUBLIC_URL              externally visible base URL; default = the
                                  request's Host header (reverse-proxy friendly)
    PICTURA_FORWARDED_ALLOW_IPS     uvicorn forwarded-allow-ips (default 127.0.0.1)
    PICTURA_IMAGE_URL_TTL            short-term cache lifetime for image URLs, seconds
                                  (default 600)
    PICTURA_IMAGE_URL_MAX            max cached images (default 64)
    PICTURA_IMAGE_URL_MAX_MB         max total cache bytes (default 512)

Client-supplied `lora` ids are restricted to a built-in default allowlist
(override via PICTURA_LORA_ALLOWLIST); URLs/local paths are rejected and weights
load safetensors-only. ControlNet is exposed as an abstract `control_type`
(e.g. 'canny'); the backing model/preprocessor stay server-side and are picked
from an internal allowlist (PICTURA_CONTROLNET_ALLOWLIST). Allowlisted models are
pre-downloaded into the cache at service startup.

`edit_image` reads host image paths only on a local stdio run; over http/sse it
accepts only in-memory data:image/...;base64,... URIs, so the server never
touches the remote host's filesystem.

Images are never written to disk. On stdio the result is returned inline as
base64 (ImageContent); over http/sse the result is a short-lived download URL
(system the image sits in an in-memory cache, TTL-bound, and the client saves
it — server and client may not share a filesystem).

Transports / remote access (CLI):
    --transport stdio|http|sse   default stdio (spawned by the MCP client)
    --host <host>                bind address for http/sse (default: $PICTURA_HOST or 127.0.0.1)
    --port <port>                TCP port for http/sse (default: $PICTURA_PORT or 8000)
    --max-body-mb <MB>           max HTTP request body for http/sse (default 16)
                                 Img2img base64 image input is sent in the body;
                                 16 MB body ≈ 12 MB image, ample for typical
                                 ~1.6 MB outputs and camera JPEGs. Raise only if
                                 you really need to pass very large images.
    --api-key <key>              require the API key on http/sse: clients send it via
                                 the PICTURA_API_KEY header
    --log-file <path>            append [pictura-mcp] logs to a file (default: stderr)

Run:
    ./.venv/bin/python server/pictura_server.py                     # stdio MCP server
    ./.venv/bin/python server/pictura_server.py --transport http \
        --host 0.0.0.0 --port 8000 --api-key sekrit            # remote HTTP server
    ./.venv/bin/python server/pictura_server.py --smoke            # self-test (no MCP)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import secrets
import sys
import threading
import time
from contextlib import contextmanager
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
# PICTURA_LOG_FILE / --log-file redirect the [pictura-mcp] log to a file (append),
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


def _env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable (PICTURA_* namespace)."""
    return os.environ.get(key, default)


if _env("PICTURA_LOG_FILE"):
    _set_log_file(_env("PICTURA_LOG_FILE"))

# Reduce CUDA allocator fragmentation (helps under CPU offload / ControlNet).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_ID = _env("PICTURA_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_ID: str | None = _env("PICTURA_VAE") or None
DEVICE = (_env("PICTURA_DEVICE", "cuda") or "cuda").lower()

# Restrict which CUDA GPU(s) are used, e.g. PICTURA_CUDA_DEVICE=0 or 0,1.
# Applied via the standard CUDA_VISIBLE_DEVICES before torch initializes CUDA.
_cuda_visible = _env("PICTURA_CUDA_DEVICE")
if _cuda_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible
    _log(f"CUDA_VISIBLE_DEVICES -> {_cuda_visible}")

# Optional model download/cache directory (overrides the default HF cache).
CACHE_DIR: str | None = _env("PICTURA_MODEL_CACHE_DIR") or None
if CACHE_DIR:
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(CACHE_DIR) / "hub"))

# --------------------------------------------------------------------------
# URL image return + short-term in-memory cache (http/sse only)
# --------------------------------------------------------------------------
# Over http/sse each generated image is stored in an in-memory cache for a
# short TTL and the tool result returns an unguessable download URL (server
# and client may not share a filesystem). Images are never written to disk:
# the cache lives only in RAM and vanishes with the process.
_URL_TTL = float(_env("PICTURA_IMAGE_URL_TTL", "600") or 600)
_URL_MAX = max(1, int(_env("PICTURA_IMAGE_URL_MAX", "64") or 64))
_URL_MAX_BYTES = max(
    1, int(_env("PICTURA_IMAGE_URL_MAX_MB", "512") or 512)
)* 1024 * 1024
# Explicit public base for image URLs. If unset, the base is derived from the
# incoming request (reverse proxy Host / X-Forwarded-Proto), falling back to
# the bind address.
_PUBLIC_BASE = (_env("PICTURA_PUBLIC_URL") or "").strip().rstrip("/") or None
# uvicorn --forwarded-allow-ips: peers whose X-Forwarded-* headers are trusted
# (reverse proxies on the same host are trusted by default).
_FWD_ALLOW_IPS = (_env("PICTURA_FORWARDED_ALLOW_IPS") or "127.0.0.1").strip()


class _ImageCache:
    """In-memory short-term store for images served back over HTTP.

    Thread-safe; evicts expired entries lazily and enforces the entry count /
    total-bytes caps (oldest first) so a burst of generations cannot exhaust
    RAM. Ids are 192-bit unguessable secrets (the URL itself is the ticket).
    """

    def __init__(self, max_entries: int, max_total_bytes: int, ttl: float):
        self._max = max_entries
        self._max_bytes = max_total_bytes
        self._ttl = ttl
        self._store: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def put(self, data: bytes) -> str | None:
        """Cache image bytes; return an unguessable id (None if it does not fit)."""
        with self._lock:
            now = time.monotonic()
            for k, (exp, _) in list(self._store.items()):
                if exp <= now:
                    del self._store[k]
            while self._store and (
                len(self._store) >= self._max
                or sum(len(v) for _, v in self._store.values()) + len(data) > self._max_bytes
            ):
                # oldest insertion first (dict preserves insertion order)
                self._store.pop(next(iter(self._store)))
            img_id = secrets.token_urlsafe(24)
            self._store[img_id] = (now + self._ttl, data)
            return img_id

    def get(self, img_id: str) -> bytes | None:
        """Return cached bytes for an id, or None when missing/expired."""
        with self._lock:
            e = self._store.get(img_id)
            if e is None:
                return None
            exp, data = e
            if exp <= time.monotonic():
                del self._store[img_id]
                return None
            return data

    @property
    def ttl(self) -> float:
        return self._ttl


_img_cache = _ImageCache(_URL_MAX, _URL_MAX_BYTES, _URL_TTL)
# Image return mode: "url" for http/sse (result is a short-lived download URL),
# "inline" for stdio (base64 ImageContent). Set in main() from the transport.
_RETURN_MODE = "inline"
# Startup fallback base URL (from the bind address); per-request Host wins.
_URL_BASE: str | None = None


def _bind_base(host: str, port: int) -> str | None:
    """Fallback image-URL base from the bind address (no request host yet).

    Returns None for 0.0.0.0, where the externally visible address is unknown;
    there the URL base is derived from the request Host (or PICTURA_PUBLIC_URL).
    """
    if host in ("0.0.0.0", "::", ""):
        return None
    return f"http://{host}:{port}"

# Default allowlists for client-supplied LoRA / ControlNet ids.
# Override with PICTURA_LORA_ALLOWLIST / PICTURA_CONTROLNET_ALLOWLIST
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

IS_XL = "xl" in MODEL_ID.lower()  # the server requires an SDXL model (checked in main)

# Default resolution / steps (SDXL).
STEPS_DEFAULT = 30
W_DEFAULT, H_DEFAULT = 1024, 1024

# --------------------------------------------------------------------------
# Slot pool: concurrent rendering on one GPU
# --------------------------------------------------------------------------

# Incoming MCP requests are always handled concurrently (the event loop never
# blocks on GPU work). Rendering itself uses a pool of "slots": each slot owns
# INDEPENDENT pipeline instances (txt2img / img2img / ControlNet) so LoRA
# adapters, scheduler state and CPU-offload hooks never collide between
# concurrent jobs (weights are therefore duplicated per slot).
#
# The pool size is derived from measured free VRAM once the first model is
# loaded: per extra slot the cost is a full weight set plus the activation
# footprint of one job at the default resolution, and a fixed margin stays
# free for decode spikes / other tenants. On a 32 GB V100 with SDXL fp16
# (~7 GB weights) this yields 3 concurrent slots; on smaller cards it simply
# degrades to 1 (= strictly serial).

_SLOT_RESERVE_GB = 2.0        # VRAM margin that is never handed to extra slots
_SLOT_ACT_BASE_GB = 1.5       # per-job activation floor (fp16)
_SLOT_ACT_PER_MPX_GB = 1.2    # + per megapixel of the default resolution
_SLOT_CAP = 3                 # safety cap in "auto" mode

_slots: list = []             # slots in the pool (slot 0 first, built lazily)
_BUILD_LOCK = threading.Lock()  # serialize pipeline construction (RAM / disk cache)
_rr_next = 0                  # rotating acquisition start index


class _Slot:
    """One rendering slot; its lock is held for the duration of a job."""

    def __init__(self):
        self.lock = threading.Lock()
        self.txt = None       # txt2img pipeline instance
        self.i2i = None       # img2img pipeline instance (built from slot.txt)
        self.cn = {}          # controlnet model id -> pipeline instance
        self.info = {}        # load info of the txt pipeline (for reporting)


def _compute_slots(info: dict) -> int:
    """Decide the pool size from measured free VRAM (or honor
    PICTURA_MAX_CONCURRENT: an integer pins the pool size)."""
    raw = (_env("PICTURA_MAX_CONCURRENT") or "auto").strip().lower()
    if raw and raw != "auto":
        try:
            return max(1, int(raw))
        except ValueError:
            _log(f"PICTURA_MAX_CONCURRENT={raw!r} ignored (not an int); using auto")
    if info.get("device") != "cuda" or info.get("vram_gb", 0) <= 0:
        return 1  # CPU: parallel instances only double RAM without real speedup
    try:
        import torch

        free_gb = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:  # noqa: BLE001
        return 1
    pixels = (W_DEFAULT * H_DEFAULT) / 1e6
    act = _SLOT_ACT_BASE_GB + _SLOT_ACT_PER_MPX_GB * pixels
    if "float16" not in str(info.get("dtype")):
        act *= 2  # fp32 activations roughly double
    cost = max(0.1, info.get("weights_gb", 0) + act)  # extra slot: full weights
    extra = int((free_gb - _SLOT_RESERVE_GB) // cost)
    n = max(1, min(_SLOT_CAP, 1 + extra))
    _log(
        f"slot sizing: free {free_gb:.1f} GB, per-slot ~{cost:.1f} GB "
        f"(weights {info.get('weights_gb', 0):.1f} + act {act:.1f}, "
        f"reserve {_SLOT_RESERVE_GB:.1f}) -> {n} slots"
    )
    return n


@contextmanager
def _slot_ctx():
    """Acquire a rendering slot: rotating first-fit, non-blocking; if every
    slot is busy, queue (block) on the next slot in rotation."""
    global _rr_next
    if not _slots:
        # bootstrap slot 0 synchronously so the pool size is final
        with _BUILD_LOCK:
            if not _slots:
                s0 = _Slot()
                _slots.append(s0)
                _build_txt(s0)  # appends pool placeholders sized from VRAM
    n = len(_slots)
    for i in range(n):
        cand = _slots[(_rr_next + i) % n]
        if cand.lock.acquire(blocking=False):
            _rr_next += i + 1
            break
    else:
        cand = _slots[_rr_next % n]
        _rr_next += 1
        cand.lock.acquire()  # all busy -> wait in queue
    try:
        yield cand
    finally:
        cand.lock.release()


def _run_on_slot(fn, *args, **kwargs):
    """Run a rendering job on an acquired slot: fn(slot, *args, **kwargs)."""
    with _slot_ctx() as slot:
        return fn(slot, *args, **kwargs)


# --------------------------------------------------------------------------
# Slot pipeline building (lazy per slot)
# --------------------------------------------------------------------------

def _build_txt(slot):
    """Build the slot's own txt2img diffusion pipeline with memory
    optimizations for a single consumer GPU (works on small-VRAM cards too).
    The caller holds the slot lock; construction is serialized by callers
    using _BUILD_LOCK."""
    if slot.txt is not None:
        return slot.txt

    import torch
    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLPipeline,
    )

    device = DEVICE if torch.cuda.is_available() else "cpu"
    if DEVICE != "cpu" and not torch.cuda.is_available():
        _log(f"CUDA not available, falling back to CPU (device={device})")

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs: dict = {"dtype": dtype}

    if not IS_XL:
        raise ValueError(
            f"PICTURA_MODEL={MODEL_ID!r} is not an SDXL model: this server "
            "supports the SDXL family only"
        )

    _log(f"Loading model {MODEL_ID} (slot {_slots.index(slot)}) ...")
    t0 = time.time()

    try:
        if VAE_ID:
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(VAE_ID, dtype=dtype)
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae, **model_kwargs)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **model_kwargs)
    except Exception:
        # Retry with lower precision headroom (some repos are fp32-safetensors only).
        _log("Initial load failed, retrying with default precision...")
        model_kwargs.pop("dtype", None)
        model_kwargs.pop("vae", None)
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **model_kwargs)
    slot.txt = pipe

    # ---- memory optimizations for low-VRAM cards -----------------------------
    try:
        pipe.enable_attention_slicing()
    except Exception as e:  # noqa: BLE001
        _log(f"attention slicing skipped: {e}")
    try:
        pipe.enable_vae_slicing()
    except Exception:  # noqa: BLE001
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:  # noqa: BLE001
        pass

    # ---- fp16 VAE NaN / black-image guard for SDXL ---------------------------
    # Done BEFORE device placement so the swapped VAE is included by the
    # offload hooks or the .to(device) move below.
    if dtype == torch.float16:
        if not VAE_ID:
            # Use the fp16-safe VAE (works in fp16 directly, no fp32 upcast or
            # vae-tiling precision conflicts).
            from diffusers import AutoencoderKL

            _log("SDXL: using fp16-safe VAE madebyollin/sdxl-vae-fp16-fix")
            pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=dtype)
        else:
            # User-explicit VAE: keep decode in fp32 to avoid NaN/black output.
            try:
                pipe.upcast_vae()
                _log("SDXL: upcast_vae() enabled to avoid fp16 black-image artifacts")
            except Exception as e:  # noqa: BLE001
                _log(f"SDXL upcast_vae skipped: {e}")

    is_first = _slots and slot is _slots[0]
    offloaded = False
    weights_gb = 0.0
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        weights_gb = _estimate_weights_bytes(pipe) / 1e9
        # SDXL in fp16 OOMs around ~768px+ with weights resident on low-VRAM
        # cards, so spill to CPU up front when the default resolution needs
        # the headroom. Only slot 0 does this: extra slots exist precisely
        # because weights fit, and offloading would stall concurrent jobs.
        proactive_offload = is_first and (W_DEFAULT >= 768 or H_DEFAULT >= 768)
        if proactive_offload or weights_gb > 0.8 * vram_gb:
            _log(
                f"(proactive_offload={proactive_offload}, weights {weights_gb:.1f}/"
                f"{vram_gb:.1f} GB) -> enabling model CPU offload"
            )
            try:
                pipe.enable_model_cpu_offload()
                offloaded = True
            except Exception as e:  # noqa: BLE001
                _log(f"model cpu offload skipped: {e} -> .to({device})")
                pipe.to(device)
        else:
            _log(f"weights {weights_gb:.1f} GB fit in VRAM {vram_gb:.1f} GB -> keeping on GPU")
            pipe.to(device)
    else:
        pipe.to(device)

    pipe.safety_checker = None
    try:
        pipe._safety_checker = None  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass

    slot.info = {
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "offload": offloaded,
        "weights_gb": round(weights_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "load_seconds": round(time.time() - t0, 1),
    }
    # Size the pool once slot 0's weights are known (measured free VRAM).
    if is_first:
        n_slots = _compute_slots(slot.info)
        while len(_slots) < n_slots:
            _slots.append(_Slot())
        _log(f"pipeline pool sized: {len(_slots)} slots")
    _log(f"Model ready (slot {_slots.index(slot)}, {slot.info})")
    return pipe


def _ensure_txt(slot):
    """Return the slot's txt2img pipeline, building it once on first use."""
    if slot.txt is None:
        with _BUILD_LOCK:
            if slot.txt is None:
                _build_txt(slot)
    return slot.txt


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
    slot,
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

    pipe = _ensure_txt(slot)
    _apply_loras(pipe, lora_spec)

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    common = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

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

def _build_i2i(slot):
    """Build the slot's img2img pipeline from its own txt2img pipeline
    (shares this slot's weights -> no extra VRAM for the base model)."""
    from diffusers import AutoPipelineForImage2Image

    slot.i2i = AutoPipelineForImage2Image.from_pipe(slot.txt)
    # Weights stay resident on this slot (extra slots imply no offload), so
    # pin them to the GPU like the txt pipeline; with slot-0 offload the
    # hooks handle placement instead.
    info = slot.info
    if not info.get("offload"):
        slot.i2i.to(info.get("device") or DEVICE)
    try:
        slot.i2i.enable_attention_slicing()
    except Exception:  # noqa: BLE001
        pass
    try:
        slot.i2i.enable_vae_slicing()
        slot.i2i.enable_vae_tiling()
    except Exception:  # noqa: BLE001
        pass
    _log("img2img pipeline ready (slot-local, shared weights)")
    return slot.i2i


def _ensure_i2i(slot):
    """Return the slot's img2img pipeline, building it once on first use."""
    if slot.i2i is None:
        _ensure_txt(slot)
        with _BUILD_LOCK:
            if slot.i2i is None:
                _build_i2i(slot)
    return slot.i2i


# Whether edit_image may read server-side file paths / file:// URIs.
# Fixed policy: allowed only when the client is local (stdio) - remote
# (http/sse) stays data-URI-only (secure default). Set in main() from transport.
_ALLOW_HOST_PATHS = False


def _load_source_image(image_src: str):
    """Load the edit_image source image.

    Accepted:
      - a data:image/...;base64,... URI (always - portable across machines);
      - a local file path or file:// URI, but only when host-path reads are
        enabled for this run (local stdio client by default). Over http/sse
        the server never reads host files - that removes the
        data-exfiltration vector for remote deployments.
    """
    from PIL import Image as PILImage

    src = image_src.strip()
    if src.startswith("data:"):
        _meta, b64 = src.split(",", 1)
        return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if _ALLOW_HOST_PATHS:
        if src.startswith("file://"):
            src = src[len("file://"):]
        try:
            return PILImage.open(src).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"image not found: {src} (pass a local file path, a file:// URI, "
                "or a data: URI)"
            )
    raise ValueError(
        "image must be a data:image/...;base64,... URI (server-side file paths "
        "and file:// URIs are not accepted over http/sse - the server never "
        "reads host files remotely)"
    )


def _edit_dim(v: int) -> int:
    """Clamp an edit_image width/height. 0 = keep source size; any positive
    value is rounded to a multiple of 8 and clamped to [256, 1024] (tiny inputs
    such as 4 px land on the 256 px floor)."""
    if v <= 0:
        return 0
    return max(256, min(1024, max(1, round(v / 8)) * 8))


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
    raw = (_env(env) or "").strip()
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

    ids = list(_allowlist("PICTURA_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST) or DEFAULT_LORA_ALLOWLIST)
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
        _check_model_id("LoRA", mid, "PICTURA_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST)
        pipe.load_lora_weights(
            mid, adapter_name=name, use_safetensors=True, cache_dir=CACHE_DIR
        )
        names.append(name)
        weights.append(weight)
    pipe.set_adapters(names, adapter_weights=weights)
    # Log only a count - LoRA ids are client tool arguments and must stay out
    # of the logs (privacy guarantee).
    _log(f"LoRA applied: {len(entries)} adapter(s)")


def _resolve_control(ctype: str) -> dict:
    """Resolve an abstract control_type to its internal preprocessor/model.
    Model identifiers stay server-side (not exposed to clients)."""
    ctype = (ctype or "").strip().lower()
    if not ctype:
        raise ValueError("control_type is empty")
    info = _CONTROL_TYPES.get(ctype)
    if info is None:
        raise ValueError(f"unknown control_type '{ctype}'; available: {list(_CONTROL_TYPES)}")
    if not info.get("supported"):
        raise ValueError(
            f"control_type '{ctype}' is not yet supported by the installed preprocessor"
        )
    _check_model_id(
        "ControlNet", info["model"], "PICTURA_CONTROLNET_ALLOWLIST", DEFAULT_CONTROLNET_ALLOWLIST
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


_PREP_LOCK = threading.Lock()  # serialize shared preprocessor init + inference
_dpt_processor = None
_dpt_model = None


def _preprocess_depth(image):
    """Depth map (near = white) via DPT on the model cache."""
    # The preprocessor model is a shared, lazily built global; serialize so
    # concurrent slots cannot race its init or run CUDA inference at once.
    with _PREP_LOCK:
        return _preprocess_depth_locked(image)


def _preprocess_depth_locked(image):
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
    # Shared lazily-built YOLO model: serialize init + inference across slots.
    with _PREP_LOCK:
        return _preprocess_openpose_locked(image)


def _preprocess_openpose_locked(image):
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


def _build_cn(slot, model_id: str):
    """Build the slot's own ControlNet img2img pipeline (independent of the
    slot's txt/i2i pipelines; lazily loaded per slot and per model id)."""
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionXLControlNetImg2ImgPipeline as cn_cls)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _check_model_id("ControlNet", model_id, "PICTURA_CONTROLNET_ALLOWLIST", DEFAULT_CONTROLNET_ALLOWLIST)
    _log(f"Loading ControlNet model {model_id} (slot {_slots.index(slot)}) ...")
    cn_model = ControlNetModel.from_pretrained(
        model_id, dtype=dtype, use_safetensors=True, cache_dir=CACHE_DIR
    )

    # SDXL ControlNet img2img with the fp16-safe VAE.
    vae_kwargs = {
        "vae": AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", dtype=dtype)
    }

    pipe = cn_cls.from_pretrained(MODEL_ID, controlnet=cn_model, dtype=dtype, **vae_kwargs)
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, fn)()
        except Exception:  # noqa: BLE001
            pass

    weights_gb = _estimate_weights_bytes(pipe) / 1e9
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Slot 0 keeps the old low-VRAM offload heuristic; extra slots exist only
    # when VRAM allows resident weights, but the CN pipe is heavier - fall
    # back to offload if it does not fit.
    proactive = (_slots and slot is _slots[0]) and (W_DEFAULT >= 768 or H_DEFAULT >= 768)
    if device == "cuda" and (proactive or weights_gb > 0.8 * vram_gb):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    slot.cn[model_id] = pipe
    _log(f"ControlNet img2img pipeline ready (slot-local, {weights_gb:.1f} GB weights)")
    return pipe


def _ensure_cn(slot, model_id: str):
    """Return the slot's ControlNet pipeline for model_id, building it once."""
    if model_id not in slot.cn:
        with _BUILD_LOCK:
            if model_id not in slot.cn:
                _build_cn(slot, model_id)
    return slot.cn[model_id]


def _image2image(
    slot,
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
        pipe = _ensure_cn(slot, ctl["model"])
        _apply_loras(pipe, lora_spec)
        control_img = _preprocess_control(source, control_type)
        control_img = control_img.resize((w, h), PILImage.LANCZOS)
        common["control_image"] = control_img
        common["controlnet_conditioning_scale"] = control_scale
    else:
        pipe = _ensure_i2i(slot)
        _apply_loras(pipe, lora_spec)

    result = _oom_retry(pipe, **common)
    return result.images[0], seed, eff_steps


def _finalize_result(image, actual_seed: int, t0: float, prefix: str = "img") -> tuple:
    """Encode a generated image. Returns (png_bytes, b64, elapsed_seconds, name).

    Stateless: nothing is written to disk. The client saves the image wherever it
    wants (inline base64 in stdio mode, or a download URL over http/sse).
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    name = f"{prefix}_{int(time.time())}_{actual_seed}.png"
    return data, b64, round(time.time() - t0, 1), name


def _request_base(ctx) -> str | None:
    """Externally visible base URL of the current HTTP request.

    Builds image download URLs when PICTURA_PUBLIC_URL is not set: a
    reverse proxy forwards the public Host (and with uvicorn's proxy headers
    the X-Forwarded-Proto / X-Forwarded-Host), so the resulting URL is
    reachable by the client. None for stdio or when no request is available;
    wildcard Host values (0.0.0.0 / ::) are rejected - such URLs are not
    reachable, so the caller falls back (bind address, then inline).
    """
    if ctx is None:
        return None
    try:
        req = ctx.request_context.request
        if req is not None:
            host = getattr(req.base_url, "hostname", None)
            if host in ("0.0.0.0", "::", "", None):
                return None
            url = str(req.base_url).rstrip("/")
            return url or None
    except Exception:  # noqa: BLE001 - missing context / non-HTTP transport
        pass
    return None


def _image_result(
    png_bytes: bytes,
    name: str,
    actual_seed: int,
    elapsed: float,
    kind: str = "image",
    base: str | None = None,
):
    """Build the tool result for a generated image.

    "url" mode (http/sse): a TextContent note with a short-lived download URL,
    no inline bytes - the client fetches the image and saves it (server and
    client may not share a filesystem). The base is PICTURA_PUBLIC_URL, else
    the requesting client's Host (reverse-proxy friendly), else the bind
    address. "inline" mode (stdio / fallback): the base64 ImageContent plus a
    note; the client decodes and saves.
    """
    eff_base = _PUBLIC_BASE or base or _URL_BASE
    if _RETURN_MODE == "url" and eff_base:
        img_id = _img_cache.put(png_bytes)
        if img_id:
            url = f"{eff_base}/images/{img_id}"
            note = (
                f"Success. Download and save the {kind} as {name} (the URL is "
                f"available for ~{_img_cache.ttl:.0f}s):\n"
                f"  GET {url}\n"
                f"  e.g. curl -sSf {url} -o {name}\n"
                f"seed={actual_seed}, {elapsed}s"
            )
            return [TextContent(type="text", text=note)]
    b64 = base64.b64encode(png_bytes).decode("ascii")
    note = (
        f"Success. The {kind} {name} is attached to this result as base64 "
        f"(image content block, mimeType image/png). If you cannot view image "
        f"content, write the base64 to a file and open it. "
        f"seed={actual_seed}, {elapsed}s"
    )
    return [
        ImageContent(type="image", data=b64, mimeType="image/png"),
        TextContent(type="text", text=note),
    ]


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

# edit_image source-image wording. Chosen at server-build time from
# _ALLOW_HOST_PATHS (stdio/local = paths allowed; http/sse = data URI only).
# Module-level constants because tool annotations are stringified (postponed
# evaluation) and must resolve against module globals - enclosing locals are
# not visible to inspect.signature(eval_str=True).
_EDIT_IMAGE_DESC_LOCAL = (
    "Transform an existing image using a text prompt (img2img). Pass the "
    "source as a local file path (server host), a file:// URI, or a "
    "data:image/...;base64,... URI. In stdio mode the result is the edited "
    "image inline as base64 PNG (ImageContent); over http/sse the result note "
    "contains a short-lived download URL - the client fetches and saves it. "
    "Nothing is written to the server disk."
)
_EDIT_IMAGE_DESC_REMOTE = (
    "Transform an existing image using a text prompt (img2img). Pass the "
    "source as a data:image/...;base64,... URI (server-side file paths "
    "and file:// URIs are NOT accepted - the server never reads host "
    "files over http/sse). The result note contains a short-lived download "
    "URL - the client fetches and saves it. Nothing is written to disk."
)
_IMG_FIELD_DESC_LOCAL = (
    "Source image: a local file path (server host), a file:// URI, or a "
    "data:image/...;base64,... URI (this is a local stdio run)."
)
_IMG_FIELD_DESC_REMOTE = (
    "Source image as a data:image/...;base64,... URI. Server-side file "
    "paths and file:// URIs are not accepted over http/sse - the server "
    "never reads host files."
)


def _build_server():
    server = MCPServer(
        name="pictura",
        version="0.1.0",
        title="Image Generation (SDXL)",
        instructions=(
            "Generate images with a local SDXL (Stable Diffusion XL) pipeline "
            "running on the host GPU. Tools return images inline as base64 "
            "ImageContent; the server never writes files - the client saves "
            "them. If your client does not visually render image content, "
            "decode the returned base64 (data field, image/png) into a file "
            "with your filesystem tools and open it to inspect the result."
        ),
    )

    @server.tool(
        name="generate_image",
        title="Generate Image",
        description=(
            "Generate an image from a text prompt using the local SDXL "
            "model. In stdio mode the result is the image inline as base64 PNG "
            "(ImageContent); over http/sse the result note contains a short-lived "
            "download URL for the client to fetch and save. Nothing is written "
            "to the server disk."
        ),
    )
    async def generate_image(
        prompt: Annotated[
            str,
            Field(description="What to draw. English works best; be specific and concrete."),
        ],
        negative_prompt: Annotated[
            str,
            Field(description="Things to avoid, e.g. 'blurry, low quality'."),
        ] = "",
        width: Annotated[
            int,
            Field(description="Image width in px; rounded to a multiple of 8 and clamped to 256..1024."),
        ] = W_DEFAULT,
        height: Annotated[
            int,
            Field(description="Image height in px; rounded to a multiple of 8 and clamped to 256..1024."),
        ] = H_DEFAULT,
        num_inference_steps: Annotated[
            int,
            Field(description="Denoising steps; clamped to 10..100 (typical 25-40 for SDXL)."),
        ] = STEPS_DEFAULT,
        guidance_scale: Annotated[
            float,
            Field(description="How closely the image follows the prompt; range 1..15 (default 7.5)."),
        ] = 7.5,
        seed: Annotated[
            int,
            Field(description="Seed for reproducibility; -1 = random. The reply note reports the actual seed used."),
        ] = -1,
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
        - num_inference_steps: 25-40 typical.
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
                _run_on_slot,
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

        data, _b64, elapsed, name = _finalize_result(image, actual_seed, t0, prefix="img")
        return _image_result(data, name, actual_seed, elapsed, kind="image", base=_request_base(ctx))

    @server.tool(
        name="edit_image",
        title="Edit Image (img2img)",
        description=_EDIT_IMAGE_DESC_LOCAL if _ALLOW_HOST_PATHS else _EDIT_IMAGE_DESC_REMOTE,
    )
    async def edit_image(
        prompt: Annotated[
            str,
            Field(description="How to transform the image. English works best; be specific."),
        ],
        image: Annotated[
            str,
            Field(description=_IMG_FIELD_DESC_LOCAL if _ALLOW_HOST_PATHS else _IMG_FIELD_DESC_REMOTE),
        ],
        negative_prompt: Annotated[
            str,
            Field(description="Things to avoid, e.g. 'blurry, low quality'."),
        ] = "",
        strength: Annotated[
            float,
            Field(description="0..1: how strongly to transform (higher = more change). Default 0.6; clamped to 0.01..1.0."),
        ] = 0.6,
        width: Annotated[
            int,
            Field(description="Target width in px (rounded to a multiple of 8, clamped to 256..1024); 0 = keep the source size."),
        ] = 0,
        height: Annotated[
            int,
            Field(description="Target height in px (rounded to a multiple of 8, clamped to 256..1024); 0 = keep the source size."),
        ] = 0,
        num_inference_steps: Annotated[
            int,
            Field(description="Denoising steps; clamped to 10..100. Effective steps ≈ steps × strength."),
        ] = 25,
        guidance_scale: Annotated[
            float,
            Field(description="How closely the result follows the prompt; range 1..15 (default 7.5)."),
        ] = 7.5,
        seed: Annotated[
            int,
            Field(description="Seed for reproducibility; -1 = random. The reply note reports the actual seed used."),
        ] = -1,
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
        """img2img edit; returns the image inline (base64 PNG). The server never
        writes files - the client saves the returned image where it wants
        (identical in local and remote modes).
        """
        steps = max(10, min(100, num_inference_steps))
        strength = max(0.01, min(1.0, strength))
        width = _edit_dim(width)
        height = _edit_dim(height)

        import asyncio

        loop = asyncio.get_running_loop()
        eff_steps = int(steps * strength)

        def on_step(step: int):
            if ctx is not None:
                asyncio.run_coroutine_threadsafe(ctx.report_progress(step, eff_steps), loop)

        t0 = time.time()
        try:
            edited, actual_seed, _eff = await asyncio.to_thread(
                _run_on_slot,
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

        data, _b64, elapsed, name = _finalize_result(edited, actual_seed, t0, prefix="img2img")
        return _image_result(data, name, actual_seed, elapsed, kind="edited image", base=_request_base(ctx))

    @server.tool(
        name="server_status",
        title="Server Status",
        description=(
            "Report the loaded image model, device, dtype, VRAM usage, weight "
            "size, offload state and the number of concurrent render slots."
        ),
    )
    async def server_status() -> str:
        import asyncio

        # Load off the event loop: the first call may load the model (minutes).
        def _ensure_slot0():
            with _slot_ctx() as slot:
                _ensure_txt(slot)
                return slot.info, len(_slots)

        info, n_slots = await asyncio.to_thread(_ensure_slot0)
        return (
            f"model={info.get('model')}\n"
            f"device={info.get('device')}\n"
            f"dtype={info.get('dtype')}\n"
            f"offload={info.get('offload')}\n"
            f"weights_gb={info.get('weights_gb')}\n"
            f"vram_gb={info.get('vram_gb')}\n"
            f"concurrency_slots={n_slots}\n"
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
        lora_ids = _allowlist("PICTURA_LORA_ALLOWLIST", DEFAULT_LORA_ALLOWLIST)
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
        with _slot_ctx() as slot:
            image, seed = _generate(
                slot,
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
        _log(f"Smoke test OK: saved {out} (seed={seed}), slots={len(_slots)}, info={_slots[0].info}")

        # img2img check: reuse the generated image. On a local (stdio/smoke)
        # run host file paths are allowed, so pass the saved path; also assert
        # the portable data: URI loads.
        import io as _io
        import base64 as _b64

        _buf = _io.BytesIO()
        image.save(_buf, format="PNG")
        _src_uri = "data:image/png;base64," + _b64.b64encode(_buf.getvalue()).decode("ascii")
        if _load_source_image(_src_uri).size != (256, 256):
            raise AssertionError("data URI load mismatch")
        if not _ALLOW_HOST_PATHS or _load_source_image(str(out)).size != (256, 256):
            raise AssertionError("host path load mismatch (local run should allow paths)")
        try:
            with _slot_ctx() as slot:
                edited, seed2, _ = _image2image(
                    slot,
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
                        " (default: $PICTURA_HOST or 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port for http/sse (default: $PICTURA_PORT or 8000)",
    )
    parser.add_argument(
        "--max-body-mb",
        type=int,
        default=int(_env("PICTURA_MAX_BODY_MB", "16") or "16"),
        help="max HTTP request body size in MB for http/sse (default 16)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="require API key on http/sse via PICTURA_API_KEY header (falls back to $PICTURA_API_KEY)",
    )
    parser.add_argument(
        "--log-file",
        default=_env("PICTURA_LOG_FILE"),
        help="append [pictura-mcp] logs to a file (default: stderr / journald)",
    )
    args = parser.parse_args()
    if not IS_XL:
        print(
            f"[pictura-mcp] PICTURA_MODEL={MODEL_ID!r} is not an SDXL-family "
            "checkpoint (model id must contain 'xl')",
            file=sys.stderr,
            flush=True,
        )
        return 2
    token = args.api_key or os.environ.get("PICTURA_API_KEY")
    http_port = args.port or int(_env("PICTURA_PORT", "8000") or "8000")
    http_host = args.host or _env("PICTURA_HOST", "127.0.0.1")
    _set_log_file(args.log_file)
    # Image return mode: http/sse -> short-lived download URLs (the base is
    # PICTURA_PUBLIC_URL, else the request Host from the reverse proxy, else
    # the bind address); stdio -> inline base64.
    global _RETURN_MODE, _URL_BASE
    if args.transport in ("http", "sse"):
        _RETURN_MODE = "url"
        _URL_BASE = _bind_base(http_host, http_port)
        if _PUBLIC_BASE:
            _log(f"image return mode: url (base {_PUBLIC_BASE})")
        else:
            _log(
                "image return mode: url (base from request Host"
                + (f", fallback {_URL_BASE})" if _URL_BASE else ", fallback bind address)")
            )
    else:
        _RETURN_MODE = "inline"
        _URL_BASE = None
    # edit_image host-path reads: fixed policy - allowed for a local stdio
    # client, denied over http/sse.
    global _ALLOW_HOST_PATHS
    _ALLOW_HOST_PATHS = args.transport == "stdio" or args.smoke
    if _ALLOW_HOST_PATHS:
        _log("edit_image: host file paths allowed (local stdio run)")
    else:
        _log("edit_image: host file paths disabled (http/sse) - data: URIs only")
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
    if not _env("PICTURA_SKIP_PREFETCH") == "1":
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


def _auth_ok(request, token: str | None) -> bool:
    """API-key check for MCP requests (PICTURA_API_KEY header).

    token=None (no key configured) allows all requests; /images is a separate
    capability URL (unguessable id + TTL) and uses no key.
    """
    if not token:
        return True
    return request.headers.get("pictura_api_key") == token


def _attach_http_middleware(mcp_app, token: str | None):
    """Add GET /images/{img_id} and the API-key check to the MCP app.

    Applied as middleware ON the MCP app itself, and uvicorn runs that app as
    the top-level ASGI app, so the MCP session manager's lifespan (task-group
    init) still runs. Wrapping the app in another Starlette app with Mount
    would skip that lifespan and crash with "Task group is not initialized".

    The image id is an unguessable secret, so the URL itself is the capability
    (valid for the short cache TTL; no additional auth).
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, Response

    async def _serve_image(request):
        img_id = request.url.path[len("/images/"):]
        data = _img_cache.get(img_id)
        if data is None:
            return JSONResponse({"error": "image not found or expired"}, status_code=404)
        return Response(content=data, media_type="image/png")

    async def _dispatch(request, call_next):
        if request.url.path.startswith("/images/"):
            return await _serve_image(request)
        if not _auth_ok(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    mcp_app.add_middleware(BaseHTTPMiddleware, dispatch=_dispatch)


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

    if transport == "sse":
        mcp_app = server.sse_app(
            host=host, max_request_body_size=max_body_bytes
        )
        path = "/sse"
    else:
        mcp_app = server.streamable_http_app(
            host=host, json_response=True, max_request_body_size=max_body_bytes
        )
        path = "/mcp"

    _attach_http_middleware(mcp_app, token)
    if token:
        _log(
            f"MCP {transport} server: http://{host}:{port}{path} "
            f"(auth: API key, max body {max_body_bytes // (1024 * 1024)} MB)"
        )
    else:
        _log(
            f"MCP {transport} server: http://{host}:{port}{path} "
            f"(NO auth - only for trusted LAN, max body {max_body_bytes // (1024 * 1024)} MB)"
        )

    uvicorn.run(
        mcp_app,
        host=host,
        port=port,
        log_level="warning",
        forwarded_allow_ips=_FWD_ALLOW_IPS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
