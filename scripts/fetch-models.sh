#!/usr/bin/env bash
# Provision every model referenced by the model config as plain local
# directories under PICTURA_MODELS_DIR (default: <project>/models):
#   models/<org>/<repo>/            LoRA / ControlNet / preprocessor / repo-id base models
#   models/<name>/                  base models placed as local dirs (config "model")
#   models/yolov8n-pose.pt          openpose preprocessor
# Config: server/model.json (deployment-local; copy a preset from
# server/examples/*.json, or pass any config path as the first argument).
# Run BEFORE starting the server (the service never downloads):
#   sudo scripts/fetch-models.sh [/path/to/model.json]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PROJECT_ROOT/server/model.json}"
MODELS_DIR="${PICTURA_MODELS_DIR:-$PROJECT_ROOT/models}"

HF="$PROJECT_ROOT/.venv/bin/hf"
if [ ! -x "$HF" ]; then
  HF="$(command -v hf || true)"
fi
if [ -z "$HF" ]; then
  echo "[fetch-models] error: huggingface 'hf' CLI not found" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "[fetch-models] error: config not found: $CONFIG" >&2
  echo "[fetch-models] hint: copy a preset (server/examples/model.*.json) to" >&2
  echo "[fetch-models]       server/model.json, or pass a config path." >&2
  exit 1
fi
echo "[fetch-models] config: $CONFIG"
echo "[fetch-models] models dir: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# kind: base|vae|lora|controlnet|prep -> org/repo id (base may be an existing
# local dir, printed as "SKIP\t<dir>"), plus a synthetic yolo line.
while IFS=$'\t' read -r kind rid; do
  case "$kind" in
    SKIP)
      echo "[fetch-models] already local (skip): $rid"
      ;;
    yolo)
      dest="$MODELS_DIR/yolov8n-pose.pt"
      if [ -s "$dest" ]; then
        echo "[fetch-models] already local (skip): $dest"
      else
        echo "[fetch-models] downloading yolo: yolov8n-pose.pt"
        curl -fsSL "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt" -o "$dest"
      fi
      ;;
    *)
      org="${rid%%/*}"
      repo="${rid##*/}"
      dest="$MODELS_DIR/$org/$repo"
      # The Qwen-Image 2.1 uncensored LoRA lives as a single file inside the
      # abenzerps/Qwen-Image-2.1-Uncensored-GGUF repo (no standalone HF repo of
      # the config id); fetch that one file, cached after the first run.
      if [ "$rid" = "abenzerps/qwen-image-2.1-uncensored-lora" ]; then
        file="$dest/qwen-image-2.1-uncensored-lora.safetensors"
        if [ -s "$file" ]; then
          echo "[fetch-models] already local (skip): $rid -> $file"
          continue
        fi
        echo "[fetch-models] downloading lora: abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
        mkdir -p "$dest"
        # Download to a temp file and move into place, so an interrupted
        # transfer never leaves a truncated safetensors that the next run
        # would treat as provisioned.
        if curl -fsSL "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/qwen-image-2.1-uncensored-lora.safetensors" -o "$file.part"; then
          mv "$file.part" "$file"
        else
          rm -f "$file.part"
          echo "[fetch-models] error: failed to fetch $rid (not a standalone repo; see README)" >&2
          exit 1
        fi
        continue
      fi
      # Viggle turbo (few-step distilled Qwen adapter): fetch only the r256
      # LoRA file and the scheduler config - the repo also carries quantized
      # merged transformers, older LoRA versions and ComfyUI assets that the
      # server does not use.
      if [ "$rid" = "Viggle/Qwen-Image-2.1-viggle-turbo" ]; then
        if [ -s "$dest/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors" ] && [ -s "$dest/scheduler/scheduler_config.json" ]; then
          echo "[fetch-models] already local (skip): $rid -> $dest"
          continue
        fi
        echo "[fetch-models] downloading turbo adapter + scheduler: $rid"
        mkdir -p "$dest"
        "$HF" download "$rid" "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors" "scheduler/scheduler_config.json" --local-dir "$dest"
        continue
      fi
      if [ -f "$dest/model_index.json" ] || [ -n "$(find "$dest" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
        echo "[fetch-models] already local (skip): $rid -> $dest"
        continue
      fi
      echo "[fetch-models] downloading $kind: $rid -> $dest"
      mkdir -p "$dest"
      "$HF" download "$rid" --local-dir "$dest"
      ;;
  esac
done < <(python3 - "$CONFIG" "$PROJECT_ROOT" <<'PY'
import json, os, sys

cfg = json.load(open(sys.argv[1]))
root = sys.argv[2]
out = []
seen = set()

def add(kind, rid):
    if isinstance(rid, str) and rid.strip() and (kind, rid.strip()) not in seen:
        seen.add((kind, rid.strip()))
        out.append((kind, rid.strip()))

def local(p):
    """Existing absolute dir -> SKIP; relative path resolved from the project root."""
    if isinstance(p, str) and p.strip():
        cand = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.isdir(cand):
            return cand
    return None

model = cfg.get("model") or ""
m = local(model)
if m:
    out.append(("SKIP", m))
elif model:
    add("base", model)
vae = cfg.get("vae") or ""
v = local(vae)
if v:
    out.append(("SKIP", v))
elif vae:
    add("vae", vae)
# Auto-VAEs declared per family in the config (e.g. SDXL fp16-safe VAE).
# Turbo entries (few-step distilled adapters, e.g. Viggle) contribute a LoRA
# id and a scheduler repo, both provisioned like any other weight.
for fam_cfg in (cfg.get("families") or {}).values():
    if isinstance(fam_cfg, dict):
        add("vae", fam_cfg.get("auto_vae"))
        turbo = fam_cfg.get("turbo") or {}
        if isinstance(turbo, dict):
            add("lora", turbo.get("lora"))
            add("scheduler", turbo.get("scheduler"))
loras = cfg.get("supported_loras") or {}
if isinstance(loras, dict):
    lora_ids = list(loras.keys())
elif isinstance(loras, list):
    lora_ids = loras
else:
    lora_ids = []
if "*" not in lora_ids:
    for rid in lora_ids:
        add("lora", rid)
for ctype, info in (cfg.get("control_types") or {}).items():
    if isinstance(info, dict):
        add("controlnet:" + ctype, info.get("model"))
        add("prep:" + ctype, info.get("prep_model"))
# openpose preprocessor is a direct file download (not an HF repo).
for _, info in (cfg.get("control_types") or {}).items():
    if isinstance(info, dict) and (info.get("model") or "").endswith("openpose"):
        add("yolo", "yolov8n-pose.pt")

for kind, rid in out:
    print(f"{kind}\t{rid}")
PY
)

echo "[fetch-models] done."
