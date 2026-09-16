#!/usr/bin/env bash
# Install image-mcp as a system-level systemd service running under a
# dedicated (unprivileged) service account.
#
# Usage (as root):
#   deploy/install-systemd.sh [PROJECT_ROOT] [SERVICE_USER] [PORT]
#   default: PROJECT_ROOT = script's repo root, SERVICE_USER = image-mcp, PORT = 8000
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "$script_dir/.." && pwd)}"
SERVICE_USER="${2:-image-mcp}"
PORT="${3:-8000}"
UNIT="${4:-/etc/systemd/system/image-mcp.service}"

if [[ $EUID -ne 0 ]]; then
  echo "error: run as root" >&2; exit 1
fi

echo "==> service account: $SERVICE_USER"
if id "$SERVICE_USER" &>/dev/null; then
  echo "    exists"
else
  useradd --system --home-dir /var/lib/"$SERVICE_USER" --shell /usr/sbin/nologin \
          "$SERVICE_USER" 2>/dev/null || useradd --system --shell /usr/sbin/nologin "$SERVICE_USER"
  echo "    created (system account)"
  mkdir -p /var/lib/"$SERVICE_USER"; chown "$SERVICE_USER":"$SERVICE_USER" /var/lib/"$SERVICE_USER"
fi

echo "==> project access ($PROJECT_ROOT)"
# The service user needs read+traverse on the project and venv.
for d in "$PROJECT_ROOT" "$PROJECT_ROOT"/server "$PROJECT_ROOT"/.venv; do
  [ -e "$d" ] && chmod o+rx "$d" 2>/dev/null || true
done
if command -v setfacl &>/dev/null; then
  setfacl -m "u:$SERVICE_USER:rX" "$PROJECT_ROOT" 2>/dev/null || true
fi

echo "==> env file"
ENV_FILE="$PROJECT_ROOT/deploy/image-mcp.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$PROJECT_ROOT/deploy/image-mcp.env.example" "$ENV_FILE"
  sed -i "s/^IMAGE_MCP_TOKEN=.*/IMAGE_MCP_TOKEN=$(head -c24 /dev/urandom | base64 | tr -d '/+=')/" "$ENV_FILE"
fi
# Ensure a writable model cache owned by the service user.
if ! grep -q '^IMAGE_MODEL_CACHE_DIR=' "$ENV_FILE"; then
  printf 'IMAGE_MODEL_CACHE_DIR=%s/.model-cache\n' "$PROJECT_ROOT" >> "$ENV_FILE"
fi
CACHE_DIR="$(grep '^IMAGE_MODEL_CACHE_DIR=' "$ENV_FILE" | cut -d= -f2-)"
mkdir -p "$CACHE_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$CACHE_DIR"
chmod o-w "$PROJECT_ROOT" 2>/dev/null || true   # keep project read-only for the service user
chown "$SERVICE_USER":"$SERVICE_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "==> render & install unit: $UNIT"
sed -e "s#<PROJECT_ROOT>#$PROJECT_ROOT#g" \
    -e "s#<SERVICE_USER>#$SERVICE_USER#g" \
    -e "s#--port 8000#--port $PORT#" \
    "$PROJECT_ROOT/deploy/image-mcp.service" > "$UNIT"
# IMAGE_MCP_TOKEN in the unit comes from the env file; keep logrotate path hint.
if grep -q '^IMAGE_LOG_FILE=' "$ENV_FILE"; then
  LOG_FILE="$(grep '^IMAGE_LOG_FILE=' "$ENV_FILE" | cut -d= -f2-)"
  LOG_DIR="$(dirname "$LOG_FILE")"
  mkdir -p "$LOG_DIR"
  chown "$SERVICE_USER":"$SERVICE_USER" "$LOG_DIR" 2>/dev/null || true
  # touch the log file so ownership/perms are set before first start
  touch "$LOG_FILE" 2>/dev/null || true
  chown "$SERVICE_USER":"$SERVICE_USER" "$LOG_FILE" 2>/dev/null || true
  chmod 640 "$LOG_FILE" 2>/dev/null || true
fi

systemctl daemon-reload
systemctl enable --now image-mcp
systemctl status image-mcp --no-pager || true
echo
echo "Done. COMMON FOLLOW-UPS:"
echo "  - set IMAGE_MCP_TOKEN / IMAGE_MODEL / IMAGE_CUDA_DEVICE / IMAGE_LOG_FILE in $ENV_FILE, then: systemctl restart image-mcp"
echo "  - logs: journalctl -u image-mcp -f   (or IMAGE_LOG_FILE when set)"
echo "  - TO READ THE LOG FILE AS A NON-ROOT OPERATOR, add them to the log group:"
echo "        sudo usermod -aG $SERVICE_USER <your-username>   # then re-login"
echo "    (logrotate recreate uses group $SERVICE_USER - see deploy/logrotate.example)"
echo "  - MemoryDenyWriteExecute may break torch; remove it from the unit if the service crashes."
