#!/usr/bin/env bash
# Install pictura-mcp as a system-level systemd service running under a
# dedicated (unprivileged) service account.
#
# Usage (as root):
#   deploy/install-systemd.sh [PROJECT_ROOT] [SERVICE_USER] [PORT]
#   default: PROJECT_ROOT = script's repo root, SERVICE_USER = pictura-mcp, PORT = 8000
#   PORT seeds IMAGE_PORT in the env file (an existing IMAGE_PORT value wins).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "$script_dir/.." && pwd)}"
SERVICE_USER="${2:-pictura-mcp}"
PORT="${3:-8000}"
UNIT="${4:-/etc/systemd/system/pictura-mcp.service}"

if [[ $EUID -ne 0 ]]; then
  echo "error: run as root" >&2; exit 1
fi
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "error: $PROJECT_ROOT/.venv/bin/python not found -" >&2
  echo "       run README §Setup step 1 first (python3 -m venv .venv && pip install -r server/requirements.txt)" >&2
  exit 1
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
ENV_FILE="$PROJECT_ROOT/deploy/pictura-mcp.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$PROJECT_ROOT/deploy/pictura-mcp.env.example" "$ENV_FILE"
  sed -i "s/^PICTURA_MCP_TOKEN=.*/PICTURA_MCP_TOKEN=$(head -c24 /dev/urandom | base64 | tr -d '/+=')/" "$ENV_FILE"
fi
# seed_env KEY VALUE: skip when active; activate the "# KEY=..." template line; append otherwise.
seed_env() {
  local key="$1" val="$2"
  grep -q "^${key}=" "$ENV_FILE" && return 0
  if grep -q "^# ${key}=" "$ENV_FILE"; then
    sed -i "s|^# ${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
  fi
}
# HF default cache (~/.cache) is read-only under ProtectSystem=strict.
seed_env IMAGE_MODEL_CACHE_DIR "$PROJECT_ROOT/.model-cache"
CACHE_DIR="$(grep '^IMAGE_MODEL_CACHE_DIR=' "$ENV_FILE" | cut -d= -f2-)"
seed_env IMAGE_HOST 0.0.0.0
seed_env IMAGE_PORT "$PORT"
# server writes only to the model cache at runtime (--smoke is a manual run).
if [[ -n $CACHE_DIR ]]; then
  mkdir -p "$CACHE_DIR"
  chown -R "$SERVICE_USER":"$SERVICE_USER" "$CACHE_DIR"
fi
chmod o-w "$PROJECT_ROOT" 2>/dev/null || true   # keep project read-only for the service user
chown "$SERVICE_USER":"$SERVICE_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "==> render & install unit: $UNIT"
sed -e "s#<PROJECT_ROOT>#$PROJECT_ROOT#g" \
    -e "s#<SERVICE_USER>#$SERVICE_USER#g" \
    "$PROJECT_ROOT/deploy/pictura-mcp.service" > "$UNIT"
# ProtectSystem=strict blocks writes outside ReadWritePaths; whitelist the
# model cache dir (and log file, when set) so prefetch/download writes work.
RW_PATHS=""
if [[ -n $CACHE_DIR ]]; then RW_PATHS="$CACHE_DIR"; fi
if grep -q '^IMAGE_LOG_FILE=' "$ENV_FILE"; then
  LOG_FILE="$(grep '^IMAGE_LOG_FILE=' "$ENV_FILE" | cut -d= -f2-)"
  [[ -n $RW_PATHS ]] && RW_PATHS="$RW_PATHS $LOG_FILE" || RW_PATHS="$LOG_FILE"
fi
if [[ -n $RW_PATHS ]]; then
  sed -i "s#^ReadWritePaths=.*#ReadWritePaths=$RW_PATHS#" "$UNIT"
else
  sed -i "s#^ReadWritePaths=.*#ReadWritePaths=#" "$UNIT"
fi
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
# register boot-autostart only; the operator starts after reviewing the env file
systemctl enable pictura-mcp
if systemctl is-active --quiet pictura-mcp; then
  echo "NOTE: already running; restart after env edits: systemctl restart pictura-mcp"
fi
echo
echo "Done. NEXT STEPS:"
echo "  1. review the env file - the token is already randomized and the essentials"
echo "     (IMAGE_MODEL_CACHE_DIR / IMAGE_HOST / IMAGE_PORT) are pre-seeded;"
echo "     edit only what needs changing:"
echo "        sudoedit $ENV_FILE"
echo "  2. start the service:"
echo "        systemctl start pictura-mcp"
PORT_EFF="$(grep '^IMAGE_PORT=' "$ENV_FILE" | cut -d= -f2-)"
PORT_EFF="${PORT_EFF:-$PORT}"
echo "  3. verify - watch the log until 'MCP http server: http://...:${PORT_EFF}/mcp'"
echo "     and 'Model ready' appear:"
echo "        journalctl -u pictura-mcp -f"
echo "  4. point your MCP client at http://<this-box-ip>:${PORT_EFF}/mcp with the"
echo "     bearer token from PICTURA_MCP_TOKEN in $ENV_FILE"
echo "     (template: deploy/mcp.remote.json.example)"
echo "  5. open the port in your firewall if remote machines must reach the GPU box."
echo
echo "More follow-ups:"
echo "  - logs: journalctl -u pictura-mcp -f   (or IMAGE_LOG_FILE when set)"
echo "  - TO READ THE LOG FILE AS A NON-ROOT OPERATOR, add them to the log group:"
echo "        sudo usermod -aG $SERVICE_USER <your-username>   # then re-login"
echo "    (logrotate recreate uses group $SERVICE_USER - see deploy/logrotate.example)"
echo "  - MemoryDenyWriteExecute may break torch; remove it from the unit if the service crashes."
