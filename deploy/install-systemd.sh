#!/usr/bin/env bash
# Install pictura-mcp as a system-level systemd service running under a
# dedicated (unprivileged) service account.
#
# Usage (as root):
#   deploy/install-systemd.sh [PROJECT_ROOT] [SERVICE_USER] [PORT]
#   default: PROJECT_ROOT = script's repo root, SERVICE_USER = pictura-mcp, PORT = 8000
#   PORT seeds PICTURA_PORT in the env file (an existing PICTURA_PORT value wins).
#
# Env handling: when the env template changes, the env file is left untouched
# and the current rendered template is written to deploy/pictura-mcp.env.new.
# After diff/merge, run with --adopt-env: it records the template and removes
# the .new file, then exits (standalone - no install work).
# The API key is generated on first install and kept afterwards.
set -euo pipefail

# Pull the optional --adopt-env flag out before positional parsing.
ADOPT=0
_args=()
for _a in "$@"; do
  if [[ "$_a" == "--adopt-env" ]]; then ADOPT=1; else _args+=("$_a"); fi
done
set -- "${_args[@]}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "$script_dir/.." && pwd)}"
SERVICE_USER="${2:-pictura-mcp}"
PORT="${3:-8000}"
UNIT="${4:-/etc/systemd/system/pictura-mcp.service}"

if [[ $EUID -ne 0 ]]; then
  echo "error: run as root" >&2; exit 1
fi

# --adopt-env is a standalone maintenance step: record the current env template
# and remove the .new artifact, then exit (no install work is done).
if [[ $ADOPT -eq 1 ]]; then
  ENV_FILE="$PROJECT_ROOT/deploy/pictura-mcp.env"
  TEMPLATE="$PROJECT_ROOT/deploy/pictura-mcp.env.example"
  NEW_FILE="${ENV_FILE}.new"
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "error: $ENV_FILE does not exist" >&2; exit 1
  fi
  tpl_sha="$(sha256sum "$TEMPLATE" 2>/dev/null | awk '{print $1}')" || true
  tpl_sha="${tpl_sha:-unknown}"
  if grep -q '^# *template-fingerprint:' "$ENV_FILE"; then
    sed -i "s|^# *template-fingerprint:.*|# template-fingerprint: $tpl_sha|" "$ENV_FILE"
  else
    printf '\n# template-fingerprint: %s\n' "$tpl_sha" >> "$ENV_FILE"
  fi
  rm -f "$NEW_FILE"
  echo "adopted the current env template: fingerprint recorded, ${NEW_FILE##*/} removed"
  exit 0
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
TEMPLATE="$PROJECT_ROOT/deploy/pictura-mcp.env.example"
NEW_FILE="$ENV_FILE.new"  # current rendered template, written when it changes
# sha256 of env.example; "recorded vs current" detects template changes.
tpl_sha="$(sha256sum "$TEMPLATE" 2>/dev/null | awk '{print $1}')" || true
tpl_sha="${tpl_sha:-unknown}"
recorded="$(grep -m1 '^# *template-fingerprint:' "$ENV_FILE" 2>/dev/null | sed 's/^# *template-fingerprint: *//' || true)"
recorded="${recorded//[[:space:]]/}"

write_fingerprint() {  # $1 = sha
  if grep -q '^# *template-fingerprint:' "$ENV_FILE"; then
    sed -i "s|^# *template-fingerprint:.*|# template-fingerprint: $1|" "$ENV_FILE"
  else
    printf '\n# template-fingerprint: %s\n' "$1" >> "$ENV_FILE"
  fi
}

created=0
if [[ ! -f "$ENV_FILE" ]]; then
  # Fresh install: build the full env from the template (with machine values).
  cp "$TEMPLATE" "$ENV_FILE"
  sed -i -e "s|^# PICTURA_MODEL_CACHE_DIR=.*|PICTURA_MODEL_CACHE_DIR=$PROJECT_ROOT/.model-cache|" \
         -e "s|^# PICTURA_HOST=.*|PICTURA_HOST=0.0.0.0|" \
         -e "s|^# PICTURA_PORT=.*|PICTURA_PORT=$PORT|" "$ENV_FILE"
  created=1
fi
CACHE_DIR="$(grep '^PICTURA_MODEL_CACHE_DIR=' "$ENV_FILE" | cut -d= -f2- || true)"

# The API key must be real: generate an sk-pictura-... key when missing/
# placeholder; an existing custom key is kept.
cur_key="$(grep '^PICTURE_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
if [[ $created -eq 1 || -z "$cur_key" || "$cur_key" == *"change-me"* ]]; then
  gen="sk-pictura-$(head -c18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  if grep -q '^PICTURE_API_KEY=' "$ENV_FILE"; then
    sed -i "s|^PICTURE_API_KEY=.*|PICTURE_API_KEY=${gen}|" "$ENV_FILE"
  else
    printf 'PICTURE_API_KEY=%s\n' "$gen" >> "$ENV_FILE"
  fi
  cur_key="$gen"
  echo "  API key: generated (${gen:0:11}...)"
else
  echo "  API key: kept (${cur_key:0:11}...)"
fi

if [[ $created -eq 1 ]]; then
  write_fingerprint "$tpl_sha"
  echo "  created from template (API key generated, fingerprint recorded)"
elif [[ -z "$recorded" ]]; then
  write_fingerprint "$tpl_sha"
  echo "  env kept; first recorded template fingerprint"
elif [[ "$tpl_sha" == "$recorded" ]]; then
  echo "  synchronized with the template (no template change)"
else
  # Template changed: leave the env untouched, write the .new file for merge.
  # The .new carries the existing env's API key so merging it keeps clients
  # working (no key rotation from the template's placeholder).
  cp "$TEMPLATE" "$NEW_FILE"
  sed -i -e "s#<PROJECT_ROOT>#$PROJECT_ROOT#g" \
         -e "s|^# PICTURA_MODEL_CACHE_DIR=.*|PICTURA_MODEL_CACHE_DIR=$PROJECT_ROOT/.model-cache|" \
         -e "s|^# PICTURA_HOST=.*|PICTURA_HOST=0.0.0.0|" \
         -e "s|^# PICTURA_PORT=.*|PICTURA_PORT=$PORT|" \
         -e "s|^PICTURE_API_KEY=.*|PICTURE_API_KEY=${cur_key}|" "$NEW_FILE"
  chmod 600 "$NEW_FILE"
  echo "  template changed - your env is untouched; wrote ${NEW_FILE##*/}"
  echo "    diff:  diff $ENV_FILE $NEW_FILE"
  echo "    then merge, drop ${NEW_FILE##*/}, and re-run with --adopt-env"
fi

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
if grep -q '^PICTURA_LOG_FILE=' "$ENV_FILE"; then
  LOG_FILE="$(grep '^PICTURA_LOG_FILE=' "$ENV_FILE" | cut -d= -f2-)"
  [[ -n $RW_PATHS ]] && RW_PATHS="$RW_PATHS $LOG_FILE" || RW_PATHS="$LOG_FILE"
fi
if [[ -n $RW_PATHS ]]; then
  sed -i "s#^ReadWritePaths=.*#ReadWritePaths=$RW_PATHS#" "$UNIT"
else
  sed -i "s#^ReadWritePaths=.*#ReadWritePaths=#" "$UNIT"
fi
if grep -q '^PICTURA_LOG_FILE=' "$ENV_FILE"; then
  LOG_FILE="$(grep '^PICTURA_LOG_FILE=' "$ENV_FILE" | cut -d= -f2-)"
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
PORT_EFF="$(grep '^PICTURA_PORT=' "$ENV_FILE" | cut -d= -f2-)"
PORT_EFF="${PORT_EFF:-$PORT}"
echo
echo "Done."
echo
echo "Next:"
echo "  1. review  $ENV_FILE          (API key + essentials pre-populated)"
echo "  2. start   systemctl start pictura-mcp"
echo "  3. watch   journalctl -u pictura-mcp -f   (until 'Model ready' appears)"
echo "  4. client  http://<this-box-ip>:${PORT_EFF}/mcp   with the header"
echo "             PICTURE_API_KEY: <the key in $ENV_FILE>"
echo "             config template: deploy/mcp.remote.json.example"
if [[ -f "$NEW_FILE" ]]; then
echo
echo "  NOTE: ${NEW_FILE##*/} exists - the env template changed. Review it, merge"
echo "        what you want, delete it, then run this script with --adopt-env"
echo "        (standalone: records the template and removes the file)."
fi
echo
echo "Help: logs go to journalctl (or PICTURA_LOG_FILE). If the unit fails at"
echo "      startup, remove MemoryDenyWriteExecute from the unit (torch conflict)"
echo "      - see README. Non-root log readers: sudo usermod -aG $SERVICE_USER <user>"
