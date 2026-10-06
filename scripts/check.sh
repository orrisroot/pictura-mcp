#!/usr/bin/env bash
# Lightweight dev checks for pictura-mcp (no GPU / no model download needed).
# Run from anywhere:  bash scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

echo "==> compile"
"$PY" -m py_compile server/pictura_server.py

echo "==> import module (catches NameError at module level)"
"$PY" - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("pictura_check", "server/pictura_server.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("import ok; tools/tokens/cache symbols present:",
      all(hasattr(m, n) for n in
          ("_issue_upload_token", "_claim_upload_token",
           "_finish_upload_token", "_ImageCache")))
EOF

echo "==> env example: active lines must not carry inline '#' (systemd)"
if grep -nE '^[A-Za-z_][A-Za-z0-9_]*=.*#' deploy/pictura-mcp.env.example; then
  echo "FAIL: inline comment on an active env line"; exit 1
fi
bash -n deploy/pictura-mcp.env.example
# declarative deploy artifacts (unit / account / sync list) must be present
test -f deploy/pictura-mcp.service
test -f deploy/sysusers.d/pictura-mcp.conf
test -f deploy/install-files.txt

echo "==> stale wording / old names must be absent"
if grep -rnE "PICTURA_IMAGE_URL|PICTURA_UPLOAD_TICKET_TTL|PICTURA_IMAGE_UPLOAD_TICKET_TTL|PICTURA_MAX_BODY_MB|API-key protected|same API-key header" \
    server/pictura_server.py README.md REFERENCE.md skills deploy/*.example; then
  echo "FAIL: stale wording found"; exit 1
fi
# the tool-table wording must not call upload_image "http/sse only"
if grep -rn "http/sse only" README.md REFERENCE.md skills/pictura-mcp/SKILL.md; then
  echo "FAIL: 'http/sse only' wording"; exit 1
fi

echo "==> every server tool appears in the docs"
"$PY" - <<'EOF'
import re
src = open("server/pictura_server.py").read()
# Names passed directly, or collected into a kwargs dict that carries a
# name="..." literal (e.g. _edit_tool_kwargs = dict(name="edit_image", ...)).
tools = set(re.findall(r'@server\.tool\(\s*name="([a-z_]+)"', src))
tools |= set(re.findall(r'dict\(\s*name="([a-z_]+)"', src))
tools = sorted(tools)
docs = open("README.md").read() + open("REFERENCE.md").read() + open("skills/pictura-mcp/SKILL.md").read()
missing = [t for t in tools if t not in docs]
assert not missing, f"tool(s) missing from docs: {missing}"
print("tools:", tools)
EOF

echo "==> every server env var is documented"
"$PY" - <<'EOF'
import re
src = open("server/pictura_server.py").read()
names = set(re.findall(r'_env\("(PICTURA_[A-Z_]+)"', src))
names |= {"PICTURA_API_KEY"}
docs = open("README.md").read() + open("REFERENCE.md").read() + open("deploy/pictura-mcp.env.example").read()
missing = sorted(n for n in names if n not in docs)
assert not missing, f"env var(s) not documented: {missing}"
print(f"{len(names)} env vars documented")
EOF

echo "all checks passed"
