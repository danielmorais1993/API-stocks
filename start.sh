#!/usr/bin/env bash
# start.sh — entrypoint to run a FastAPI app with uvicorn
# Usage:
#   ./start.sh
#   PORT=8000 UVICORN_WORKERS=2 APP_MODULE="myapp:app" ./start.sh
# On Render/Heroku set the Start Command to: /bin/bash start.sh

set -euo pipefail

# --- Load .env (only simple KEY=VALUE lines) ---
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    # skip empty lines and comments
    case "$line" in
      ''|\#*) continue ;;
    esac
    # accept only simple KEY=VALUE (no spaces around =)
    if printf '%s\n' "$line" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*=' 2>/dev/null; then
      key=${line%%=*}
      value=${line#*=}
      # strip surrounding quotes if present
      if [[ ${value} == \"*\" && ${value} == *\" ]]; then
        value=${value:1:-1}
      elif [[ ${value} == \'*\' && ${value} == *\' ]]; then
        value=${value:1:-1}
      fi
      export "$key=$value"
    fi
  done < .env
fi

# --- runtime config (can be overridden by environment) ---
PORT="${PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"
APP_MODULE="${APP_MODULE:-app:app}"    # default FastAPI module:object

# If uvicorn is not in PATH (virtualenv not activated), try python -m uvicorn
if command -v uvicorn >/dev/null 2>&1; then
  UVICORN_CMD="$(command -v uvicorn)"
else
  UVICORN_CMD="$(python -m uvicorn)"
fi

# run uvicorn (exec replaces shell process with uvicorn)
exec $UVICORN_CMD "$APP_MODULE" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  --proxy-headers
