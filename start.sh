#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    if printf '%s\n' "$line" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*='; then
      key=${line%%=*}
      value=${line#*=}
      if [[ ${value} == \"*\" && ${value} == *\" ]]; then
        value=${value:1:-1}
      elif [[ ${value} == \'*\' && ${value} == *\' ]]; then
        value=${value:1:-1}
      fi
      export "$key=$value"
    fi
  done < .env
fi
PORT="${PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"
APP_MODULE="${APP_MODULE:-app:app}"
exec uvicorn "$APP_MODULE" --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}" --proxy-headers
