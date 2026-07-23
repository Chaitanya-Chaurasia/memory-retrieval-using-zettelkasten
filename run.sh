#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# needs a python whose sqlite3 can load extensions (homebrew works, Xcode's doesn't)
PY="${PY:-/opt/homebrew/bin/python3.13}"

if [ ! -x "$PY" ]; then
  echo "couldn't find $PY. install with: brew install python@3.13 (or set PY=/path/to/python)"
  exit 1
fi

if [ ! -d backend/.venv ]; then
  echo "first run: creating venv and installing python deps (this pulls pytorch, give it a few minutes)"
  "$PY" -m venv backend/.venv
  backend/.venv/bin/pip install -q -r backend/requirements.txt
fi

if grep -q "paste-your-key-here" backend/.env 2>/dev/null; then
  echo "BYOK: put your Anthropic API key in backend/.env before running."
  exit 1
fi

if [ ! -d frontend/node_modules ]; then
  echo "first run: installing npm packages"
  (cd frontend && npm install)
fi

# kill both servers on ctrl-c
trap 'kill 0' EXIT INT TERM

(cd backend && .venv/bin/uvicorn main:app --port 8000) &
(cd frontend && npm run dev) &

echo ""
echo "backend  -> http://localhost:8000"
echo "frontend -> http://localhost:3000"
echo "ctrl-c stops both"
wait
