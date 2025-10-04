#!/usr/bin/env bash
set -euo pipefail
export JULIA_BASE=${JULIA_BASE:-http://localhost:9000}
export BASE_CAPACITY=${BASE_CAPACITY:-8}
export TAU_SECONDS=${TAU_SECONDS:-4.0}
uvicorn apps.api.main:app --reload --port 8000
