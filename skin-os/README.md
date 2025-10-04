# skin-OS

## run
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. (optional) run Julia stub: `julia julia/limps_control.jl` and set `JULIA_BASE`
4. export CHOPPY_BASE=http://localhost:9100  # your Choppy FastAPI
5. ./scripts/run_dev.sh

## try
```bash
curl -X POST http://localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"content":"hello epidermis","pigment":0.3}'

curl http://localhost:8000/health
curl http://localhost:8000/metrics   # Prometheus exposition
curl http://localhost:8000/entropy/stats
curl http://localhost:8000/entropy/graph

## metrics (Prometheus)
- `skin_os_queue_wait_seconds` (histogram)
- `skin_os_queue_size` (gauge)
- `skin_os_queue_capacity` (gauge)
- `skin_os_channel_latency_seconds{channel=...}` (histogram)
- `skin_os_inflow_hz` (gauge)
- `skin_os_entropy_delta` (histogram)
- `skin_os_entropy_last` (gauge)
- `skin_os_entropy_tokens_total` (counter)

## wiring your Entropy Engine
Place your `entropy_engine.py` under `libs/entropy/` or install it on PYTHONPATH.
The adapter auto-builds a demo graph if none is provided and still computes SHA-based entropy.
```

## Notes & next steps
- Add provenance→policy routing (pigment‑aware queues)
- Persist histology metrics (queue depth, latency, hydration) for plots
- Swap enrich_fn with real Choppy/LIMPS transforms
- Dockerize + compose a Julia service for real control loops
