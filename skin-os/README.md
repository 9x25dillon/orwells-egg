# skin-OS

## run
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. (optional) run Julia PolyServe (your file):
   - `PORT=9000 julia polyserve.jl`  # the file you pasted
   - or `julia julia/limps_control.jl` for the tiny stub
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

# Julia features (proxy → your Julia service)
curl -X POST http://localhost:8000/qvnm/estimate_id -H 'content-type: application/json' \
  -d '{"d":4,"N":3,"V":[0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8, 0.9,1.0,1.1,1.2]}'
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
- `skin_os_julia_rpc_seconds{route=...}` (histogram)

## wiring your Entropy Engine
Place your `entropy_engine.py` under `libs/entropy/` or install it on PYTHONPATH.
The adapter auto-builds a demo graph if none is provided and still computes SHA-based entropy.
```

## Notes & next steps
- Add provenance→policy routing (pigment‑aware queues)
- Persist histology metrics (queue depth, latency, hydration) for plots
- Swap enrich_fn with real Choppy/LIMPS transforms
- Dockerize + compose a Julia service for real control loops
