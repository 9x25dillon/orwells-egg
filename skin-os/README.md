# skin-OS

## run
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. (optional) run Julia stub: `julia julia/limps_control.jl` and set `JULIA_BASE`
4. ./scripts/run_dev.sh

## try
```bash
curl -X POST http://localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"content":"hello epidermis","pigment":0.3}'

curl http://localhost:8000/health
```

## Notes & next steps
- Add provenance→policy routing (pigment‑aware queues)
- Persist histology metrics (queue depth, latency, hydration) for plots
- Swap enrich_fn with real Choppy/LIMPS transforms
- Dockerize + compose a Julia service for real control loops
