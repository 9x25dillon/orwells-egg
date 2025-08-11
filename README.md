# Chaos RAG/SQL + ML2 minimal scaffold

## Quickstart

- Create and activate a virtualenv, then install base deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Start the API:

```bash
uvicorn app:app --reload --port 8000
```

Endpoints:
- POST `/pq/lease` — lease a job from AA PQ (stubbed)
- POST `/rfv/publish` — publish RFV metadata (stubbed)
- POST `/ds/select` — generate prefix SQL and return a stub `result_uri`
- POST `/ml2/train_step` — return stub metrics + snapshot id

## Postgres schema

Apply `sql/schema.sql` to provision minimal tables used by AA/IA/DS:

```bash
psql "$DATABASE_URL" -f sql/schema.sql
```

## Optional ML deps

If you want to run the PyTorch bits (`ml2_core.py`, `train_step.py`), install ML extras:

```bash
pip install -r requirements-ml.txt
```

## Files
- `app.py` — FastAPI surface for AA/IA/DS/ML2
- `ml2_core.py` — CompoundNode, SkipPreserveBlock, gradient normalizers
- `ds_adapter.py` — simple prefix SQL generator
- `rfv.py` — RFV snapshot helper
- `coach.py` — entropy-aware coach policy
- `train_step.py` — training step helper showing GradNormalizer usage
- `sql/schema.sql` — Postgres schema

## Docker

Build and run:

```bash
docker build -t chaos-aa-ia:latest .
docker run --rm -p 8000:8000 chaos-aa-ia:latest
```

Test:

```bash
curl http://127.0.0.1:8000/healthz
```