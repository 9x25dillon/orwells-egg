# Chaos RAG/SQL + ML2 minimal scaffold

## Quickstart

- Create and activate a virtualenv, then install base deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Set database and optional API key (Postgres recommended):

```bash
export DATABASE_URL="postgresql+psycopg://limps:limps@localhost:5432/limps"
# optional auth
export API_KEY="dev-key"
```

- Apply `sql/schema.sql` to provision tables:

```bash
psql "$DATABASE_URL" -f sql/schema.sql
```

- Start the API:

```bash
uvicorn app:app --reload --port 8000
```

Endpoints (pass header `x-api-key: $API_KEY` if set):
- POST `/pq/lease` — lease a job from AA PQ (DB-backed)
- POST `/rfv/publish` — publish RFV metadata (persists to repo tables)
- POST `/ds/select` — generate prefix SQL and log to `ds_query_log`
- POST `/ml2/train_step` — stub metrics + coach state update
- POST `/rfv/snapshot` — demo: create/record an RFV snapshot using a trivial model

## Optional ML deps

If you want to run the PyTorch bits (`ml2_core.py`, `train_step.py`), install ML extras:

```bash
pip install -r requirements-ml.txt
```

## Files
- `app.py` — FastAPI surface for AA/IA/DS/ML2 wired to Postgres via SQLAlchemy
- `db.py` — SQLAlchemy engine, session, and ORM models
- `ml2_core.py` — CompoundNode, SkipPreserveBlock, gradient & BPTT normalizers
- `ds_adapter.py` — simple prefix SQL generator
- `rfv.py` — RFV snapshot helper
- `coach.py` — entropy-aware coach policy
- `train_step.py` — training step helper showing GradNormalizer usage
- `sql/schema.sql` — Postgres schema

## Docker

Build and run:

```bash
docker build -t chaos-aa-ia:latest .
docker run --rm -p 8000:8000 -e DATABASE_URL="postgresql+psycopg://limps:limps@host.docker.internal:5432/limps" -e API_KEY=dev-key chaos-aa-ia:latest
```

Test:

```bash
curl -H "x-api-key: dev-key" http://127.0.0.1:8000/healthz
```