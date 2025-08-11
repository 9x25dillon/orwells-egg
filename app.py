from fastapi import FastAPI, Body
from datetime import datetime, timedelta
import uuid

app = FastAPI(title="AA/IA minimal surface", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


# ---- AA: lease from PQ
@app.post("/pq/lease")
def lease_job(duration_minutes: int = 15):
    # In production: SELECT ... FOR UPDATE SKIP LOCKED against aalc_queue
    return {
        "job_id": str(uuid.uuid4()),
        "task_spec": {"dataset": "speech_conv_v1", "objective": "ASR"},
        "ifv_spec": {"features": ["mfcc", "prosody"]},
        "leased_until": (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat() + "Z",
    }


# ---- IA: publish RFV (well-defined feature vectors)
@app.post("/rfv/publish")
def publish_rfv(payload: dict = Body(...)):
    # payload: {name, rml_meta, rdata_uri, labels(optional)}
    # In production: insert into repo_rml/repo_rfv and return rfv_id
    return {"rfv_id": str(uuid.uuid4())}


# ---- DS: RAG/SQL + entropy filter (fast path generates prefix SQL)
@app.post("/ds/select")
def ds_select(payload: dict = Body(...)):
    # payload: {query, top_k, entropy_threshold}
    query_text = payload.get("query", "").strip()
    first_token = query_text.split()[0] if query_text else ""
    prefix = first_token.replace("'", "''")
    top_k = int(payload.get("top_k", 50))
    sql = f"SELECT * FROM corpus WHERE text ILIKE '{prefix}%' LIMIT {top_k};"
    # In production: execute/log ds_query_log and return a real result handle
    return {"sql": sql, "result_uri": "s3://bucket/ds/123.parquet"}


# ---- ML2: one training step (receives DS result handle)
@app.post("/ml2/train_step")
def ml2_train_step(payload: dict = Body(...)):
    # payload: {result_uri, rfv_id(optional), coach_cfg}
    return {"metrics": {"loss": 0.231, "wer": 10.4}, "snapshot_id": str(uuid.uuid4())}