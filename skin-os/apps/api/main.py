from fastapi import FastAPI, Response
from pydantic import BaseModel
import asyncio, httpx
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from libs.skin.pipeline import Packet, run_skin
from libs.skin.scheduler import ViscoElasticQueue
from libs.skin.perfusion import Channel, perfusion_controller
from config.settings import JULIA_BASE, CHOPPY_BASE, BASE_CAPACITY, TAU_SECONDS


app = FastAPI(title="skin-OS API", version="0.1.0")

# --- global workers & channels ---
veq = ViscoElasticQueue(base_capacity=BASE_CAPACITY, tau=TAU_SECONDS)


async def choppy_enrich(pkt: Packet):
    try:
        async with httpx.AsyncClient(timeout=2.0) as cl:
            r = await cl.post(f"{CHOPPY_BASE}/api/transform", json={
                "text": pkt.content,
                "options": {
                    "overlap": 0.15,
                    "entropy": True,
                    "normalize": True,
                },
            })
        if r.status_code == 200:
            data = r.json()
            pkt.meta["choppy"] = {
                "chunks": len(data.get("chunks", [])),
                "avg_entropy": data.get("metrics", {}).get("avg_entropy"),
            }
            pkt.hydration = min(1.0, pkt.hydration + 0.4)
            return pkt
    except Exception as e:
        pkt.meta["choppy_error"] = str(e)
    # graceful degrade
    await asyncio.sleep(0.01)
    pkt.hydration = min(1.0, pkt.hydration + 0.1)
    return pkt


enrich = Channel("enrich", choppy_enrich)
final = Channel("final", lambda x: asyncio.sleep(0.002, result=x))

ctrl = {"inflow_hz": 5}


async def vascular_loop():
    asyncio.create_task(enrich.pump(final))
    while True:
        perfusion_controller(ctrl, enrich, final)
        # ask Julia for override (optional)
        try:
            async with httpx.AsyncClient(timeout=0.3) as cl:
                r = await cl.get(f"{JULIA_BASE}/control", params={"avg": ctrl["inflow_hz"]})
                if r.status_code == 200:
                    data = r.json()
                    ctrl["inflow_hz"] = int(data.get("inflow_hz", ctrl["inflow_hz"]))
        except Exception:
            pass
        await asyncio.sleep(0.25)


@app.on_event("startup")
async def boot():
    asyncio.create_task(vascular_loop())
    asyncio.create_task(worker_loop())


class Ingest(BaseModel):
    content: str
    pigment: float = 0.0


@app.post("/ingest")
async def ingest(inp: Ingest):
    pkt = Packet(content=inp.content, meta={}, hydration=0.0, pigment=inp.pigment)
    pkt = run_skin(pkt)
    if pkt.meta.get("rejected"):
        return {"status": "rejected", "layer": pkt.meta["rejected"]}
    await veq.put(pkt)
    return {"status": "queued", "hydration": pkt.hydration, "pigment": pkt.pigment}


@app.get("/health")
async def health():
    return {"queue": veq.qsize(), "inflow_hz": ctrl["inflow_hz"]}


@app.get("/metrics")
async def metrics():
    # Prometheus exposition format
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def worker_loop():
    # strain‑rate aware intake → route to vascular network
    while True:
        pkt = await veq.get()
        # spawn enrich if basale requested
        for job in pkt.meta.get("spawn", []):
            if job == "enrich":
                await enrich.q.put(pkt)
        await asyncio.sleep(max(0.001, 1.0 / ctrl["inflow_hz"]))
