from fastapi import FastAPI
from pydantic import BaseModel
import asyncio, httpx

from libs.skin.pipeline import Packet, run_skin
from libs.skin.scheduler import ViscoElasticQueue
from libs.skin.perfusion import Channel, perfusion_controller
from config.settings import JULIA_BASE, BASE_CAPACITY, TAU_SECONDS


app = FastAPI(title="skin-OS API", version="0.1.0")

# --- global workers & channels ---
veq = ViscoElasticQueue(base_capacity=BASE_CAPACITY, tau=TAU_SECONDS)


async def enrich_fn(pkt: Packet):
    await asyncio.sleep(0.01)
    pkt.hydration = min(1.0, pkt.hydration + 0.2)
    return pkt


enrich = Channel("enrich", enrich_fn)
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


async def worker_loop():
    while True:
        pkt = await veq.get()
        await enrich.q.put(pkt)
        await asyncio.sleep(max(0.001, 1.0 / ctrl["inflow_hz"]))
