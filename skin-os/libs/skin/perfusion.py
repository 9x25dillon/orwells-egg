import asyncio
from typing import Callable, Any, Optional, Dict, List
from statistics import mean


class Channel:
    def __init__(self, name: str, fn: Callable[[Any], Any]):
        self.name, self.fn = name, fn
        self.q: asyncio.Queue = asyncio.Queue()
        self.lat_hist: List[float] = []

    async def pump(self, downstream: Optional["Channel"] = None):
        while True:
            item = await self.q.get()
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            out = await self.fn(item)
            dt = loop.time() - t0
            self.lat_hist.append(dt)
            if downstream:
                await downstream.q.put(out)


def perfusion_controller(ctrl: Dict[str, float], *channels: "Channel") -> None:
    # Reduce inflow when rolling latency rises ("ischemia")
    recent = []
    for c in channels:
        sample = c.lat_hist[-10:]
        recent.append(mean(sample) if sample else 0.01)
    avg = mean(recent) if recent else 0.01
    ctrl["inflow_hz"] = max(1, int(20 / (1 + 10 * avg)))
