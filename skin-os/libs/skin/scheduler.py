import asyncio, time
from collections import deque


class ViscoElasticQueue:
    def __init__(self, base_capacity: int = 8, tau: float = 4.0):
        self.base = base_capacity  # elastic baseline
        self.tau = tau  # relaxation window seconds
        self._events = deque(maxlen=256)
        self._q = asyncio.Queue()

    def _rate(self, now: float) -> float:
        self._events.append(now)
        window = [t for t in self._events if now - t <= self.tau]
        return len(window) / max(1e-6, self.tau)

    def capacity(self, rate: float) -> int:
        # fast spikes stiffen capacity, slow trickles relax
        stiff = max(1, self.base // 4)
        soft = max(1, self.base * 2)
        x = min(1.0, rate / (self.base * 2 / max(0.1, self.tau)))
        return int(soft * (1 - x) + stiff * x)

    async def put(self, item):
        now = time.time()
        r = self._rate(now)
        cap = self.capacity(r)
        while self._q.qsize() >= cap:
            await asyncio.sleep(0.008)  # creep under load
        await self._q.put(item)

    async def get(self):
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()
