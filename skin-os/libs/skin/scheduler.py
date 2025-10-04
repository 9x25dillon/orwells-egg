import asyncio, time
from collections import deque
from prometheus_client import Histogram, Gauge


# Prometheus metrics
QUEUE_WAIT = Histogram(
    'skin_os_queue_wait_seconds', 'Time items wait due to stiffness limits')
QUEUE_SIZE = Gauge('skin_os_queue_size', 'Current visco queue size')
CAPACITY = Gauge('skin_os_queue_capacity', 'Current dynamic capacity')


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
        cap = int(soft * (1 - x) + stiff * x)
        CAPACITY.set(cap)
        return cap

    async def put(self, item):
        now = time.time()
        r = self._rate(now)
        cap = self.capacity(r)
        start = time.time()
        while self._q.qsize() >= cap:
            await asyncio.sleep(0.008)  # creep under load
        QUEUE_WAIT.observe(time.time() - start)
        await self._q.put(item)
        QUEUE_SIZE.set(self._q.qsize())

    async def get(self):
        item = await self._q.get()
        QUEUE_SIZE.set(self._q.qsize())
        return item

    def qsize(self) -> int:
        return self._q.qsize()
