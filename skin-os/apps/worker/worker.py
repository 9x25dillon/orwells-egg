import asyncio
from libs.skin.scheduler import ViscoElasticQueue


async def main():
    q = ViscoElasticQueue()

    async def consume():
        while True:
            item = await q.get()
            print("WORK:", getattr(item, "meta", {}))

    asyncio.create_task(consume())
    for i in range(50):
        await q.put({"i": i})
        await asyncio.sleep(0.01)
    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
