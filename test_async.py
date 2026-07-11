import asyncio
import time
from src.llm.client import acomplete, complete

PROMPTS = [
    "Name a fruit. One word.",
    "Name a color. One word.",
    "Name an animal. One word.",
    "Name a country. One word.",
    "Name a planet. One word.",
]


def run_sequential():
    start = time.time()
    results = [complete(p) for p in PROMPTS]
    elapsed = time.time() - start
    print(f"\nSEQUENTIAL: {elapsed:.2f}s")
    return results


async def run_concurrent():
    start = time.time()
    # Fire all 5 at once, wait for all to finish together
    results = await asyncio.gather(*(acomplete(p) for p in PROMPTS))
    elapsed = time.time() - start
    print(f"CONCURRENT: {elapsed:.2f}s")
    return results


print("Running 5 prompts sequentially, then concurrently...")
run_sequential()
asyncio.run(run_concurrent())