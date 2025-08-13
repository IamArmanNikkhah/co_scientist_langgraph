from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional


class AsyncTaskQueue:
    """
    Minimal async task queue with a fixed-size worker pool.

    Each submitted job is a zero-argument coroutine function returning a partial state dict.
    Results are collected in submission order once all tasks complete.
    """

    def __init__(self, worker_count: int = 2):
        self._queue: asyncio.Queue[Callable[[], Awaitable[dict]]] = asyncio.Queue()
        self._results: List[Optional[dict]] = []
        self._worker_count = max(1, int(worker_count))

    def submit(self, job: Callable[[], Awaitable[dict]]) -> None:
        self._results.append(None)
        index = len(self._results) - 1

        async def wrapped() -> dict:
            result = await job()
            self._results[index] = result
            return result

        self._queue.put_nowait(wrapped)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await job()
            finally:
                self._queue.task_done()

    async def run_until_empty(self) -> List[dict]:
        workers = [asyncio.create_task(self._worker()) for _ in range(self._worker_count)]
        await self._queue.join()
        for w in workers:
            w.cancel()
        # Drain cancellations to avoid warnings
        _ = await asyncio.gather(*workers, return_exceptions=True)
        return [r or {} for r in self._results]


