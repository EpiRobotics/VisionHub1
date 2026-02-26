"""Job queue and GPU worker pool for VisionHub service.

Provides:
- Per-project job queues with capacity limits
- Global GPU worker pool with round-robin scheduling
- Timeout protection per job
- CPU thread pool for post-processing (save artifacts, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from app.config import ServiceConfig

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A single inference or train job."""

    job_id: str
    project_id: str
    job_type: str = "infer"  # "infer" or "train"
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float = 0.0
    completed_at: float = 0.0
    _future: asyncio.Future[dict[str, Any]] | None = field(default=None, repr=False)


class ProjectQueue:
    """Per-project job queue with capacity limit."""

    def __init__(self, project_id: str, max_size: int = 20):
        self.project_id = project_id
        self.max_size = max_size
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_size)
        self._pending_count = 0

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    async def put(self, job: Job) -> bool:
        """Try to enqueue a job. Returns False if queue is full."""
        try:
            self._queue.put_nowait(job)
            self._pending_count += 1
            return True
        except asyncio.QueueFull:
            return False

    async def get(self) -> Job | None:
        """Get next job (non-blocking). Returns None if empty."""
        try:
            job = self._queue.get_nowait()
            self._pending_count -= 1
            return job
        except asyncio.QueueEmpty:
            return None


class GpuWorkerPool:
    """
    Global GPU worker pool.

    Pulls jobs from project queues in round-robin order and executes
    them on GPU worker(s). Supports configurable number of workers
    (default 1 for stability).
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        infer_callback: Callable[[Job], Coroutine[Any, Any, dict[str, Any]]],
    ):
        self._config = service_config
        self._num_workers = service_config.gpu.workers
        self._job_timeout_ms = service_config.scheduler.job_timeout_ms
        self._poll_interval_s = service_config.scheduler.poll_interval_ms / 1000.0
        self._infer_callback = infer_callback

        self._project_queues: dict[str, ProjectQueue] = {}
        self._round_robin_keys: list[str] = []
        self._rr_index = 0

        self._running = False
        self._workers: list[asyncio.Task[None]] = []

        # CPU thread pool for post-processing
        self._cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu_post")

        self._lock = asyncio.Lock()

    @property
    def cpu_pool(self) -> ThreadPoolExecutor:
        return self._cpu_pool

    async def register_project(self, project_id: str, max_queue: int | None = None) -> ProjectQueue:
        """Register a project queue."""
        if max_queue is None:
            max_queue = self._config.scheduler.max_queue_per_project
        async with self._lock:
            if project_id not in self._project_queues:
                pq = ProjectQueue(project_id, max_size=max_queue)
                self._project_queues[project_id] = pq
                self._round_robin_keys = list(self._project_queues.keys())
                logger.info("Registered project queue: %s (max=%d)", project_id, max_queue)
            return self._project_queues[project_id]

    async def unregister_project(self, project_id: str) -> None:
        """Unregister a project queue."""
        async with self._lock:
            self._project_queues.pop(project_id, None)
            self._round_robin_keys = list(self._project_queues.keys())
            logger.info("Unregistered project queue: %s", project_id)

    def get_queue(self, project_id: str) -> ProjectQueue | None:
        return self._project_queues.get(project_id)

    async def submit_job(self, job: Job) -> asyncio.Future[dict[str, Any]]:
        """
        Submit a job to the appropriate project queue.

        Returns a Future that will be resolved when the job completes.
        Raises RuntimeError if queue is full.
        """
        pq = self._project_queues.get(job.project_id)
        if pq is None:
            raise RuntimeError(f"No queue registered for project: {job.project_id}")

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        job._future = future

        if not await pq.put(job):
            future.set_exception(RuntimeError("BUSY_QUEUE_FULL"))
            return future

        logger.debug("Job %s submitted to queue %s (size=%d)", job.job_id, job.project_id, pq.size)
        return future

    async def start(self) -> None:
        """Start GPU worker tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self._num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)
        logger.info("Started %d GPU worker(s)", self._num_workers)

    async def stop(self) -> None:
        """Stop all GPU workers."""
        self._running = False
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._cpu_pool.shutdown(wait=False)
        logger.info("Stopped GPU workers")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single GPU worker."""
        logger.info("GPU worker %d started", worker_id)
        while self._running:
            job = await self._pick_next_job()
            if job is None:
                await asyncio.sleep(self._poll_interval_s)
                continue

            job.status = JobStatus.RUNNING
            job.started_at = time.monotonic()
            logger.debug("Worker %d processing job %s (project=%s)", worker_id, job.job_id, job.project_id)

            try:
                timeout_s = self._job_timeout_ms / 1000.0
                result = await asyncio.wait_for(
                    self._infer_callback(job),
                    timeout=timeout_s,
                )
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = time.monotonic()
                if job._future and not job._future.done():
                    job._future.set_result(result)
            except asyncio.TimeoutError:
                job.status = JobStatus.TIMEOUT
                job.error = f"Job timeout after {self._job_timeout_ms}ms"
                job.completed_at = time.monotonic()
                logger.warning("Job %s timed out", job.job_id)
                if job._future and not job._future.done():
                    job._future.set_exception(TimeoutError(job.error))
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.monotonic()
                logger.exception("Job %s failed: %s", job.job_id, e)
                if job._future and not job._future.done():
                    job._future.set_exception(e)

    async def _pick_next_job(self) -> Job | None:
        """Pick the next job using round-robin across project queues."""
        async with self._lock:
            keys = self._round_robin_keys
        if not keys:
            return None

        n = len(keys)
        for _ in range(n):
            idx = self._rr_index % n
            self._rr_index += 1
            project_id = keys[idx]
            pq = self._project_queues.get(project_id)
            if pq is None:
                continue
            job = await pq.get()
            if job is not None:
                return job
        return None

    def get_all_queue_stats(self) -> dict[str, dict[str, Any]]:
        """Return queue stats for all projects."""
        stats: dict[str, dict[str, Any]] = {}
        for pid, pq in self._project_queues.items():
            stats[pid] = {
                "queue_size": pq.size,
                "max_size": pq.max_size,
            }
        return stats
