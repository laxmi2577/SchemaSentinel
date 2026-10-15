from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncIterator, Dict, List, Optional
from pydantic import BaseModel

import src.stream_lakehouse as slh


class BackpressureBuffer:
    def __init__(self, config: Optional[slh.StreamConfig] = None) -> None:
        self.config = config or slh.StreamConfig()
        self._queue: asyncio.Queue[slh.StreamEvent] = asyncio.Queue(maxsize=self.config.buffer_maxsize)
        self._high_watermark: int = 0
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_stall_events: int = 0
        self._total_stall_duration_seconds: float = 0.0
        self._tenant_counts: Dict[str, int] = {}

    async def put(self, event: slh.StreamEvent, timeout: Optional[float] = None) -> bool:
        effective_timeout = timeout if timeout is not None else self.config.stall_timeout_seconds
        is_stalled = self._queue.full()
        t0 = time.monotonic()

        if is_stalled:
            self._total_stall_events += 1
            try:
                await asyncio.wait_for(self._queue.put(event), timeout=effective_timeout)
                duration = time.monotonic() - t0
                self._total_stall_duration_seconds += duration
            except asyncio.TimeoutError:
                duration = time.monotonic() - t0
                self._total_stall_duration_seconds += duration
                return False
        else:
            await self._queue.put(event)

        self._total_enqueued += 1
        q_len = self._queue.qsize()
        if q_len > self._high_watermark:
            self._high_watermark = q_len

        tid = event.tenant_id
        self._tenant_counts[tid] = self._tenant_counts.get(tid, 0) + 1
        return True

    async def get(self, timeout: Optional[float] = None) -> slh.StreamEvent:
        if timeout is not None:
            event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            event = await self._queue.get()
        self._total_dequeued += 1
        self._queue.task_done()
        return event

    def get_metrics(self) -> slh.QueueMetrics:
        curr_size = self._queue.qsize()
        max_sz = self.config.buffer_maxsize
        fill_ratio = (curr_size / max_sz) if max_sz > 0 else 0.0
        high_tripped = fill_ratio >= self.config.high_watermark_ratio

        return slh.QueueMetrics(
            current_size=curr_size,
            maxsize=max_sz,
            fill_ratio=round(fill_ratio, 4),
            high_watermark=self._high_watermark,
            total_enqueued=self._total_enqueued,
            total_dequeued=self._total_dequeued,
            total_stall_events=self._total_stall_events,
            total_stall_duration_seconds=round(self._total_stall_duration_seconds, 4),
            is_high_watermark_tripped=high_tripped,
            tenant_distribution=dict(self._tenant_counts),
        )

    def is_empty(self) -> bool:
        return self._queue.empty()

    def is_full(self) -> bool:
        return self._queue.full()

    def qsize(self) -> int:
        return self._queue.qsize()


class StreamProducer:
    def __init__(
        self,
        config: Optional[slh.StreamConfig] = None,
        buffer: Optional[BackpressureBuffer] = None,
    ) -> None:
        self.config = config or slh.StreamConfig()
        self.buffer = buffer or BackpressureBuffer(self.config)
        self._rng = random.Random(self.config.seed)
        self._seq: int = 0

    def generate_event(
        self,
        tenant_id: Optional[str] = None,
        event_type: str = "query_execution",
        is_valid: bool = True,
    ) -> slh.StreamEvent:
        self._seq += 1
        tid = tenant_id or f"{self.config.tenant_id_prefix}{self._rng.randint(0, self.config.num_tenants - 1)}"
        threshold = self._rng.randint(50, 500)
        table_options = ["orders", "transactions", "audit_log", "customers"]
        selected_table = self._rng.choice(table_options)
        natural_q = f"Retrieve records from {selected_table} for tenant {tid} exceeding {threshold}"

        if is_valid:
            sql = f"SELECT id, amount, status FROM {selected_table} WHERE tenant_id = '{tid}' AND amount > {threshold};"
            tables = [selected_table]
            columns = ["id", "amount", "status", "tenant_id"]
        else:
            sql = f"SELECT * FROM {selected_table} WHERE unauthorized_clause;"
            tables = [selected_table]
            columns = ["*"]

        latency = round(self._rng.uniform(1.5, 45.0), 2)
        tokens = self._rng.randint(12, 110)

        return slh.StreamEvent(
            tenant_id=tid,
            event_type=event_type,
            natural_query=natural_q,
            generated_sql=sql,
            tables_accessed=tables,
            columns_accessed=columns,
            latency_ms=latency,
            tokens_used=tokens,
            is_valid=is_valid,
            anomaly_score=round(self._rng.uniform(0.01, 0.25), 4) if is_valid else 0.95,
            metadata={"seq": self._seq, "seed": self.config.seed},
        )

    async def emit_event(
        self,
        event: Optional[slh.StreamEvent] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        evt = event if event is not None else self.generate_event()
        return await self.buffer.put(evt, timeout=timeout)

    async def emit_batch(
        self,
        events: List[slh.StreamEvent],
        timeout_per_event: Optional[float] = None,
    ) -> int:
        success_count = 0
        for evt in events:
            ok = await self.buffer.put(evt, timeout=timeout_per_event)
            if ok:
                success_count += 1
            else:
                break
        return success_count

    async def produce_continuous(
        self,
        max_events: Optional[int] = None,
        interval_seconds: float = 0.001,
    ) -> AsyncIterator[slh.StreamEvent]:
        emitted = 0
        while max_events is None or emitted < max_events:
            evt = self.generate_event()
            ok = await self.buffer.put(evt)
            if ok:
                emitted += 1
                yield evt
            if interval_seconds > 0:
                await asyncio.sleep(interval_seconds)

    def get_metrics(self) -> slh.QueueMetrics:
        return self.buffer.get_metrics()

    def get_tenant_distribution(self) -> Dict[str, int]:
        return dict(self.buffer._tenant_counts)
