# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Concurrent embedding request coalescing and bounded successful-result caching."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _ResolvedEmbedding:
    vector: EmbeddingVector
    batch_id: object
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share in-flight embeddings and retain successful vectors in an LRU cache."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be a finite non-negative number")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")

        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._in_flight: dict[str, asyncio.Future[_ResolvedEmbedding]] = {}
        self._pending: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()
        self._dispatcher: asyncio.Task[None] | None = None
        self._batch_tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_ResolvedEmbedding]] = {}
        async with self._lock:
            for text in dict.fromkeys(texts):
                vector = self._cache.get(text)
                if vector is not None:
                    self._cache.move_to_end(text)
                    cached[text] = vector
                    continue
                future = self._in_flight.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    # Retrieve background failures even if every caller is cancelled.
                    future.add_done_callback(_consume_unobserved_exception)
                    self._in_flight[text] = future
                    self._pending[text] = None
                waiting[text] = future
            if self._pending and (self._dispatcher is None or self._dispatcher.done()):
                self._dispatcher = asyncio.create_task(self._dispatch_pending())

        resolved: dict[str, _ResolvedEmbedding] = {}
        for text, future in waiting.items():
            # A caller owns only its wait, never the shared provider operation.
            resolved[text] = await asyncio.shield(future)

        batch_usages: dict[int, InferenceUsage] = {}
        for item in resolved.values():
            batch_usages.setdefault(id(item.batch_id), item.usage)
        usage = _sum_usage(tuple(batch_usages.values()))
        vectors = tuple(cached[text] if text in cached else resolved[text].vector for text in texts)
        return EmbeddingResult(vectors=vectors, usage=usage)

    async def _dispatch_pending(self) -> None:
        # sleep(0) is intentional: coalesce_ms=0 still combines callers that became
        # runnable in the same event-loop turn.
        await asyncio.sleep(self._coalesce_seconds)
        while True:
            async with self._lock:
                if not self._pending:
                    self._dispatcher = None
                    return
                texts = tuple(self._pending)[: self._max_batch_size]
                for text in texts:
                    self._pending.pop(text)
                task = asyncio.create_task(self._run_batch(texts))
                self._batch_tasks.add(task)
                task.add_done_callback(self._batch_tasks.discard)

    async def _run_batch(self, texts: tuple[str, ...]) -> None:
        batch_id = object()
        try:
            result = await self._model.embed(texts)
            if len(result.vectors) != len(texts):
                raise ValueError("embedding provider returned a different number of vectors than texts")
        except BaseException as error:
            async with self._lock:
                for text in texts:
                    future = self._in_flight.pop(text, None)
                    if future is not None and not future.done():
                        future.set_exception(error)
            return

        async with self._lock:
            for text, vector in zip(texts, result.vectors, strict=True):
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                future = self._in_flight.pop(text, None)
                if future is not None and not future.done():
                    future.set_result(_ResolvedEmbedding(vector=vector, batch_id=batch_id, usage=result.usage))


def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    if not usages:
        return InferenceUsage(requests=0)

    def optional_sum(field: str) -> int | None:
        values = [getattr(usage, field) for usage in usages]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=optional_sum("input_tokens"),
        output_tokens=optional_sum("output_tokens"),
    )


def _consume_unobserved_exception(future: asyncio.Future[_ResolvedEmbedding]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
