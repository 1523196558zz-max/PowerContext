"""Identity-preserving batched HTTP Memory reranking."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(self, query: str, candidates: tuple[MemoryHit, ...], limit: int, /) -> MemoryRerankDecision:
        if not candidates or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(candidates):
            raise ValueError("invalid rerank bounds")
        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self.batch_size):
            batch = candidates[offset : offset + self.batch_size]
            ranks = tuple(range(offset + 1, offset + len(batch) + 1))
            response = await self.client.post(
                self.endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            try:
                payload = response.json()
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"scores"}
                    or not isinstance(payload["scores"], list)
                ):
                    raise ValueError("invalid response schema")
                found: dict[int, float] = {}
                for item in payload["scores"]:
                    if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                        raise ValueError("invalid score schema")
                    rank, score = item["rank"], item["score"]
                    if type(rank) is not int or rank not in ranks or rank in found:
                        raise ValueError("invalid rank coverage")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("invalid score")
                    found[rank] = float(score)
                if set(found) != set(ranks):
                    raise ValueError("invalid rank coverage")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("invalid reranker response") from error
            scores.update(found)
        selected = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))


__all__ = ["HTTPMemoryReranker"]
