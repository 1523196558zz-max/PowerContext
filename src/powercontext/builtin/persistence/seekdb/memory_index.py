# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""seekdb-native one-statement hybrid Memory candidate retrieval."""

from __future__ import annotations

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex, memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_HYBRID_SQL = """
SELECT
    candidates.memory_artifact_id,
    candidates.head_revision,
    candidates.entry_id,
    candidates.entry_version_id,
    candidates.text,
    candidates.distance,
    candidates.channel,
    candidates.channel_score
FROM (
    SELECT * FROM (
        SELECT
            f.memory_artifact_id,
            f.head_revision,
            f.entry_id,
            f.entry_version_id,
            fv.text,
            CAST(NULL AS DOUBLE) AS distance,
            'fts' AS channel,
            MATCH(f.searchable_text) AGAINST (:analyzed_query) AS channel_score
        FROM pc_memory_entry_heads AS f
        JOIN pc_memory_entry_versions AS fv
          ON fv.scope_id = f.scope_id
         AND fv.memory_artifact_id = f.memory_artifact_id
         AND fv.entry_version_id = f.entry_version_id
        WHERE f.scope_id = :scope_id
          AND f.memory_artifact_id IN :memory_artifact_ids
          AND MATCH(f.searchable_text) AGAINST (:analyzed_query)
          {fts_tag_filter}
        ORDER BY channel_score DESC,
                 f.memory_artifact_id,
                 f.entry_id,
                 f.entry_version_id
        LIMIT :candidate_limit
    ) AS fts_candidates
    UNION ALL
    SELECT * FROM (
        SELECT
            m.memory_artifact_id,
            m.head_revision,
            m.entry_id,
            m.entry_version_id,
            mv.text,
            l2_distance(m.embedding, :query_vector) AS distance,
            'vector' AS channel,
            CAST(NULL AS DOUBLE) AS channel_score
        FROM pc_memory_vector_entries AS m
        JOIN pc_memory_entry_versions AS mv
          ON mv.scope_id = m.scope_id
         AND mv.memory_artifact_id = m.memory_artifact_id
         AND mv.entry_version_id = m.entry_version_id
        WHERE m.scope_id = :scope_id
          AND m.memory_artifact_id IN :memory_artifact_ids
          {vector_tag_filter}
        ORDER BY l2_distance(m.embedding, :query_vector) {approximate}
        LIMIT :candidate_limit
    ) AS vector_candidates
) AS candidates
"""


class SeekDBMemoryIndex(CompositeMemoryIndex):
    """Keep single-channel behavior while issuing hybrid candidates in one SQL."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile
        super().__init__(OceanBaseMemoryFTSIndex(), OceanBaseMemoryVectorIndex(profile))
        self._hybrid = self._statement(tagged=False)
        self._tagged_hybrid = self._statement(tagged=True)

    def _statement(self, *, tagged: bool):
        sql = _HYBRID_SQL.format(
            fts_tag_filter=memory_tag_sql("f") if tagged else "",
            vector_tag_filter=memory_tag_sql("m") if tagged else "",
            approximate="" if tagged else "APPROXIMATE",
        )
        return text(sql).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self.profile.dimension)),
            *((bindparam("tag_keys", expanding=True), bindparam("tag_hashes", expanding=True)) if tagged else ()),
        )

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return await super().search(connection, scope_id, request)
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            (
                await connection.execute(
                    self._hybrid if request.tag_filter is None else self._tagged_hybrid,
                    {
                        "scope_id": scope_id,
                        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                        "analyzed_query": request.analyzed_query,
                        "query_vector": validate_embedding(request.query_vector, dimension=self.profile.dimension),
                        "candidate_limit": request.candidate_limit,
                        **memory_tag_parameters(request.tag_filter),
                    },
                )
            )
            .mappings()
            .all()
        )
        fts_rows = sorted(
            (row for row in rows if row["channel"] == "fts"),
            key=lambda row: (
                -float(row["channel_score"]),
                str(row["memory_artifact_id"]),
                str(row["entry_id"]),
                str(row["entry_version_id"]),
            ),
        )
        vector_rows = sorted(
            (row for row in rows if row["channel"] == "vector"),
            key=lambda row: (
                float(row["distance"]),
                str(row["memory_artifact_id"]),
                str(row["entry_id"]),
                str(row["entry_version_id"]),
            ),
        )
        return MemorySearchChannels(
            fts=memory_channel_hits(fts_rows, request.memories),
            vector=memory_channel_hits(vector_rows, request.memories),
        )


__all__ = ["SeekDBMemoryIndex"]
