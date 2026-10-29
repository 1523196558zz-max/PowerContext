# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persisted source-dependency lifecycle propagation for built-in Runtime scopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory import MemoryEntryVersion
    from powercontext.builtin.artifacts.topic_memory import TopicMemorySearchResult
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """Reject an invalid registration, event, or lifecycle resolution."""


class SourceLifecycleEvent(BaseModel):
    """One versioned availability event for a registered source root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: Annotated[StrictInt, Field(gt=0)]
    sequence: Annotated[StrictInt, Field(gt=0)]
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...]
    policy: Literal["all", "any"] | None
    revision: int | None
    available: bool
    event_sequence: int
    event_payload: bytes | None


class SourceLifecycleApplication:
    """Select a persisted lifecycle service for one Runtime scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, self._database, scope_id)

    async def filter_topic_result(self, scope_id: str, result: TopicMemorySearchResult) -> TopicMemorySearchResult:
        if not result.hits:
            return result
        async with self._database.transaction() as connection:
            rows = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE.c.reference).where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.kind == "topic",
                        SOURCE_LIFECYCLE_NODES_TABLE.c.available.is_(False),
                    )
                )
            ).scalars()
            inactive = {
                (reference.artifact_id, reference.revision)
                for payload in rows
                for reference in (ArtifactRef.model_validate_json(payload),)
            }
        if not inactive:
            return result
        return result.model_copy(
            update={
                "hits": tuple(
                    hit
                    for hit in result.hits
                    if (hit.artifact_ref.artifact_id, hit.artifact_ref.revision) not in inactive
                )
            }
        )


class ScopedSourceLifecycleApplication:
    """Register and propagate an immutable dependency graph in one scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int) -> None:
        self._validate_node_id(node_id)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
            raise SourceLifecycleError("revision")
        await self._runtime.records.for_scope(self.scope_id).get_source(reference.source_type, reference.source_id)
        await self._register(node_id, "source", reference, (), None, revision)

    async def register_memory(
        self,
        node_id: str,
        reference: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        self._validate_node_id(node_id)
        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
        await self._register(node_id, "memory", reference, parents, policy, None)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        self._validate_node_id(node_id)
        if reference.family != "topic-memory":
            raise SourceLifecycleError("topic-reference")
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            reference.family, reference.artifact_id, reference.revision
        )
        await self._register(node_id, "topic", reference, parents, policy, None)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            root = nodes.get(event.node_id)
            if root is None or root.kind != "source" or root.revision != event.revision:
                raise SourceLifecycleError("source-event")
            payload = event.model_dump_json().encode()
            if event.sequence < root.event_sequence:
                raise SourceLifecycleError("event-sequence")
            if event.sequence == root.event_sequence:
                if root.event_payload != payload:
                    raise SourceLifecycleError("event-conflict")
                return sorted(node.node_id for node in nodes.values() if node.available)

            desired = self._desired(nodes, event.node_id, event.available)
            await self._transition_memories(nodes, desired, event.reason)
            async with self._database.transaction() as connection:
                for node_id, available in desired.items():
                    if nodes[node_id].available != available:
                        await connection.execute(
                            update(SOURCE_LIFECYCLE_NODES_TABLE)
                            .where(
                                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                            )
                            .values(available=available)
                        )
                await connection.execute(
                    update(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == event.node_id,
                    )
                    .values(event_sequence=event.sequence, event_payload=payload)
                )
            return sorted(node_id for node_id, available in desired.items() if available)

    async def resolve(
        self, node_id: str, /, *, purpose: Literal["generation", "audit"]
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose")
        node = (await self._load()).get(node_id)
        if node is None or (purpose == "generation" and not node.available):
            raise SourceLifecycleError("unavailable")
        return node.reference

    async def _register(
        self,
        node_id: str,
        kind: Literal["source", "memory", "topic"],
        reference: SourceRef | MemoryCitation | ArtifactRef,
        parents: tuple[str, ...],
        policy: Literal["all", "any"] | None,
        revision: int | None,
    ) -> None:
        if kind != "source":
            if policy not in {"all", "any"} or not parents or len(set(parents)) != len(parents):
                raise SourceLifecycleError("parents")
            if node_id in parents:
                raise SourceLifecycleError("cycle")
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            available = True
            if kind != "source":
                if any(parent not in nodes for parent in parents):
                    raise SourceLifecycleError("missing-parent")
                states = [nodes[parent].available for parent in parents]
                available = all(states) if policy == "all" else any(states)
            expected = _Node(node_id, kind, reference, parents, policy, revision, available, 0, None)
            existing = nodes.get(node_id)
            if existing is not None:
                if (
                    existing.node_id,
                    existing.kind,
                    existing.reference,
                    existing.parents,
                    existing.policy,
                    existing.revision,
                ) != (
                    expected.node_id,
                    expected.kind,
                    expected.reference,
                    expected.parents,
                    expected.policy,
                    expected.revision,
                ):
                    raise SourceLifecycleError("registration-conflict")
                return
            if kind == "memory" and not available:
                await self._transition_memory(reference, False, "registered without active support")
            async with self._database.transaction() as connection:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                        scope_id=self.scope_id,
                        node_id=node_id,
                        kind=kind,
                        reference=reference.model_dump_json().encode(),
                        parents=json.dumps(parents, ensure_ascii=False, separators=(",", ":")).encode(),
                        policy=policy,
                        revision=revision,
                        available=available,
                        event_sequence=0,
                        event_payload=None,
                    )
                )

    async def _load(self) -> dict[str, _Node]:
        async with self._database.transaction() as connection:
            rows = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id)
                )
            ).mappings()
            return {row["node_id"]: self._decode(row) for row in rows}

    @staticmethod
    def _decode(row) -> _Node:
        kind = row["kind"]
        model = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[kind]
        return _Node(
            node_id=row["node_id"],
            kind=kind,
            reference=model.model_validate_json(row["reference"]),
            parents=tuple(json.loads(row["parents"])),
            policy=row["policy"],
            revision=row["revision"],
            available=bool(row["available"]),
            event_sequence=row["event_sequence"],
            event_payload=row["event_payload"],
        )

    @staticmethod
    def _desired(nodes: dict[str, _Node], changed: str, available: bool) -> dict[str, bool]:
        desired = {
            node_id: (available if node_id == changed else node.available)
            for node_id, node in nodes.items()
            if node.kind == "source"
        }
        pending = {node_id for node_id, node in nodes.items() if node.kind != "source"}
        while pending:
            progressed = False
            for node_id in tuple(pending):
                node = nodes[node_id]
                if all(parent in desired for parent in node.parents):
                    states = [desired[parent] for parent in node.parents]
                    desired[node_id] = all(states) if node.policy == "all" else any(states)
                    pending.remove(node_id)
                    progressed = True
            if not progressed:
                raise SourceLifecycleError("dependency-cycle")
        return desired

    async def _transition_memories(self, nodes: dict[str, _Node], desired: dict[str, bool], reason: str) -> None:
        changes = [node for node in nodes.values() if node.kind == "memory" and node.available != desired[node.node_id]]
        if not changes:
            return
        target = desired[changes[0].node_id]
        if any(desired[node.node_id] != target for node in changes):
            raise SourceLifecycleError("mixed-transition")
        references = [node.reference for node in changes]
        await self._transition_memory_entries(references, target, reason)

    async def _transition_memory(self, reference: MemoryCitation, available: bool, reason: str) -> None:
        await self._transition_memory_entries([reference], available, reason)

    async def _transition_memory_entries(
        self, references: list[MemoryCitation | SourceRef | ArtifactRef], available: bool, reason: str
    ) -> None:
        citations = [reference for reference in references if isinstance(reference, MemoryCitation)]
        if not citations:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            versions = {entry.entry_id: entry for entry in await service.entries(current)}
            states = {entry.entry_id: entry.state for entry in current.content.manifest.entries}
            targets: list[MemoryEntryVersion] = []
            desired_state = "active" if available else "inactive"
            for citation in citations:
                version = versions.get(citation.entry_id)
                if version is None:
                    raise SourceLifecycleError("memory-entry")
                if states.get(citation.entry_id) != desired_state:
                    targets.append(version)
            if not targets:
                return
            if available:
                await service.reactivate(current, entries=targets, reason=reason)
            else:
                await service.forget(current, entries=targets, reason=reason)

    @staticmethod
    def _validate_node_id(node_id: str) -> None:
        if not isinstance(node_id, str) or not node_id.strip() or node_id != node_id.strip():
            raise SourceLifecycleError("node-id")


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent"]
