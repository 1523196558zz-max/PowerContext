"""Offline Pi v3 branch-history evidence and incremental connector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    source_id: str
    event: PiEventInput


def _visible(record: dict[str, Any]) -> tuple[str, str] | None:
    kind = record.get("type")
    if kind in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        return (kind, summary) if isinstance(summary, str) and summary else None
    if kind == "custom_message":
        body = _content(record.get("content"))
        return ("custom", body) if body else None
    if kind != "message" or not isinstance(record.get("message"), dict):
        return None
    message = record["message"]
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "bashExecution":
        command, output = message.get("command"), message.get("output")
        if not isinstance(command, str):
            return None
        return role, f"$ {command}" + (f"\n{output}" if isinstance(output, str) and output else "")
    body = _content(message.get("content"), role=role, message=message)
    return (role, body) if body else None


def _content(value: Any, *, role: str | None = None, message: dict[str, Any] | None = None) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if isinstance(name, str) and isinstance(call_id, str):
                arguments = json.dumps(
                    block.get("arguments", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                parts.append(f"[tool_call {name} {call_id}] {arguments}")
    body = "\n".join(parts)
    if role == "toolResult" and message is not None and body:
        return f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {body}"
    return body


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    try:
        lines = [line for line in value.content.decode("utf-8-sig").splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Pi session") from error
    if not records or not isinstance(records[0], dict):
        raise ValueError("missing Pi header")
    header = records[0]
    if (
        header.get("type") != "session"
        or header.get("version") != 3
        or not isinstance(header.get("id"), str)
        or not header["id"]
    ):
        raise ValueError("invalid Pi header")
    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("invalid Pi event")
        event_id, parent = record["id"], record.get("parentId")
        if event_id in nodes or (parent is not None and (not isinstance(parent, str) or parent not in nodes)):
            raise ValueError("invalid Pi graph")
        nodes[event_id] = record
        order.append(event_id)
    if not order:
        return ()
    leaf = value.leaf_id if value.leaf_id is not None else order[-1]
    if leaf not in nodes:
        raise ValueError("missing leaf")
    path: list[dict[str, Any]] = []
    current: str | None = leaf
    while current is not None:
        record = nodes[current]
        path.append(record)
        current = record.get("parentId")
    events: list[PiEventInput] = []
    for record in reversed(path):
        visible = _visible(record)
        if visible is None:
            continue
        role, body = visible
        events.append(
            PiEventInput(
                session_id=header["id"],
                event_id=record["id"],
                parent_id=record.get("parentId"),
                event_type=record.get("type", ""),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def _identity(event: PiEventInput) -> str:
    raw = json.dumps([event.session_id, event.event_id], ensure_ascii=False, separators=(",", ":")).encode()
    return "pi:" + hashlib.sha256(raw).hexdigest()


class _Adapter:
    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = _identity(value)
        return PiEventSource(
            name=identity, source_id=identity, materialization=SourceMaterialization.CAPTURED, event=value
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return source.event


class _Projection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        event = source.event
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=event.content,
            metadata={
                "session_id": event.session_id,
                "event_id": event.event_id,
                "parent_id": event.parent_id,
                "event_type": event.event_type,
                "role": event.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(_Adapter(), projections=(_Projection(),))


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path, self.leaf_id = Path(path), leaf_id

    async def run(self, session, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        previous = session.checkpoint if isinstance(session.checkpoint, dict) else {}
        hashes = dict(previous.get("events", {})) if isinstance(previous.get("events", {}), dict) else {}
        for event in events:
            digest = hashlib.sha256(
                json.dumps(event.record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            old = hashes.get(event.event_id)
            if old is not None and old != digest:
                raise ValueError("previously imported Pi event changed")
            if old is None:
                await session.submit(_identity(event), "pi-event", event)
                hashes[event.event_id] = digest
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint={"events": hashes})


__all__ = ["PI_EVENT_SOURCE_DEFINITION", "PiEventInput", "PiSessionConnector", "PiSessionInput", "parse_pi_session"]
