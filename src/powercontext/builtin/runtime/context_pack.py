"""Portable full and prefix-delta encodings for prepared context."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass


class ContextPackError(ValueError):
    pass


_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": _encoded(items)}
    return "sha256:" + hashlib.sha256(_json(unsigned)).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
            raise ContextPackError("items")
        digest = _digest(items)
        byte_count = 0
        while True:
            payload = {"schema": _PACK_SCHEMA, "items": _encoded(items), "digest": digest, "byte_count": byte_count}
            measured = len(_json(payload))
            if measured == byte_count:
                return cls(items, digest, measured)
            byte_count = measured

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes")
        if not isinstance(content, str):
            raise ContextPackError("content")
        data = content.encode("utf-8")
        items: list[bytes] = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end > start:
                try:
                    data[start:end].decode("utf-8")
                except UnicodeDecodeError:
                    end -= 1
                else:
                    break
            if end == start:
                raise ContextPackError("chunk boundary")
            items.append(data[start:end])
            start = end
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        payload = {
            "schema": _PACK_SCHEMA,
            "items": _encoded(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        encoded = _json(payload)
        if self.digest != _digest(self.items) or self.byte_count != len(encoded):
            raise ContextPackError("invalid pack")
        return encoded

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPack:
        try:
            value = json.loads(payload)
            if not isinstance(value, dict) or set(value) != {"schema", "items", "digest", "byte_count"}:
                raise ContextPackError("pack fields")
            if value["schema"] != _PACK_SCHEMA or type(value["byte_count"]) is not int:
                raise ContextPackError("pack schema")
            items = tuple(base64.b64decode(item, validate=True) for item in value["items"])
            pack = cls(items, value["digest"], value["byte_count"])
            pack.to_bytes()
            return pack
        except ContextPackError:
            raise
        except (ValueError, TypeError, KeyError, binascii.Error, UnicodeDecodeError) as error:
            raise ContextPackError("invalid pack") from error


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items")
        if (
            not isinstance(self.base_digest, str)
            or not isinstance(self.target_digest, str)
            or not _DIGEST.fullmatch(self.base_digest)
            or not _DIGEST.fullmatch(self.target_digest)
        ):
            raise ContextPackError("digest")
        return _json({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPackDelta:
        try:
            value = json.loads(payload)
            fields = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
            if not isinstance(value, dict) or set(value) != fields or value["schema"] != _DELTA_SCHEMA:
                raise ContextPackError("delta fields")
            if type(value["prefix_items"]) is not int or value["prefix_items"] < 0:
                raise ContextPackError("prefix_items")
            if not isinstance(value["base_digest"], str) or not _DIGEST.fullmatch(value["base_digest"]):
                raise ContextPackError("base_digest")
            if not isinstance(value["target_digest"], str) or not _DIGEST.fullmatch(value["target_digest"]):
                raise ContextPackError("target_digest")
            appended = tuple(base64.b64decode(item, validate=True) for item in value["appended"])
            delta = cls(value["base_digest"], value["target_digest"], value["prefix_items"], appended)
            delta.to_bytes()
            return delta
        except ContextPackError:
            raise
        except (ValueError, TypeError, KeyError, binascii.Error, UnicodeDecodeError) as error:
            raise ContextPackError("invalid delta") from error


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if base.digest != delta.base_digest or delta.prefix_items > len(base.items):
        raise ContextPackError("base mismatch")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target mismatch")
    return target


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
