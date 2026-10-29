"""Complete offline PDF page evidence and scoped Memory retrieval."""

from __future__ import annotations

import hashlib
import io
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")
        return value


class PDFPageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: str
    document_sha256: str
    page_number: int = Field(gt=0)
    text: str


class PDFPageSource(Source):
    source_id: str
    page: PDFPageInput


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    if not value.content:
        raise ValueError("empty PDF")
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("encrypted PDF")
        extracted = tuple((page.extract_text() or "") for page in reader.pages)
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("invalid PDF") from error
    if not extracted:
        raise ValueError("PDF has no pages")
    digest = hashlib.sha256(value.content).hexdigest()
    return tuple(
        PDFPageInput(document_id=value.document_id, document_sha256=digest, page_number=index, text=text)
        for index, text in enumerate(extracted, start=1)
    )


def _identity(page: PDFPageInput) -> str:
    raw = json.dumps(
        [page.document_id, page.document_sha256, page.page_number], ensure_ascii=False, separators=(",", ":")
    ).encode()
    return "pdf:" + hashlib.sha256(raw).hexdigest()


class _Adapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = _identity(value)
        return PDFPageSource(
            name=identity, source_id=identity, materialization=SourceMaterialization.CAPTURED, page=value
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return source.page


class _Projection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.source_id,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(_Adapter(), projections=(_Projection(),))


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit")
    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=limit))
    return tuple(
        MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
        for hit in page.hits[:limit]
    )


__all__ = ["PDF_PAGE_SOURCE_DEFINITION", "PDFDocumentInput", "PDFPageInput", "parse_pdf_document", "search_pdf_pages"]
