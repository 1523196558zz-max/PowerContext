"""PDF retrieval across documents and arbitrary Memory inventory ordering."""

import io
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceDefinitionRegistry, manifest_for_definition, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


class PagePipeline:
    async def extract(self, request, /):
        return tuple(
            MemoryEntryInput(
                kind="source-evidence",
                sources=(source,),
                text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
            )
            for source in request.sources
        )


@pytest.mark.parametrize("algorithm", ["AES-128", "AES-256"])
def test_empty_password_aes_pdf(algorithm):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.encrypt(user_password="", owner_password="owner", algorithm=algorithm)  # noqa: S106
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = parse_pdf_document(PDFDocumentInput(document_id="permissions", content=buffer.getvalue()))
    assert len(pages) == 1 and pages[0].text == ""


async def test_document_selection_and_inventory_order(tmp_path):
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = []
    for document, text in (
        ("ACME_2023_10K", "Operating income was 125 million dollars."),
        ("OTHER_2023_10K", "Operating income " * 20),
        ("ACME_2022_10K", "Operating income " * 20),
    ):
        for page in range(1, 26):
            source = await registry.resolve(
                PDFPageInput(
                    document_id=document,
                    document_sha256="a" * 64,
                    page_number=page,
                    text=text if page == 13 or document != "ACME_2023_10K" else "General information.",
                )
            )
            observations.append(project_source_for_transport(registry, source))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pages.db'}"))
    async with open_builtin_runtime(config, candidate_pipeline=PagePipeline()) as runtime:
        descriptor = await runtime.scopes.create(ScopeDraft(title="Pages", summary="Reports", idempotency_key="pages"))
        scope = descriptor.scope_id
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for observation in observations:
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope, observation=observation))
        for _ in range(len(observations) + 1):
            if not (await runtime.memory.for_scope(scope).flush()).processed:
                break
        else:
            pytest.fail("Page ingestion did not drain")
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope)
        question = "What was Acme's operating income in 2023?"
        result = await search_pdf_pages(memory, question, limit=5)
        first = await memory.get(GetMemoryEntryRequest(citation=result[0]))
        assert first.entry.text == "ACME_2023_10K\nOperating income was 125 million dollars."

        class ReorderedMemory:
            search = memory.search

            async def list(self):
                entries = (await memory.list()).entries
                return SimpleNamespace(entries=tuple(reversed(entries)))

        assert await search_pdf_pages(ReorderedMemory(), question, limit=5) == result
