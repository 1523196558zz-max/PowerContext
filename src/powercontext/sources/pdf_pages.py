"""Complete offline PDF page evidence and scoped Memory retrieval."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter

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


_MEMORY_TEXT_BYTE_LIMIT = 8192
_MEMORY_TEXT_BYTE_MARGIN = 32


def _fit_text(document_id: str, text: str) -> str:
    # The projection content becomes a Memory entry body whose canonical form must
    # not exceed 8192 UTF-8 bytes.  Long exhibit-index pages exceed that bound, so
    # cap the page text (never mid-codepoint) instead of failing the whole persist.
    budget = _MEMORY_TEXT_BYTE_LIMIT - len(document_id.encode("utf-8")) - 1 - _MEMORY_TEXT_BYTE_MARGIN
    if len(text.encode("utf-8")) <= budget:
        return text
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


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
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=index,
            text=_fit_text(value.document_id, text),
        )
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
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question")

    # The Runtime's default FTS admission floor is deliberately conservative for
    # general Memory searches.  A FinanceBench question is comparatively long and
    # often names a derived metric (for example EBITDA or capex) which does not occur
    # verbatim on either of the two pages needed to answer it.  A single raw FTS call
    # therefore has poor page recall even though the correct pages are indexed.
    #
    # Keep the native search in the path, then rerank the exact current Memory entries
    # locally.  ``list`` and the citations below are public scoped-Memory operations;
    # no PDF, answer key, page label, or evaluation asset is consulted here.
    native = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(32, limit * 4)))
    entries = (await memory.list()).entries
    if not entries:
        return ()

    query_terms, phrases, section_phrases = _financial_query(question)
    document_names = [item.entry.text.partition("\n")[0] for item in entries]
    affinities = _document_affinities(document_names, question)
    documents = [Counter(_words(item.entry.text)) for item in entries]
    lengths = [sum(tokens.values()) for tokens in documents]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    frequencies = Counter(term for term in set(query_terms) for tokens in documents if term in tokens)
    native_rank = {
        (hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id): rank
        for rank, hit in enumerate(native.hits, start=1)
    }

    scored = []
    for item, tokens, length, document in zip(entries, documents, lengths, document_names, strict=True):
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            term_frequency = tokens.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = frequencies[term]
            inverse_frequency = math.log(1.0 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + 1.2 * (0.25 + 0.75 * length / max(average_length, 1.0))
            score += min(query_frequency, 2) * inverse_frequency * term_frequency * 2.2 / denominator

        normalized = " ".join(_words(item.entry.text))
        score += sum(18.0 + 2.0 * len(phrase.split()) for phrase in phrases if phrase in normalized)
        # Section anchors (statement titles, MD&A openers) are the standard
        # FinanceBench evidence pages; questions usually paraphrase them rather
        # than quoting the header, so weigh the header phrase above loose terms.
        score += sum(30.0 + 3.0 * len(phrase.split()) for phrase in section_phrases if phrase in normalized)
        identity = (item.memory_ref.artifact_id, item.entry.entry_id, item.entry.entry_version_id)
        if identity in native_rank:
            score += 1.0 / (8.0 + native_rank[identity])
        scored.append((affinities[document], score, identity, item.citation))

    # The inventory is ordered by entry identity, not physical PDF page number.
    # Never infer page adjacency from its order. Explicit report identity in the
    # question takes precedence over repeated financial terminology in other reports.
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    return tuple(item[3] for item in scored[:limit])


def _document_affinities(names: list[str], question: str) -> dict[str, float]:
    query_words = re.findall(r"[a-z0-9]+", question.casefold())
    # A report label may join the words in a company name or retain digits.
    # Compare contiguous query words as well as individual words (e.g. 3M or
    # Coca-Cola versus COCACOLA), without matching substrings inside other words.
    query = {
        "".join(query_words[start : start + width])
        for start in range(len(query_words))
        for width in range(1, min(6, len(query_words) - start) + 1)
    }
    query_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    ignored = {"annual", "report", "form", "inc", "corporation", "corp", "company", "limited", "ltd", "10k", "10q"}
    identities = {}
    for name in set(names):
        normalized = name.casefold()
        years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", normalized))
        without_dates = re.sub(r"(?:fy)?(?:19|20)\d{2}(?:q[1-4])?", " ", normalized)
        companies = {
            word
            for word in re.findall(r"[a-z0-9]+", without_dates)
            if not word.isdigit() and len(word) > 1 and word not in ignored
        }
        identities[name] = (companies, years)
    # If a document ID is opaque or no company is named, preserve content-based
    # retrieval. A year by itself is not enough to infer the intended document.
    company_matches = {name: len(company & query) / max(1, len(company)) for name, (company, _) in identities.items()}
    if not any(company_matches.values()):
        return dict.fromkeys(identities, 0.0)
    return {
        name: 2 * company_matches[name] + (bool(years & query_years) if company_matches[name] else 0)
        for name, (_, years) in identities.items()
    }


_WORD = re.compile(r"[a-z]+|\d+", re.IGNORECASE)
_QUERY_STOPWORDS = frozenset([
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "assuming",
    "at",
    "be",
    "been",
    "by",
    "can",
    "calculate",
    "could",
    "details",
    "did",
    "directly",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "shown",
    "state",
    "than",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "to",
    "use",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "within",
    "would",
    "answer",
    "question",
    "respond",
    "response",
    "perspective",
    "relying",
    "only",
    "company",
    "fy",
    "fiscal",
    "year",
    "years",
    "million",
    "millions",
    "usd",
    "reported",
    "report",
    "reports",
])


def _words(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _WORD.finditer(value))


def _financial_query(question: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    normalized_question = re.sub(r"\bcashflows?\b", "cash flow", question, flags=re.IGNORECASE)
    raw = _words(normalized_question)
    # "FY22" tokenizes as "fy" + "22"; statement pages carry four-digit years.
    mapped = [
        "20" + term
        if ("fy" in raw[max(0, index - 2) : index] or "fiscal" in raw[max(0, index - 2) : index])
        and re.fullmatch(r"(?:1[0-9]|2[0-9])", term)
        else term
        for index, term in enumerate(raw)
    ]
    terms = [term for term in mapped if term not in _QUERY_STOPWORDS and len(term) > 1]
    text = " ".join(raw)
    expansions: list[str] = []
    phrase_expansions: list[str] = []
    section_phrases: list[str] = []

    def add(words: str) -> None:
        values = words.split()
        expansions.extend(values)
        if len(values) > 1:
            phrase_expansions.append(words)

    def add_section(words: str) -> None:
        section_phrases.append(words)

    balance_sheet_anchors = (
        "consolidated balance sheets",
        "consolidated balance sheet",
        "current assets",
        "current liabilities",
    )
    income_statement_anchors = (
        "consolidated statements of operations",
        "consolidated statement of operations",
        "consolidated statement of income",
        "consolidated statements of income",
    )

    if any(
        key in text
        for key in (
            "quick ratio",
            "current ratio",
            "working capital",
            "balance sheet",
            "days payable",
            "debt to equity",
            "accounts payable",
            "accounts receivable",
        )
    ) or any(term in raw for term in ("liquidity", "inventory", "dpo", "leverage", "solvency")):
        for anchor in balance_sheet_anchors:
            add_section(anchor)
    if any(
        key in text
        for key in (
            "gross margin",
            "operating margin",
            "net margin",
            "gross profit",
            "operating income",
            "net income",
            "tax rate",
            "effective tax",
            "cost of sales",
            "cost of goods",
            "statement of income",
            "income statement",
            "earnings per share",
            "margin",
            "cogs",
            "eps",
        )
    ):
        for anchor in income_statement_anchors:
            add_section(anchor)
    if ("revenue" in raw or "sales" in raw or "net sales" in text) and any(
        word in raw
        for word in (
            "change",
            "changes",
            "drove",
            "drive",
            "driven",
            "growth",
            "grew",
            "increased",
            "increase",
            "decreased",
            "decrease",
            "declined",
            "decline",
            "improved",
            "improving",
            "yoy",
            "compare",
            "comparison",
            "drivers",
        )
    ):
        add_section("results of operations")
        add_section("management s discussion and analysis")
    if "segment" in raw or "segments" in raw or "product categories" in text or "service categories" in text:
        add_section("business segment")
        add_section("segment information")
        add_section("reportable segments")
    if any(
        key in text
        for key in (
            "primary customers",
            "primary customer",
            "main customers",
            "main customer",
            "major customers",
            "major customer",
            "principal customers",
            "principal customer",
            "customer base",
        )
    ):
        add_section("significant portion of our")
        add_section("substantial portion of our")
        add_section("limited number of")
        add_section("pursuant to u s government")
    if "concentration" in raw:
        add_section("consolidated net revenue")
        add("accounted for")
        add("of our consolidated net revenue")
    if (
        "products and services" in text
        or "major products" in text
        or "product offerings" in text
        or ("products" in raw and "services" in raw)
    ):
        add_section("primarily offering")
        add("products and services")

    if "ebitda" in raw:
        add("operating income")
        add("operating profit")
        add("depreciation amortization")
    if "capex" in raw or "capital expenditure" in text or "capital expenditures" in text:
        add("capital spending")
        add("capital expenditures")
    if "d a" in text or ("depreciation" in raw and "amortization" in raw):
        add("depreciation amortization")
    if "p l" in text or "income statement" in text or "income statements" in text:
        add("consolidated statement of income")
    if "cash flow" in text or "cash flows" in text:
        add("consolidated statement of cash flows")
        add("consolidated statements of cash flows")
        add("cash flows from operating activities")
    if "free cash" in text or "cash conversion" in text:
        add("net cash provided by operating activities")
        add("capital expenditures")
        add("net income")
    if "capital intensive" in text or "capital intensity" in text:
        add("property plant and equipment")
        add("capital expenditures")
        add("purchases of property plant and equipment")
        add("net sales")
        add("total assets")
        add_section("consolidated balance sheet")
        add_section("consolidated statements of operations")
        add_section("consolidated statement of operations")
        add_section("consolidated statement of income")
        add_section("consolidated statement of cash flows")
    if "segment" in raw or "segments" in raw:
        add("business segments")
        add("segment information")
        if "growth" in raw or "organic" in raw:
            add("organic sales")
            add("local currency")
            add("acquisitions divestitures")
    if "legal" in raw or "litigation" in raw or "battles" in raw:
        add("legal proceedings")
        add("legal proceeding")
        add("legal actions")
        add("litigation claims regulatory proceedings")
    if "geographies" in raw or "geography" in raw or "regions" in raw:
        add("our operations")
        add("reportable segments divisions")
        add("countries territories")
        add("north america latin america europe africa middle east south asia pacific australia new zealand china")
    if "restructuring" in raw:
        add("restructuring impairment charges")
    if "cogs" in raw or "dpo" in raw:
        add("cost of sales")
        add("cost of goods sold")

    return tuple(terms + expansions), tuple(dict.fromkeys(phrase_expansions)), tuple(dict.fromkeys(section_phrases))


__all__ = ["PDF_PAGE_SOURCE_DEFINITION", "PDFDocumentInput", "PDFPageInput", "parse_pdf_document", "search_pdf_pages"]
