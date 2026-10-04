"""The knowledge base and the BM25 search over it.

Each document in `app/kb/` is a markdown file with a small YAML front matter:

    ---
    id: kb-returns          # must equal the file name
    title: Returns
    visibility: public      # optional: public (default) or internal
    ---
    Body text.

Every document is indexed, including the internal notes and the supplier
bulletin. Those two are traps for the safety layer: the assistant has to
cope with them when they are retrieved.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

KB_DIR = Path(__file__).resolve().parent / "kb"

Visibility = Literal["public", "internal"]

_ID = re.compile(r"^kb-[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_KEYS = {"id", "title", "visibility"}


class KnowledgeBaseError(ValueError):
    """A knowledge-base file is missing or malformed."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    visibility: Visibility = "public"

    @property
    def text(self) -> str:
        """What is indexed and what the assistant sees: the title, then the body."""
        return f"{self.title}\n\n{self.body}"


def parse_document(source: str, *, name: str) -> Document:
    """Parse one knowledge-base file; `name` is its file name without `.md`."""
    match = _FRONT_MATTER.match(source.replace("\r\n", "\n"))
    if match is None:
        raise KnowledgeBaseError(f"{name}: no front matter between '---' lines")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise KnowledgeBaseError(f"{name}: front matter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise KnowledgeBaseError(f"{name}: front matter is not a mapping")
    if unknown := sorted(set(meta) - _KEYS):
        raise KnowledgeBaseError(f"{name}: unknown front matter key: {', '.join(unknown)}")

    doc_id = meta.get("id")
    if not isinstance(doc_id, str) or not _ID.match(doc_id):
        raise KnowledgeBaseError(f"{name}: id must look like kb-something, got {doc_id!r}")
    if doc_id != name:
        raise KnowledgeBaseError(f"{name}: id {doc_id} does not match the file name")
    title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        raise KnowledgeBaseError(f"{name}: title is missing")
    visibility = meta.get("visibility", "public")
    if visibility not in ("public", "internal"):
        raise KnowledgeBaseError(f"{name}: visibility must be public or internal")
    body = match.group(2).strip()
    if not body:
        raise KnowledgeBaseError(f"{name}: body is empty")
    return Document(doc_id=doc_id, title=title.strip(), body=body, visibility=visibility)


def load_kb(directory: Path | str = KB_DIR) -> tuple[Document, ...]:
    """All documents in `directory`, sorted by id."""
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise KnowledgeBaseError(f"no documents in {directory}")
    return tuple(parse_document(path.read_text(encoding="utf-8"), name=path.stem) for path in paths)


# Common English words that carry no meaning for search. Negations ("no",
# "not") are kept on purpose.
_STOP_WORDS_TEXT = """
a about am an and any are as at be been but by can could did do does for from
had has have how i if in into is it its me my of on or our please should so
than that the their them then there these they this to us was we were what
when where which who why will with would you your
"""
STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())
_WORD = re.compile(r"[a-z0-9]+")


def _fold_plural(token: str) -> str:
    """A deliberately small plural folder: enough for "returns" to match "return"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, stop words removed, simple plurals folded."""
    return [_fold_plural(word) for word in _WORD.findall(text.lower()) if word not in STOP_WORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    text: str


class BM25Index:
    """Okapi BM25 over the tokenised title and body of each document.

    score(q, d) = sum over the distinct terms t of q of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)). This idf (the one
    Lucene uses) stays positive for a term found in most documents.
    """

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("no documents to index")
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Document] = {}
        for document in documents:
            if document.doc_id in self._docs:
                raise ValueError(f"document {document.doc_id} is indexed twice")
            self._docs[document.doc_id] = document
        self._tf = {doc_id: Counter(tokenize(d.text)) for doc_id, d in self._docs.items()}
        self._length = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avgdl = sum(self._length.values()) / len(self._docs)
        df = Counter(term for tf in self._tf.values() for term in tf)
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._docs)

    def get(self, doc_id: str) -> Document:
        return self._docs[doc_id]

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """The best `k` documents for `query`, best first; ties go by document id.

        A document that shares no term with the query is never returned, so a
        query can get fewer than `k` hits, or none.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        scored = []
        for doc_id, tf in self._tf.items():
            norm = self.k1 * (1 - self.b + self.b * self._length[doc_id] / self._avgdl)
            score = sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((-score, doc_id))
        scored.sort()
        return [
            Hit(doc_id=doc_id, score=-neg, text=self._docs[doc_id].text)
            for neg, doc_id in scored[:k]
        ]


@cache
def default_index() -> BM25Index:
    """The index over `app/kb/`, built on first use."""
    return BM25Index(load_kb())


def search(query: str, k: int = 3) -> list[Hit]:
    """Search the Toolshop knowledge base."""
    return default_index().search(query, k)
