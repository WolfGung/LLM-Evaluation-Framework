"""The knowledge base and the search over it.

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

import re
from dataclasses import dataclass
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
