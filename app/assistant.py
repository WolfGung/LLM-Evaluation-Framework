"""The support assistant: retrieve documents, ask the model, collect citations.

The system turn holds the prompt of the chosen version with the retrieved
documents, each wrapped in `<document id="kb-...">`. The customer's question
is the user turn, on its own. The answer is returned as written; an empty
answer (for example, the token budget ran out) is a result to evaluate, not an
error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.prompting import ChatModel, load_prompt, render
from app.retrieval import BM25Index, Hit, default_index
from llmeval.cassettes import CallTag
from llmeval.client import CallResult
from llmeval.config import RoleConfig

RAG_FUNCTION = "rag"
ADHOC_CASE = "adhoc"
DEFAULT_K = 3
NO_DOCUMENTS = "No documents matched the question."

_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
_DOC_ID = re.compile(r"^kb-[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class AssistantAnswer:
    text: str
    cited_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    hits: tuple[Hit, ...]
    call: CallResult


def format_documents(hits: tuple[Hit, ...]) -> str:
    if not hits:
        return NO_DOCUMENTS
    return "\n\n".join(f'<document id="{hit.doc_id}">\n{hit.text}\n</document>' for hit in hits)


def prepare(
    question: str,
    version: str,
    *,
    index: BM25Index | None = None,
    k: int = DEFAULT_K,
) -> tuple[list[dict[str, str]], tuple[Hit, ...]]:
    """The exact messages for `question` and the hits they contain.

    Pure and deterministic, so a planner can compute request keys without
    calling the model.
    """
    if not question.strip():
        raise ValueError("question is empty")
    template = load_prompt("assistant", version)
    hits = tuple((index or default_index()).search(question, k))
    system = render(template, documents=format_documents(hits))
    return [{"role": "system", "content": system}, {"role": "user", "content": question}], hits


def extract_citations(text: str) -> tuple[str, ...]:
    """Document ids cited in square brackets, in order of first appearance.

    `[kb-a]`, `[kb-a][kb-b]` and `[kb-a, kb-b]` all count. Ids are kept even if
    the search did not return them: citing a document that was not retrieved
    is something the checks look for.
    """
    cited: dict[str, None] = {}
    for inside in _BRACKETS.findall(text):
        for part in re.split(r"[,;\s]+", inside):
            if _DOC_ID.match(part):
                cited.setdefault(part, None)
    return tuple(cited)


def answer(
    client: ChatModel,
    role: RoleConfig,
    question: str,
    version: str,
    *,
    index: BM25Index | None = None,
    k: int = DEFAULT_K,
    repeat: int = 0,
    case: str | None = None,
) -> AssistantAnswer:
    """Answer `question` with prompt `version` in model role `role`."""
    messages, hits = prepare(question, version, index=index, k=k)
    tag = CallTag(function=RAG_FUNCTION, case=case or ADHOC_CASE, version=version)
    call = client.complete(messages, role=role, repeat=repeat, tag=tag)
    return AssistantAnswer(
        text=call.content,
        cited_ids=extract_citations(call.content),
        retrieved_ids=tuple(hit.doc_id for hit in hits),
        hits=hits,
        call=call,
    )
