"""A synthetic OpenRouter for the estimate, status and record tests.

Synthetic data: every reply, verdict, price, quota number and header served
here is made up for the test, never a model output. It answers through an
httpx MockTransport, so no request leaves the process. Workspaces (config,
datasets, cassettes) live in pytest's `tmp_path`; nothing is written to the
repository's `cassettes/` or `results/`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# A fake key with a recognisable shape: tests assert it never shows up.
FAKE_KEY = "sk-or-v1-SYNTHETIC-not-a-real-key-0123456789abcdef"

CONFIG_YAML = """\
system:
  model: synthetic/system:free
  temperature: 0.2
  max_tokens: 200
  structured_output: false
judge:
  model: synthetic/judge:free
  temperature: 0
  seed: 7
  max_tokens: 200
  structured_output: true
repeats: {repeats}
rpm: 18
judge_repeats: {judge_repeats}
stability_cases: {stability_cases}
"""

RAG_ROWS = [
    {
        "id": "rag-001",
        "category": "answerable",
        "question": "How many days do I have to return an item?",
        "expected": "answer",
        "required_facts": ["30 days"],
        "expected_docs": ["kb-returns"],
        "forbidden": [],
    },
    {
        "id": "rag-002",
        "category": "unanswerable",
        "question": "Do you rent out ladders by the day?",
        "expected": "dont_know",
        "forbidden": [],
    },
]
TRIAGE_ROWS = [
    {
        "id": "tri-001",
        "text": "Synthetic: where is order TS-111111?",
        "category": "order_status",
        "priority": "normal",
        "order_id": "TS-111111",
        "priority_rule": "N1",
    }
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def unanswerable_rows(count: int) -> list[dict[str, Any]]:
    """`count` synthetic unanswerable RAG cases with distinct questions."""
    return [
        {
            "id": f"rag-{n:03d}",
            "category": "unanswerable",
            "question": f"Synthetic question number {n}: do you rent ladders?",
            "expected": "dont_know",
            "forbidden": [],
        }
        for n in range(1, count + 1)
    ]


def make_workspace(
    root: Path,
    *,
    repeats: int = 2,
    judge_repeats: str = "first",
    stability_cases: str = "null",
    rag_rows: list[dict[str, Any]] | None = None,
) -> Path:
    config = CONFIG_YAML.format(
        repeats=repeats, judge_repeats=judge_repeats, stability_cases=stability_cases
    )
    (root / "config.yaml").write_text(config, encoding="utf-8")
    write_jsonl(root / "datasets" / "rag.jsonl", RAG_ROWS if rag_rows is None else rag_rows)
    write_jsonl(root / "datasets" / "triage.jsonl", TRIAGE_ROWS)
    (root / "cassettes").mkdir()
    (root / "cassettes" / ".gitkeep").write_text("", encoding="utf-8")
    return root


def reply_for(body: dict[str, Any]) -> str:
    """A fixed synthetic reply per kind of call; the two prompt versions differ."""
    schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
    if schema == "judge_verdict":
        return json.dumps(
            {"groundedness": 5, "helpfulness": 4, "tone": 5, "pass": True, "reasons": "Fine."}
        )
    if schema == "pairwise_verdict":
        return json.dumps({"preferred": "tie", "reasons": "Synthetic reasons."})
    question = body["messages"][-1]["content"]
    if question.startswith("Synthetic:"):
        return json.dumps(
            {
                "category": "order_status",
                "priority": "normal",
                "order_id": "TS-111111",
                "summary": "Synthetic status question.",
            }
        )
    if "Follow these rules" in body["messages"][0]["content"]:  # assistant v2
        return "Returns are accepted within 30 days [kb-returns]."
    return "You have 30 days [kb-returns]."


Handler = Callable[[httpx.Request, dict[str, Any]], httpx.Response | None]


@dataclass
class SyntheticOpenRouter:
    """Serves `/chat/completions`, `/key` and `/models`.

    - `remaining`: the free-model requests the key endpoint reports (None:
      the endpoint leaves the field out); each answered chat call uses one;
    - `chat_override`: called before the normal reply; a response it returns
      is sent instead (for 429s and errors), None means answer normally;
    - `cost`: the `usage.cost` of every chat reply (None: the field is absent);
    - `content`: called for each answered chat call; a text it returns
      replaces the fixed reply (None keeps it).
    """

    remaining: int | None = 1000
    cost: float | None = 0.0
    chat_override: Handler | None = None
    content: Callable[[dict[str, Any]], str | None] | None = None
    prices: dict[str, tuple[str, str]] = field(default_factory=dict)
    chat_bodies: list[dict[str, Any]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.endswith("/key"):
            if self.remaining is None:
                return httpx.Response(200, json={"data": {}})
            used = max(0, 50 - self.remaining)
            quota = {"used": used, "limit": used + self.remaining, "remaining": self.remaining}
            return httpx.Response(200, json={"data": {"free_model_daily_requests": quota}})
        if path.endswith("/models"):
            data = [
                {"id": model, "pricing": {"prompt": prompt, "completion": completion}}
                for model, (prompt, completion) in self.prices.items()
            ]
            return httpx.Response(200, json={"data": data})
        body = json.loads(request.content)
        if self.chat_override is not None:
            response = self.chat_override(request, body)
            if response is not None:
                return response
        self.chat_bodies.append(body)
        if self.remaining is not None:
            self.remaining -= 1
        text = (self.content(body) if self.content else None) or reply_for(body)
        usage: dict[str, Any] = {"prompt_tokens": 120, "completion_tokens": 30}
        if self.cost is not None:
            usage["cost"] = self.cost
        return httpx.Response(
            200,
            json={
                "id": f"gen-synthetic-{len(self.chat_bodies)}",
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"content": text}}
                ],
                "usage": usage,
            },
        )


class NoWait:
    def acquire(self) -> float:
        return 0.0


def everything_under(root: Path) -> str:
    """The text of every file under `root`, to search for a leaked key."""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
