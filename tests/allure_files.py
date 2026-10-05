"""Read what allure-pytest writes into an `--alluredir`, as the Allure report would.

The Allure command line is not needed: each test result is a
`*-result.json` file, each attachment a file named by its `source`, and the
categories are matched by Allure's own rule (`categories_of`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# What must never reach a report: an OpenRouter key, request headers, and
# long hex strings such as cassette keys (a sha256 of a request).
KEY_LIKE = re.compile(r"sk-or-|Bearer|Authorization|\b[0-9a-f]{40,}\b")


def results_of(out_dir: Path) -> list[dict]:
    return [json.loads(path.read_text("utf-8")) for path in sorted(out_dir.glob("*-result.json"))]


def labels(result: dict, name: str) -> list[str]:
    return [label["value"] for label in result["labels"] if label["name"] == name]


def parameters(result: dict) -> dict[str, str]:
    return {param["name"]: param["value"] for param in result.get("parameters", [])}


def attachments(out_dir: Path, result: dict) -> dict[str, str]:
    return {
        item["name"]: (out_dir / item["source"]).read_text("utf-8")
        for item in result.get("attachments", [])
    }


def written_text(out_dir: Path, result: dict) -> str:
    """Everything the repository writes into one result: the title, the
    description, labels, parameters and attachments."""
    parts = [
        result["name"],
        result.get("description") or "",
        json.dumps(result["labels"]),
        json.dumps(result.get("parameters", [])),
        *attachments(out_dir, result).values(),
    ]
    return "\n".join(parts)


def categories_of(out_dir: Path, result: dict) -> list[str]:
    """The categories of categories.json a result falls in, by Allure's rule:
    the status is one of `matchedStatuses`, and `messageRegex` matches the
    whole message (Java's Pattern.matches with DOTALL)."""
    categories = json.loads((out_dir / "categories.json").read_text("utf-8"))
    message = (result.get("statusDetails") or {}).get("message")
    found = []
    for category in categories:
        if result["status"] not in category["matchedStatuses"]:
            continue
        regex = category.get("messageRegex")
        if regex is None or (message is not None and re.fullmatch(regex, message, re.DOTALL)):
            found.append(category["name"])
    return found
