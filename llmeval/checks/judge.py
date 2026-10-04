"""The judge layer: a second model grades what rules cannot check.

The judge grades RAG answers on three criteria: groundedness to the retrieved
documents, helpfulness and tone. It never grades safety or required facts;
the rule-based layers own those.

The rubric lives in `rubrics/judge.md`. Its body is the judge's instructions
(criteria, anchors for scores 1, 3 and 5, the pass rule, the pairwise rule).
Its front matter gives the code the criteria and the minimum score per
criterion. The loader refuses a rubric whose prose pass rule and front matter
disagree, so the judge and the code apply one rule.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RUBRIC_PATH = Path("rubrics/judge.md")
CRITERIA = ("groundedness", "helpfulness", "tone")
SCORES = (1, 2, 3, 4, 5)
ANCHORS = (5, 3, 1)

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_KEYS = frozenset({"criteria", "pass_rule"})
_PASS_RULE_SECTION = "Pass rule"


class RubricError(ValueError):
    """The rubric file is missing, broken, or its prose and front matter disagree."""


@dataclass(frozen=True)
class Rubric:
    """A loaded rubric.

    - `text`: the body the judge reads (the front matter is left out);
    - `criteria`: the graded criteria, in order;
    - `pass_rule`: the minimum score per criterion for a pass;
    - `sha256`: of the file's text, so results can name the rubric they used.
    """

    text: str
    criteria: tuple[str, ...]
    pass_rule: Mapping[str, int]
    sha256: str

    def short_of(self, scores: Mapping[str, int]) -> tuple[str, ...]:
        """The criteria whose score is below the minimum, in rubric order."""
        return tuple(name for name in self.criteria if scores[name] < self.pass_rule[name])

    def passes(self, scores: Mapping[str, int]) -> bool:
        """The pass rule: every criterion at least its minimum."""
        return not self.short_of(scores)


def _sections(body: str) -> dict[str, str]:
    """`## Title` sections of the body, by title."""
    parts = re.split(r"^## +(.+?) *$", body, flags=re.MULTILINE)
    return {title: text for title, text in zip(parts[1::2], parts[2::2], strict=True)}


def _check_front_matter(meta: Any, name: str) -> dict[str, int]:
    if not isinstance(meta, dict):
        raise RubricError(f"{name}: front matter is not a mapping")
    if unknown := sorted(set(meta) - _KEYS):
        raise RubricError(f"{name}: unknown front matter key: {', '.join(unknown)}")
    if tuple(meta.get("criteria") or ()) != CRITERIA:
        raise RubricError(f"{name}: criteria must be {', '.join(CRITERIA)}, in that order")
    rule = meta.get("pass_rule")
    if not isinstance(rule, dict) or set(rule) != set(CRITERIA):
        raise RubricError(
            f"{name}: pass_rule must give a minimum score for each of {', '.join(CRITERIA)}"
        )
    for criterion, minimum in rule.items():
        # bool is an int in Python; a minimum of `true` is a typo, not a 1.
        if type(minimum) is not int or minimum not in SCORES:
            raise RubricError(
                f"{name}: pass_rule.{criterion} must be a whole number from 1 to 5, got {minimum!r}"
            )
    return {criterion: rule[criterion] for criterion in CRITERIA}


def _check_body(body: str, rule: Mapping[str, int], name: str) -> None:
    sections = _sections(body)
    for criterion in CRITERIA:
        title = criterion.capitalize()
        if title not in sections:
            raise RubricError(f"{name}: no '## {title}' section")
        for score in ANCHORS:
            if not re.search(rf"^- {score}: \S", sections[title], flags=re.MULTILINE):
                raise RubricError(f"{name}: {title} has no anchor for score {score}")
    if _PASS_RULE_SECTION not in sections:
        raise RubricError(f"{name}: no '## {_PASS_RULE_SECTION}' section")
    prose = " ".join(sections[_PASS_RULE_SECTION].split())
    for criterion, minimum in rule.items():
        stated = re.findall(rf"\b{criterion} is at least (\d)\b", prose)
        if not stated:
            raise RubricError(
                f"{name}: the pass rule does not say '{criterion} is at least {minimum}'"
            )
        for value in stated:
            if int(value) != minimum:
                raise RubricError(
                    f"{name}: the pass rule says {criterion} is at least {value}; "
                    f"the front matter says {minimum}"
                )


def parse_rubric(source: str, *, name: str) -> Rubric:
    """Parse and check a rubric; `name` is used in error messages."""
    text = source.replace("\r\n", "\n")
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise RubricError(f"{name}: no front matter between '---' lines")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise RubricError(f"{name}: front matter is not valid YAML: {exc}") from None
    rule = _check_front_matter(meta, name)
    body = match.group(2).strip()
    _check_body(body, rule, name)
    return Rubric(
        text=body,
        criteria=CRITERIA,
        pass_rule=rule,
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def load_rubric(path: Path | str = RUBRIC_PATH) -> Rubric:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RubricError(f"cannot read the rubric {path}: {exc.strerror}") from None
    return parse_rubric(source, name=path.name)
