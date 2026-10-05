"""Numbers, tables and Markdown for the generated blocks and the published page.

Percentages and seconds have one decimal, and a half rounds up (never to
even), so a rate reads the same everywhere it is shown. A block is a list of
parts: a `Table`, a `Heading` or a paragraph (a plain string); `markdown`
writes them for README.md and docs/, and `tools.site` writes the same parts
as HTML.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# A cell with nothing to show: the measure does not apply, or cannot be computed.
NONE = "—"


class RenderError(ValueError):
    """A block cannot be rendered or placed: broken results, or broken markers."""


def decimal(value: Decimal | float | int, places: int = 1) -> str:
    """`value` with `places` decimals; a half rounds up."""
    step = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(step, rounding=ROUND_HALF_UP))


def percent(count: int, total: int) -> str:
    """`count` of `total` as a percentage with one decimal; a half rounds up."""
    if not total:
        return NONE
    return f"{decimal(Decimal(count) * 100 / Decimal(total))}%"


def share(rate: float | None) -> str:
    """A stored rate (a share such as 0.9551) as a percentage with one decimal."""
    if rate is None:
        return NONE
    return f"{decimal(Decimal(str(rate)) * 100)}%"


def seconds(ms: float | None) -> str:
    """Milliseconds as seconds with one decimal; a half rounds up."""
    if ms is None:
        return NONE
    return f"{decimal(Decimal(str(ms)) / 1000)} s"


def count_of(count: int, total: int) -> str:
    """`count` of `total`, with the percentage when there is a total: "20 of 38 (52.6%)"."""
    if not total:
        return f"{count} of {total}"
    return f"{count} of {total} ({percent(count, total)})"


@dataclass(frozen=True)
class Table:
    """A table: a header, rows of cells, and an optional line under it.

    `numeric`: the columns after the first hold numbers and are right-aligned.
    `caption` names the table on the published page.
    """

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    line: str = ""
    numeric: bool = True
    caption: str = ""


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


Part = Table | Heading | str


def table_markdown(table: Table) -> str:
    """The table in Markdown, then the line under it when there is one."""
    align = "---:|" if table.numeric else "---|"
    lines = [
        "| " + " | ".join(table.header) + " |",
        "|---|" + align * (len(table.header) - 1),
        *("| " + " | ".join(row) + " |" for row in table.rows),
    ]
    text = "\n".join(lines)
    return f"{text}\n\n{table.line}" if table.line else text


def markdown(parts: Sequence[Part]) -> str:
    """The parts in Markdown, separated by blank lines."""
    out = []
    for part in parts:
        if isinstance(part, Table):
            out.append(table_markdown(part))
        elif isinstance(part, Heading):
            out.append(f"{'#' * part.level} {part.text}")
        else:
            out.append(part)
    return "\n\n".join(out)
