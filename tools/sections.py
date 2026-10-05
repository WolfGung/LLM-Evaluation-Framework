"""The generated blocks other than the main table, rendered from results/.

Each block function takes the recorded run (`Recorded`) and returns parts
(`tools.formatting`): tables, headings and paragraphs. `tools.render` puts
them between their markers in README.md and docs/, and `tools.site` shows
some of them on the published page. Every number comes from the results
files, the run manifest, `results/judge-agreement.json`, the gate tolerances
in `config/gate.yaml` or the datasets; the same files give the same text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmeval.baseline import RunResults
from llmeval.cassettes import RunManifest


@dataclass(frozen=True)
class Recorded:
    """The recorded run and where the blocks read the rest from."""

    manifest: RunManifest
    run: RunResults
    results_dir: Path
    gate_config: Path
    rag_dataset: Path
