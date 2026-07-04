"""I/O helpers: jsonl reading/writing and results serialisation with provenance headers.

Kept dependency-free (stdlib json only) so it runs in the light local environment.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Yield one parsed object per non-blank line of a JSONL file.

    Streams rather than loading the whole file, since the real corpus.jsonl has ~168k premises.
    """
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line") from exc


def write_json(path: str, obj: Any) -> None:
    """Write `obj` as indented UTF-8 JSON, creating parent directories as needed.

    Results JSON is expected to carry a provenance header (config, model id, dataset version,
    git commit, seed, split — docs/EVALUATION.md §5); this helper just serialises whatever the
    caller assembled, so provenance construction stays in the eval loop where the values live.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
