"""Per-source-file resumable state tracking.

Each source file gets one JSON state file at
`<output>/.state/<source_id>.json` recording which expensive stages have
already completed and where their cached artifacts live. `run` (in
tts_pipeline.py) checks this before doing any expensive work, so re-running
the pipeline on the same input folder does not redo Whisper transcription,
the 16kHz analysis copy, or streaming metrics unless `--force` is passed or
the source file itself has visibly changed (size/mtime).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

STATE_SCHEMA_VERSION = 1


@dataclasses.dataclass
class FileState:
    source_path: str
    source_size: int
    source_mtime: float
    source_id: str
    stages: Dict[str, Any] = dataclasses.field(default_factory=dict)
    status: str = "pending"  # pending | in_progress | completed | failed
    error: Optional[str] = None
    updated_at: float = dataclasses.field(default_factory=time.time)
    schema_version: int = STATE_SCHEMA_VERSION

    def mark_stage(self, name: str, **data: Any) -> None:
        self.stages[name] = {"completed_at": time.time(), **data}
        self.updated_at = time.time()

    def has_stage(self, name: str) -> bool:
        return name in self.stages

    def stage_data(self, name: str) -> Dict[str, Any]:
        return self.stages.get(name, {})


def source_id_for(path: Path) -> str:
    """Stable, filesystem-safe id derived from the filename. Used for cache
    keys, output subfolders, and utterance_id prefixes.
    """
    stem = path.stem
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
    return safe or "source"


def load_state(state_dir: Path, source_id: str) -> Optional[FileState]:
    p = state_dir / f"{source_id}.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return FileState(**raw)
    except Exception:
        return None


def save_state(state_dir: Path, state: FileState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / f"{state.source_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(state), indent=2, default=str))
    tmp.replace(p)


def is_stale(state: FileState, current_size: int, current_mtime: float) -> bool:
    """A cached state is stale if the underlying source file changed since
    it was recorded (different size, or mtime moved by more than a second).
    """
    return state.source_size != current_size or abs(state.source_mtime - current_mtime) > 1.0
