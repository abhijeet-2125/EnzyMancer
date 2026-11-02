"""Runtime configuration = weights/mode.json merged over defaults (unknown keys ignored, errors are logged)."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    scoring: str = "vote"                 # vote | learned
    vote_k: int = 50
    vote_power: float = 12.0
    source_weights: dict = field(default_factory=lambda: {"sp_": 1.0, "un_": 0.2, "u2_": 0.1})
    extra_db: str = "trembl_ev2.dmnd"     # second, larger reference searched in a fast pass ('' = off)
    extra_mode: str = "fast"
    extra_guard: float = 330.0            # skip the extra pass if more than this many seconds have elapsed
    weak_pid: float = 30.0                # queries with best hit <= this identity are re-searched ultra-sensitively
    weak_max: int = 3000
    ultra_guard: float = 215.0            # ... only if the main search finished before this many seconds
    top_k: int = 50
    deadline_s: float = 420.0             # DIAMOND is killed past this many seconds after start
    threads: int = 2
    use_external: bool = True             # False = strict mode: train-only references (needs train_only.dmnd)

    @classmethod
    def load(cls, path, log=print):
        cfg = cls()
        path = Path(path)
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            log(f"config: {path} not found, using defaults")
            return cfg
        except Exception as exc:                       # v16 swallowed this silently; we never do
            log(f"config: cannot parse {path}: {exc!r}; using defaults")
            return cfg
        names = {f.name for f in fields(cls)}
        for k, v in raw.items():
            if k in names:
                setattr(cfg, k, v)
        return cfg
