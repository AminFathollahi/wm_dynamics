"""io_utils.py — safe concurrent updates to the shared results JSON file.

Every analysis script reads results/all_statistics.json, adds or overwrites
its own key, and writes the whole file back. Run sequentially (as this
project has done so far) that is safe; run two such scripts concurrently and
the second writer can silently clobber the first's update (a lost-update
race). locked_json_update serializes just that read-modify-write critical
section via an exclusive file lock, so independent analyses can run in
parallel while still writing to one shared file safely.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def locked_json_update(path: str | Path) -> Iterator[dict]:
    """Read a JSON file, yield its contents for in-place mutation, write it
    back — all under an exclusive lock held for the duration of the block,
    so concurrent callers serialize rather than race.

    Missing files are treated as an empty dict. Usage:

        with locked_json_update(RESULTS / "all_statistics.json") as stats:
            stats["my_analysis_key"] = result
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
            else:
                data = {}
            yield data
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
