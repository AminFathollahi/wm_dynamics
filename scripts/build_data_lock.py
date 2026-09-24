from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "scripts"):
    path = str(ROOT / part)
    if path not in sys.path:
        sys.path.insert(0, path)

from data_integrity import corpus_inventory, load_datasets_config  # noqa: E402
from project_config import data_root  # noqa: E402
from provenance import _json_safe  # noqa: E402

OUTPUT_PATH = ROOT / "provenance" / "data_lock.json"
CRCNS_STATUS_DIR_NAME = "crcns-downloader"

CRCNS_EXCLUDED = {
    "pfc-2": "rat prefrontal-cortex corpus, deliberately excluded from this project (never registered in "
             "config/datasets.json); its verification_failed status is not chased.",
}


def _crcns_log_tally(log_text: str) -> dict | None:
    match = re.search(r"Number validated=(\d+)\.\s*Number hash not matching=(\d+)", log_text)
    if match is None:
        return None
    return {"validated": int(match.group(1)), "not_matching": int(match.group(2))}


def _crcns_status(root: Path) -> dict:
    status_dir = root / CRCNS_STATUS_DIR_NAME / "status"
    log_dir = root / CRCNS_STATUS_DIR_NAME / "logs"
    report = {}
    if not status_dir.is_dir():
        return report
    for status_path in sorted(status_dir.glob("*.status")):
        key = status_path.stem
        status_value = status_path.read_text().strip()
        entry = {"status_file_value": status_value}
        if key in CRCNS_EXCLUDED:
            entry["verdict"] = "excluded_from_project"
            entry["note"] = CRCNS_EXCLUDED[key]
        elif status_value == "verification_failed":
            log_path = log_dir / f"{key}.log"
            log_text = log_path.read_text() if log_path.is_file() else ""
            tally = _crcns_log_tally(log_text)
            corpus_dir = root / key
            on_disk = len(list(corpus_dir.rglob("*"))) if corpus_dir.is_dir() else 0
            checksum_path = corpus_dir / "checksums.md5"
            required_total = len(checksum_path.read_text().splitlines()) if checksum_path.is_file() else None
            if tally is not None and tally["not_matching"] == 0:
                entry["verdict"] = "status_file_stale_false_positive"
                entry["note"] = ("the wrapper exit code that writes this status file does not reflect the "
                                  "verify script's own tally; its log reports every checksum matched")
            elif tally is not None:
                entry["verdict"] = "genuinely_incomplete"
                entry["note"] = ("checksum mismatches and files never downloaded (network connection failures "
                                  "in the log); on-disk files are undersized relative to their checksummed size")
            else:
                entry["verdict"] = "unverified"
                entry["note"] = "no parseable verify tally in the log"
            entry["log_tally"] = tally
            entry["files_present_on_disk"] = on_disk
            entry["files_required_by_checksum_manifest"] = required_total
        else:
            entry["verdict"] = "ok"
        report[key] = entry
    return report


def build(root: Path) -> dict:
    config = load_datasets_config()
    corpora = {}
    for dataset_key in sorted(config["datasets"]):
        corpora[dataset_key] = corpus_inventory(root, dataset_key, config)
    all_complete = all(row["status"] == "complete" for row in corpora.values())
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_root": str(root),
        "all_complete": all_complete,
        "corpora": corpora,
        "crcns_status": _crcns_status(root),
    }


def main() -> None:
    root = data_root()
    result = build(root)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_json_safe(result), indent=2))
    n_complete = sum(1 for row in result["corpora"].values() if row["status"] == "complete")
    print(f"{n_complete}/{len(result['corpora'])} corpora complete, all_complete={result['all_complete']}")
    for key, row in result["corpora"].items():
        print(f"  {key:32s} {row['status']:22s} present={row['present']:5d} absent={row['absent_total']:4d} mode={row['mode']}")


if __name__ == "__main__":
    main()
