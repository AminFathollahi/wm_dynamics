from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from info_decoding import aggregate_seed_records, matched_method_contrast


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary.write_text(text)
    temporary.replace(path)


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    seen_seeds = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("scope", {}).get("seed") != int(path.stem.split("_")[-1]):
            raise ValueError(f"incomplete or mismatched seed output: {path}")
        seed = payload["scope"]["seed"]
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed output: {seed}")
        seen_seeds.add(seed)
        records.extend(payload.get("records", []))
    if len(seen_seeds) != len(paths):
        raise ValueError("missing requested seed output")
    return records


def aggregate_results(records: list[dict], candidates: list[str], reference: str, n_bootstrap: int, seed: int) -> dict:
    identities = [(r.get("seed"), r.get("corpus"), r.get("session"), r.get("level"), r.get("candidate"), r.get("decoder")) for r in records]
    if any(item[0] is None for item in identities) or len(identities) != len(set(identities)):
        raise ValueError("missing or duplicate seed cell records")
    seeds = sorted({identity[0] for identity in identities})
    cells_by_seed = {
        item: {identity[1:] for identity in identities if identity[0] == item}
        for item in seeds
    }
    expected = cells_by_seed[seeds[0]] if seeds else set()
    if not expected or any(cells != expected for cells in cells_by_seed.values()):
        raise ValueError("seed outputs do not contain identical benchmark cells")
    failures = [record for record in records if record.get("status") not in {"computed", "excluded"}]
    if failures:
        raise ValueError("one or more repeated-seed benchmark cells did not compute")
    cells = aggregate_seed_records(records)
    decoders = sorted({r["decoder"] for r in cells})
    metrics = ("same_time", "cross_temporal")
    contrasts = {f"{candidate}|{decoder}|{metric}": matched_method_contrast(records, candidate, reference, decoder=decoder, metric=metric, n_bootstrap=n_bootstrap, seed=seed)
                 for candidate in candidates if candidate != reference for decoder in decoders for metric in metrics}
    return {"status": "complete", "seeds": seeds, "cells": cells, "contrasts": contrasts,
            "uncertainty_unit": "recording_session", "n_input_records": len(records)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run and aggregate repeated information-benchmark seeds.")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "info_benchmark_seeds")
    p.add_argument("--aggregate-output", type=Path, default=ROOT / "results" / "info_benchmark_seed_sweep.json")
    p.add_argument("--aggregate-summary", type=Path, default=ROOT / "INFO_BENCHMARK_SEEDS.md")
    p.add_argument("--reference", default="native_full_rank")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--aggregate-only", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--benchmark-args" not in values:
        args = parser().parse_args(values)
        args.benchmark_args = []
        return args
    split = values.index("--benchmark-args")
    args = parser().parse_args(values[:split])
    args.benchmark_args = values[split + 1:]
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in args.seeds:
        output = args.output_dir / f"seed_{seed}.json"
        paths.append(output)
        if args.aggregate_only and not output.exists():
            raise SystemExit(f"missing seed output: {output}")
        if output.exists() and not args.aggregate_only:
            try:
                payload = json.loads(output.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("status") == "complete" and payload.get("scope", {}).get("seed") == seed:
                continue
        if output.exists() and args.aggregate_only:
            continue
        checkpoint = args.output_dir / f"checkpoints_seed_{seed}"
        command = [sys.executable, str(ROOT / "scripts" / "run_info_benchmark.py"), "--seed", str(seed),
                   "--output", str(output), "--summary", str(args.output_dir / f"seed_{seed}.md"),
                   "--checkpoint-dir", str(checkpoint), *args.benchmark_args]
        subprocess.run(command, check=True)
    records = load_records(paths)
    requested = set(args.seeds)
    observed = {r.get("seed") for r in records}
    if not requested <= observed:
        raise ValueError(f"missing requested seed records: {sorted(requested - observed)}")
    candidates = sorted({r["candidate"] for r in records if r.get("candidate")})
    result = aggregate_results(records, candidates, args.reference, args.n_bootstrap, 0)
    atomic_write(args.aggregate_output, result)
    lines = ["# Repeated information benchmark", "", "Status: complete", "",
             "Seed scores are averaged within each recording/session/item-count cell before method contrasts. "
             "Confidence intervals resample recording sessions while retaining item-count cells within session.", "",
             "| Candidate, decoder, metric | Contrast | Estimate | 95% CI | Sessions | Cells |", "|---|---|---:|---:|---:|---:|"]
    for candidate, contrast in result["contrasts"].items():
        if contrast["status"] != "computed":
            lines.append(f"| {candidate} | not computable | | | 0 | 0 |")
        else:
            lines.append(f"| {candidate} | matched vs {args.reference} | {contrast['estimate']:.4f} | [{contrast['ci95'][0]:.4f}, {contrast['ci95'][1]:.4f}] | {contrast['n_sessions']} | {contrast['n_cells']} |")
    atomic_write(args.aggregate_summary, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
