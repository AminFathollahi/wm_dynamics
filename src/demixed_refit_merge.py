"""Replaces the demixed principal component rows of a finished artifact with refitted ones.

The superseding artifact holds every other row of the finished artifact unchanged, the refitted rows in the
demixed rows' places, and every summary block recomputed by the producer's own summary function over the
combined rows. The merge refuses when the finished artifact's own summaries cannot be reproduced from its rows,
when the two files record different settings, when the refitted rows do not cover exactly the rows they replace,
when a refitted row was scored on a different design (trial count, folds, held-out units), or when a summary
entry that does not involve the demixed candidate would change.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from provenance import canonical_json, sha256_file

DEMIXED = "demixed_principal_components"

DESIGN_FIELDS = (
    "corpus", "region", "session", "level", "decoder", "label", "mode", "metric", "seed", "n_trials", "n_splits",
    "n_time_bins", "n_folds", "n_obs", "n_channels", "n_bands", "n_spikes", "n_held_out_neurons", "n_held_in_neurons",
    "held_out_neuron_indices", "n_shared_channels", "n_test_trials", "contact_sets_identical", "patient", "session_a",
    "session_b",
)

Locator = Callable[[dict], dict]


class RefusedMerge(ValueError):
    pass


def _json(value) -> str:
    return canonical_json(value)


def _at(artifact: dict, path: tuple):
    for step in path:
        artifact = artifact[step]
    return artifact


def _put(artifact: dict, path: tuple, value) -> None:
    _at(artifact, path[:-1])[path[-1]] = value


def list_rows(path: tuple, key_fields: tuple, flag: str = "candidate") -> Locator:
    """Locator of the demixed rows of the list at `path`, keyed by `key_fields`."""
    def locate(artifact: dict) -> dict:
        rows, found = _at(artifact, path), {}
        for index, row in enumerate(rows):
            if row.get(flag) != DEMIXED:
                continue
            key = (".".join(path),) + tuple(_json(row.get(field)) for field in key_fields)
            if key in found:
                raise RefusedMerge(f"two demixed rows share the key {key}")
            found[key] = (rows, index)
        return found
    return locate


def nested_rows(walk: Callable[[dict], list]) -> Locator:
    """Locator for rows reached by a walk yielding (key, container, slot) for each demixed row."""
    def locate(artifact: dict) -> dict:
        found = {}
        for key, container, slot in walk(artifact):
            if key in found:
                raise RefusedMerge(f"two demixed rows share the key {key}")
            found[key] = (container, slot)
        return found
    return locate


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(handle, "w") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _masked(artifact: dict, locators: list[Locator], blocks: list[tuple]) -> str:
    clone = copy.deepcopy(artifact)
    for locate in locators:
        for container, slot in locate(clone).values():
            container[slot] = None
    for path in blocks:
        _put(clone, path, None)
    return _json(clone)


def _covers(old, new) -> bool:
    """True when `new` repeats every value of `old`; `new` may carry additional fields (a summary function that
    reports more than the one that produced the finished file)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and _covers(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(_covers(a, b) for a, b in zip(old, new))
    return old == new


def _added_fields(old, new, found: set) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in new.items():
            if key not in old:
                found.add(key)
            else:
                _added_fields(old[key], value, found)
    elif isinstance(old, list) and isinstance(new, list):
        for a, b in zip(old, new):
            _added_fields(a, b, found)


def _normal(value):
    return json.loads(_json(value))


def _compare_block(old, new, path: tuple, added: set) -> tuple[int, int]:
    """Counts entries unchanged and changed; an entry that does not mention the demixed candidate must not change."""
    if not isinstance(old, (dict, list)):
        return (1, 0) if _json(old) == _json(new) else (0, 1)
    if isinstance(old, dict):
        if sorted(old) != sorted(new):
            raise RefusedMerge(f"the keys of {'.'.join(path)} changed")
        pairs = [((key, old[key]), (key, new[key])) for key in sorted(old)]
    else:
        if len(old) != len(new):
            raise RefusedMerge(f"the length of {'.'.join(path)} changed: {len(old)} to {len(new)}")
        pairs = list(zip(old, new))
    unchanged = changed = 0
    for before, after in pairs:
        if _covers(_normal(before), _normal(after)):
            _added_fields(_normal(before), _normal(after), added)
            unchanged += 1
            continue
        if DEMIXED not in _json(before) and DEMIXED not in _json(after):
            raise RefusedMerge(f"an entry of {'.'.join(path)} that does not involve the demixed candidate would "
                               f"change: {_json(before)[:300]}")
        changed += 1
    return unchanged, changed


def merge_refit(
    delivered_path: Path, refit_path: Path, output_path: Path, *, locators: dict[str, Locator],
    settings: Callable[[dict], dict], recompute: Callable[[dict, bool], dict], code_identity: dict,
    partial: bool = False, accept_summary_rule_change: bool = False, notes: tuple = (),
    relaxed_when_partial: tuple = (), extra_checks: Callable[[dict, dict], None] | None = None,
    markdown: Callable[[dict], str] | None = None,
) -> dict:
    """Writes the superseding artifact at `output_path` and returns the account of what changed.

    `locators` maps a section name to a function finding that section's demixed rows; `settings` extracts the
    settings both files must share; `recompute(artifact, superseding)` returns {path tuple: block} for every summary
    block that involves the demixed candidate, from an artifact whose rows are already combined; `superseding` is
    False when reproducing the finished artifact and True for the combined one (inputs the summaries read from
    other finished files switch to their superseding versions). With `partial`, the refit may cover a
    subset of the demixed rows and only those are replaced.
    """
    delivered = json.loads(Path(delivered_path).read_text())
    refit = json.loads(Path(refit_path).read_text())
    if refit.get("status") != "complete":
        raise RefusedMerge(f"the refit file is not complete: status {refit.get('status')}")

    shared = {k: v for k, v in settings(delivered).items() if not (partial and k in relaxed_when_partial)}
    refit_settings = {k: v for k, v in settings(refit).items() if k in shared}
    if _json(shared) != _json(refit_settings):
        differing = sorted(k for k in shared if _json(shared[k]) != _json(refit_settings.get(k)))
        raise RefusedMerge(f"the files record different settings: {differing}")

    original_blocks = recompute(delivered, False)
    irreproducible = [".".join(path) for path, block in original_blocks.items()
                      if not _covers(_normal(_at(delivered, path)), _normal(block))]
    if irreproducible and not accept_summary_rule_change:
        raise RefusedMerge(f"the delivered {irreproducible} cannot be reproduced from its own rows")

    merged = copy.deepcopy(delivered)
    replaced = {}
    for name, locate in locators.items():
        old_rows, new_rows = locate(merged), locate(refit)
        if (set(old_rows) != set(new_rows)) if not partial else (not set(new_rows) <= set(old_rows)):
            raise RefusedMerge(f"{name}: demixed row keys differ between the two files "
                               f"({len(old_rows)} delivered, {len(new_rows)} refitted)")
        for key, (container, slot) in new_rows.items():
            before, after = old_rows[key][0][old_rows[key][1]], container[slot]
            for field in DESIGN_FIELDS:
                if field in before and field in after and _normal(before[field]) != _normal(after[field]):
                    raise RefusedMerge(f"{name}: {key} was scored on a different design ({field})")
            old_rows[key][0][old_rows[key][1]] = after
        replaced[name] = [list(key) for key in new_rows]

    blocks = recompute(merged, True)
    if _masked(delivered, list(locators.values()), list(blocks)) != _masked(merged, list(locators.values()), list(blocks)):
        raise RefusedMerge("a row outside the demixed rows and the recomputed blocks would change")
    for path, block in blocks.items():
        _put(merged, path, block)
    if extra_checks is not None:
        extra_checks(delivered, refit)

    unchanged = changed = 0
    added: set = set()
    unchanged_rule = 0
    for path, block in blocks.items():
        if ".".join(path) in irreproducible:
            unchanged_rule += len(block)
            continue
        a, b = _compare_block(_at(delivered, path), block, path, added)
        unchanged, changed = unchanged + a, changed + b

    merged["demixed_refit"] = {
        "rows_refitted": replaced,
        "refitted_row_count": sum(len(v) for v in replaced.values()),
        "partial": partial,
        "supersedes": {"path": str(delivered_path), "sha256": sha256_file(delivered_path)},
        "refit_file": {"path": str(refit_path), "sha256": sha256_file(refit_path)},
        "refit_code": {k: refit.get(k) for k in ("code_commit", "code_identity", "implementation") if k in refit},
        "merge_code_identity": code_identity,
        "recomputed_blocks": [".".join(path) for path in blocks],
        "notes": list(notes),
        "summary_entries_unchanged": unchanged,
        "summary_entries_changed": changed,
        "summary_fields_added": sorted(added),
        "blocks_computed_by_a_different_summary_rule": irreproducible,
        "entries_in_those_blocks": unchanged_rule,
    }
    _write_atomic(Path(output_path), _json(merged))
    if markdown is not None:
        _write_atomic(Path(output_path).with_suffix(".md"), markdown(merged))
    return merged["demixed_refit"]
