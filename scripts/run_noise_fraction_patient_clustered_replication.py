#!/usr/bin/env python3
"""run_noise_fraction_patient_clustered_replication.py -- patient-clustered
companions to the two paired tests behind
results/band_versus_sensor_decomposition.json's pre-declared deciding
quantity (the factor-analysis observation-noise variance fraction), computed
from per-session values already on disk.

Comparison A (band effect at fixed sensor: depth-contact low band minus the
existing high-gamma reference) and Comparison B (sensor effect at fixed
band: scalp minus depth-contact, both at the low band) are session-level in
the artifact even though DANDI 000574 nests roughly four sessions per
patient. This script adds a patient-clustered version beside each
session-level one -- no session is refit -- following the same reduction
(median of each patient's sessions, then the same paired sign-flip test
re-run over patients) that results/band_versus_sensor_decomposition.json's
own patient_clustered_persistence_replication field already applies to the
persistence-contrast observable.

Deliverable: one new top-level field, added extend-only. Every pre-existing
key is verified byte-identical before and after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import locked_json_update  # noqa: E402
from provenance import canonical_json  # noqa: E402
from run_band_versus_sensor_decomposition import (  # noqa: E402
    _extract_noise_fraction, load_existing_high_gamma_sessions,
)
from run_band_versus_sensor_decomposition_extensions import (  # noqa: E402
    load_checkpoint_sessions, sessions_for_cell,
)
from run_persistence_patient_clustered_replication import (  # noqa: E402
    _paired_patient_stats, _patient_median,
)

BAND_SENSOR_ARTIFACT = ROOT / "results" / "band_versus_sensor_decomposition.json"


def noise_fraction_patient_clustered() -> dict:
    checkpoint_sessions = load_checkpoint_sessions()
    hg_by_bin: dict[int, dict] = {}
    for (patient, session, bin_ms), s in load_existing_high_gamma_sessions().items():
        hg_by_bin.setdefault(bin_ms, {})[(patient, session)] = s

    by_bin = {}
    for bin_ms in (100, 200):
        depth_a = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_a_bin{bin_ms}", bin_ms)
        depth_b = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_b_bin{bin_ms}", bin_ms)
        scalp = sessions_for_cell(checkpoint_sessions, "scalp_low_band_bin{bin_ms}", bin_ms)
        hg = hg_by_bin.get(bin_ms, {})

        depth_a_levels = {k: v for k, v in ((key, _extract_noise_fraction(cell))
                                             for key, cell in depth_a.items()) if v is not None}
        depth_b_levels = {k: v for k, v in ((key, _extract_noise_fraction(cell))
                                             for key, cell in depth_b.items()) if v is not None}
        scalp_levels = {k: v for k, v in ((key, _extract_noise_fraction(cell))
                                           for key, cell in scalp.items()) if v is not None}
        hg_levels = {k: v for k, v in ((key, _extract_noise_fraction(cell))
                                        for key, cell in hg.items()) if v is not None}

        # Comparison B's own pairing (scalp vs depth) is restricted to sessions present
        # in both arms, matching resolve_comparison's own construction for Comparison B.
        shared_b = set(depth_b_levels) & set(scalp_levels)
        depth_b_shared = {k: v for k, v in depth_b_levels.items() if k in shared_b}
        scalp_shared = {k: v for k, v in scalp_levels.items() if k in shared_b}

        depth_a_patient = _patient_median(depth_a_levels)
        hg_patient = _patient_median(hg_levels)
        depth_b_patient = _patient_median(depth_b_shared)
        scalp_patient = _patient_median(scalp_shared)

        by_bin[f"bin{bin_ms}"] = {
            "n_sessions_band_pairing": len(set(depth_a_levels) & set(hg_levels)),
            "n_sessions_sensor_pairing": len(shared_b),
            "comparison_a_band_effect_at_fixed_sensor": _paired_patient_stats(
                depth_a_patient, hg_patient, ("patient_noise_fraction_comparison_a", bin_ms)),
            "comparison_b_sensor_effect_at_fixed_band": _paired_patient_stats(
                scalp_patient, depth_b_patient, ("patient_noise_fraction_comparison_b", bin_ms)),
        }
    return {
        "method": (
            "each patient reduced to the median of that patient's session-level factor-analysis "
            "observation-noise variance fraction (dimensionality.factor_analysis."
            "observation_noise_variance_fraction, this artifact's own pre-declared deciding quantity, "
            "predeclared_branches.deciding_quantity) before the same two-sided paired sign-flip test "
            "already used at the session level (comparison_a_band_effect_at_fixed_sensor and "
            "comparison_b_sensor_effect_at_fixed_band) is re-run over patients instead of sessions; "
            "session-level values and tests are unchanged and stand beside this as the descriptive number. "
            "Interest-minus-reference direction matches the session-level comparisons: Comparison A is "
            "depth-contact low band minus the existing high-gamma reference, Comparison B is scalp minus "
            "depth-contact, both at the low band."
        ),
        "by_bin_width": by_bin,
    }


def main() -> None:
    new_fields = {"patient_clustered_noise_fraction_replication": noise_fraction_patient_clustered()}
    with locked_json_update(BAND_SENSOR_ARTIFACT) as data:
        before = canonical_json({k: v for k, v in data.items() if k not in new_fields})
        already_present = [f for f in new_fields if f in data]
        if already_present:
            raise RuntimeError(f"refusing to overwrite already-present extend-only fields: {already_present}")
        data.update(new_fields)
        after = canonical_json({k: v for k, v in data.items() if k not in new_fields})
        if before != after:
            raise RuntimeError("extend-only violation: an existing band_versus_sensor_decomposition.json key would have changed")

    print(f"extended {BAND_SENSOR_ARTIFACT.name} with patient_clustered_noise_fraction_replication", file=sys.stderr)
    print(json.dumps(new_fields["patient_clustered_noise_fraction_replication"]["by_bin_width"], indent=2, default=str))


if __name__ == "__main__":
    main()
