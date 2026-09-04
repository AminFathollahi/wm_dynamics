"""run_behavior_amplitude_rate_controls.py -- is the leading-component
gain's correlation with trial outcome anything beyond total firing rate or
within-session drift?

results/state_behavior_link.json already established, pooled across the 11
macaque lPFC sessions that reach the matched-trial-count reachability floor
(at least 60 error trials), that trial outcome correlates with three cheap
per-trial covariates: the leading-component gain of the centred trial x
window matrix (r=-0.168, p=0.006), total spike count in the delay epoch
(r=-0.242, p=0.0013), and trial index within the session (r=-0.045,
p=0.032). None of the three had been tested against the two things that
trivially explain a firing-rate correlate of accuracy: a trial with more
spikes projects further along almost any component, so the gain correlate
could be firing rate wearing a geometric label; and accuracy declines across
a session (the trial-index correlate), so a rate that also rises across a
session would make the rate-accuracy correlation nonstationarity rather than
a trial-by-trial effect. This module runs both controls.

Every partial correlation reuses statistics.partial_correlation_permutation_
test (added for this purpose: none of this project's existing permutation-
test helpers control for a second variable). Every per-session covariate
(gain, total spike count, trial index) reuses
run_state_behavior_link.trial_amplitude_covariates unchanged -- the exact
computation results/state_behavior_link.json's cheap_first_look already
performed, refactored out of that function so it can be reused here without
forking it. Every pooling step across the 11 sessions reuses
state_persistence.slope_across_sessions_test, the same two-sided paired
sign-flip pooling this project uses for any per-session scalar, slope or
otherwise.

Does NOT touch results/state_behavior_link.json (finished and correct, not
regenerated) and is not a search for a fourth or fifth behavioural
correlate: it controls the three correlates that artifact already found.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import data_root  # noqa: E402
from run_state_behavior_link import (  # noqa: E402
    MIN_ERROR_TRIALS_FOR_REACHABILITY, _counts_from_spikes, _panichello_directory, trial_amplitude_covariates,
)
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import partial_correlation_permutation_test, pearson_permutation_test  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "behavior_amplitude_rate_controls.json"
N_PERM = 10000

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Compute, per reachable macaque lPFC session, the partial point-biserial correlation of trial "
    "outcome with the leading-component gain controlling for total spike count, and the reverse (outcome "
    "with total spike count controlling for gain). Pool each set of 11 per-session partial correlations "
    "across sessions with the two-sided paired sign-flip test used elsewhere in this project. Before any "
    "fit runs: if the gain correlate's pooled partial correlation is not significant at the two-sided "
    "0.05 level once total spike count is controlled for, the branch is "
    "'amplitude_correlate_of_accuracy_is_rate_not_geometry' and the geometric framing of the leading-"
    "component gain result is withdrawn everywhere this artifact is cited. If it remains significant, "
    "the branch is 'gain_correlate_survives_controlling_for_total_spike_count', and the same question is "
    "asked a second time after also controlling for trial index and, separately, for both trial index and "
    "total spike count together -- reported alongside, not substituted for, the single-control result."
)


def _reachable_sessions(root: Path) -> list[Path]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    paths = []
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, simplify_cells=True)
        is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
        n_error = int((~is_corr).sum())
        if n_error >= MIN_ERROR_TRIALS_FOR_REACHABILITY:
            paths.append(Path(path))
    return paths


def _session_covariates(path: Path) -> dict | None:
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    counts_all = _counts_from_spikes(spikes, time_ms)
    covariates = trial_amplitude_covariates(counts_all)
    if covariates["status"] != "computed":
        return None
    return {
        "is_corr": is_corr.astype(float),
        "gain": covariates["leading_component_score_gain"],
        "spike_count": covariates["total_spike_count"],
        "trial_index": covariates["trial_index"],
    }


def _partial(is_corr: np.ndarray, x: np.ndarray, controls: list[np.ndarray], seed_tag: str) -> dict:
    from statistics import stable_seed
    rng = np.random.default_rng(stable_seed(seed_tag))
    return partial_correlation_permutation_test(is_corr, x, controls=controls, n_perm=N_PERM, rng=rng)


def _analyze_session(session_id: str, cov: dict) -> dict:
    is_corr, gain, spike_count, trial_index = cov["is_corr"], cov["gain"], cov["spike_count"], cov["trial_index"]
    tag = f"behavior_amplitude_rate_controls|{session_id}"
    spike_vs_trial_index = {"status": "computed",
                             **pearson_permutation_test(spike_count, trial_index, n_perm=N_PERM,
                                                         rng=np.random.default_rng(0))}
    return {
        "n_trials": int(len(is_corr)),
        "raw_no_controls": {
            "gain": _partial(is_corr, gain, [], f"{tag}|gain|raw"),
            "spike_count": _partial(is_corr, spike_count, [], f"{tag}|spike_count|raw"),
        },
        "spike_count_vs_trial_index": spike_vs_trial_index,
        "control_spike_count": {
            "gain_given_spike_count": _partial(is_corr, gain, [spike_count], f"{tag}|gain|ctrl_spike"),
            "spike_count_given_gain": _partial(is_corr, spike_count, [gain], f"{tag}|spike_count|ctrl_gain"),
        },
        "control_trial_index": {
            "gain_given_trial_index": _partial(is_corr, gain, [trial_index], f"{tag}|gain|ctrl_trial"),
            "spike_count_given_trial_index": _partial(is_corr, spike_count, [trial_index], f"{tag}|spike_count|ctrl_trial"),
        },
        "control_trial_index_and_other_amplitude_measure": {
            "gain_given_trial_index_and_spike_count": _partial(
                is_corr, gain, [trial_index, spike_count], f"{tag}|gain|ctrl_trial_spike"),
            "spike_count_given_trial_index_and_gain": _partial(
                is_corr, spike_count, [trial_index, gain], f"{tag}|spike_count|ctrl_trial_gain"),
        },
    }


def _pool(sessions: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for s in sessions:
        node = s["analysis"]
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and node.get("status") == "computed":
            values.append(node["r"])
    return slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))


def main() -> None:
    root = data_root()
    t0 = time.time()
    paths = _reachable_sessions(root)

    output = {
        "version": "2026-08-24",
        "scope": (
            "Macaque lPFC only (Panichello et al. 2024): human trial-level accuracy is at ceiling and the "
            "mouse ALM corpus has no comparable per-trial accuracy, so this control cannot be run in "
            "either corpus. The 11 sessions reaching the pre-declared matched-contrast reachability floor "
            "(at least 60 error trials, the same floor results/state_behavior_link.json uses), the same 11 "
            "sessions that artifact's cheap_first_look_pooled field pools across. No trial is pooled across "
            "sessions or animals for any correlation here -- every correlation (raw or partial) is computed "
            "within one session, and only the resulting per-session correlation coefficients are pooled "
            "across sessions, by the paired sign-flip test. Accuracy stratifies hard by cohort (0.91/0.72/"
            "0.98) and by animal, which is exactly why trial-level pooling is not done."
        ),
        "limits": (
            "A firing-rate-versus-accuracy correlation, even after these controls, has a standard reading "
            "this corpus cannot exclude: arousal, motivation or task engagement, none of them measured "
            "here and none of them the cross-unit state's geometry. Controlling for total spike count and "
            "trial index rules out two specific confounds (the gain correlate being rate wearing a "
            "geometric label, and the rate correlate being within-session nonstationarity); it does not "
            "rule out a shared physiological driver of both firing rate and behavioral performance."
        ),
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_perm": N_PERM,
        "n_sessions_reachable": len(paths),
        "sessions": [], "sessions_completed": [], "sessions_pending": [p.stem for p in paths],
    }
    _flush(output)

    for i, path in enumerate(paths):
        session_id = path.stem
        cov = _session_covariates(path)
        if cov is None:
            result = {"session": session_id, "status": "not_computable"}
        else:
            result = {"session": session_id, "status": "computed", "analysis": _analyze_session(session_id, cov)}
        output["sessions"].append(result)
        output["sessions_completed"].append(session_id)
        output["sessions_pending"] = [p.stem for p in paths[i + 1:]]
        _flush(output)
        print(f"progress {i + 1}/{len(paths)} {session_id} flushed", flush=True)

    computed = [s for s in output["sessions"] if s["status"] == "computed"]
    pooled = {
        "gain_raw": _pool(computed, ("raw_no_controls", "gain")),
        "spike_count_raw": _pool(computed, ("raw_no_controls", "spike_count")),
        "spike_count_vs_trial_index": _pool(computed, ("spike_count_vs_trial_index",)),
        "gain_given_spike_count": _pool(computed, ("control_spike_count", "gain_given_spike_count")),
        "spike_count_given_gain": _pool(computed, ("control_spike_count", "spike_count_given_gain")),
        "gain_given_trial_index": _pool(computed, ("control_trial_index", "gain_given_trial_index")),
        "spike_count_given_trial_index": _pool(computed, ("control_trial_index", "spike_count_given_trial_index")),
        "gain_given_trial_index_and_spike_count": _pool(
            computed, ("control_trial_index_and_other_amplitude_measure", "gain_given_trial_index_and_spike_count")),
        "spike_count_given_trial_index_and_gain": _pool(
            computed, ("control_trial_index_and_other_amplitude_measure", "spike_count_given_trial_index_and_gain")),
    }
    output["pooled"] = pooled

    gain_survives = bool(pooled["gain_given_spike_count"].get("significant", False))
    output["branch"] = (
        "gain_correlate_survives_controlling_for_total_spike_count" if gain_survives else
        "amplitude_correlate_of_accuracy_is_rate_not_geometry"
    )
    output["n_sessions_computed"] = len(computed)
    output["wall_clock_s"] = time.time() - t0
    output["status"] = "complete"
    _flush(output)
    print(json.dumps({"branch": output["branch"], "n_sessions_computed": len(computed)}, indent=2))


if __name__ == "__main__":
    main()
