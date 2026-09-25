from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from statistics import permutation_pvalue

MIN_TOTAL_TRIALS_DEFAULT = 10
MIN_TRIALS_PER_LEVEL_DEFAULT = 2
OMNIBUS_ALPHA_DEFAULT = 0.05
PERMUTATION_ALPHA_DEFAULT = 0.05
N_PERM_DEFAULT = 10000


def trial_level_reduce(rates: NDArray, labels: NDArray, trial_id: NDArray
                        ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    order = np.argsort(trial_id, kind="stable")
    sorted_tid, sorted_labels, sorted_rates = trial_id[order], labels[order], rates[order]
    boundaries = np.flatnonzero(np.diff(sorted_tid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(sorted_tid)]])
    trial_ids_unique = sorted_tid[starts]
    label_per_trial = sorted_labels[starts]
    trial_sum = np.add.reduceat(sorted_rates, starts)
    rows_per_trial = ends - starts
    return trial_ids_unique, label_per_trial, trial_sum, rows_per_trial


def two_stage_selectivity_test(
    rates: NDArray, labels: NDArray, trial_id: NDArray, rng: np.random.Generator,
    n_perm: int = N_PERM_DEFAULT, min_total_trials: int = MIN_TOTAL_TRIALS_DEFAULT,
    min_trials_per_level: int = MIN_TRIALS_PER_LEVEL_DEFAULT,
    omnibus_alpha: float = OMNIBUS_ALPHA_DEFAULT, permutation_alpha: float = PERMUTATION_ALPHA_DEFAULT,
) -> dict:
    """Two-stage non-parametric selectivity test: a trial-block permutation omnibus across label
    levels, then a right-tailed permutation test of the preferred level's mean against the rest.

    rates, labels, trial_id are one row per presentation event (a trial may contribute more than one
    row, e.g. an encoding presentation and a probe, all sharing one trial_id); a permutation reassigns
    labels at the trial level and moves every row of a trial together, never individual rows.
    """
    _, label_per_trial, trial_sum, rows_per_trial = trial_level_reduce(rates, labels, trial_id)
    n_trials_total = len(label_per_trial)
    if n_trials_total < min_total_trials:
        return {"status": "underpowered", "reason": f"fewer than {min_total_trials} trials"}
    levels, counts = np.unique(label_per_trial, return_counts=True)
    usable_levels = levels[counts >= min_trials_per_level]
    if len(usable_levels) < 2:
        return {"status": "underpowered",
                "reason": f"fewer than 2 levels with >={min_trials_per_level} trials"}
    usable = np.isin(label_per_trial, usable_levels)
    lab_u, sum_u, rows_u = label_per_trial[usable], trial_sum[usable], rows_per_trial[usable]
    n_trials_used = usable.sum()
    total_sum, total_rows = sum_u.sum(), rows_u.sum()
    grand_mean = total_sum / total_rows

    def _between_ss(label_matrix: NDArray) -> NDArray:
        ss = np.zeros(label_matrix.shape[0])
        for level in usable_levels:
            is_level = label_matrix == level
            sum_level = (is_level * sum_u[None, :]).sum(axis=1)
            rows_level = (is_level * rows_u[None, :]).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_level = np.where(rows_level > 0, sum_level / rows_level, grand_mean)
            ss += rows_level * (mean_level - grand_mean) ** 2
        return ss

    means_obs = np.array([sum_u[lab_u == lv].sum() / rows_u[lab_u == lv].sum() for lv in usable_levels])
    preferred = usable_levels[np.argmax(means_obs)]
    between_ss_obs = _between_ss(lab_u[None, :])[0]

    pref_mask = lab_u == preferred
    pref_rows, pref_sum = rows_u[pref_mask].sum(), sum_u[pref_mask].sum()
    rest_rows, rest_sum = total_rows - pref_rows, total_sum - pref_sum
    obs_diff = float(pref_sum / pref_rows - rest_sum / rest_rows)

    def _max_level_minus_rest(label_matrix: NDArray) -> NDArray:
        level_sum = np.stack([(label_matrix == lv) * sum_u[None, :] for lv in usable_levels], axis=1).sum(axis=2)
        level_rows = np.stack([(label_matrix == lv) * rows_u[None, :] for lv in usable_levels], axis=1).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            level_mean = np.where(level_rows > 0, level_sum / level_rows, grand_mean)
        best = np.argmax(level_mean, axis=1)
        draw_idx = np.arange(label_matrix.shape[0])
        pref_sum_draw, pref_rows_draw = level_sum[draw_idx, best], level_rows[draw_idx, best]
        rest_sum_draw, rest_rows_draw = total_sum - pref_sum_draw, total_rows - pref_rows_draw
        return pref_sum_draw / pref_rows_draw - rest_sum_draw / rest_rows_draw

    perm_idx = np.argsort(rng.random((n_perm, n_trials_used)), axis=1)
    perm_labels = lab_u[perm_idx]
    between_ss_null = _between_ss(perm_labels)
    omnibus_p = permutation_pvalue(between_ss_null >= between_ss_obs)

    diff_null = _max_level_minus_rest(perm_labels)
    perm_p = permutation_pvalue(diff_null >= obs_diff)

    meets_selectivity_criterion = bool(
        np.isfinite(omnibus_p) and omnibus_p < omnibus_alpha and perm_p < permutation_alpha
    )
    return {
        "status": "computed", "n_trials": int(n_trials_total), "n_levels_usable": int(len(usable_levels)),
        "preferred_level": preferred.item() if hasattr(preferred, "item") else preferred,
        "omnibus_between_group_ss": float(between_ss_obs), "omnibus_p": float(omnibus_p),
        "preferred_vs_rest_mean_difference": obs_diff, "permutation_p": perm_p,
        "meets_selectivity_criterion": meets_selectivity_criterion,
    }
