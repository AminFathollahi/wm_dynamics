"""spike_pipeline.py — shared single-unit Sternberg WM pipeline.

Rutishauser-lab Sternberg single-unit datasets (DANDI 000469, 001187, 000673,
and — via Boran's co-located microwires — 000574) share the same underlying
analysis: bin spike times into a firing-rate population vector, PCA-project,
run cross-temporal generalisation (load and item-identity) with a
label-permutation null, cross-validated participation ratio, and a
correct-vs-error drift comparison. NWB field/group names differ slightly
across releases (e.g. 000469's ``trials`` vs 001187's ``WM_trials``, and
``loadsEnc1_PicIDs`` vs ``PicIDs_Encoding1``), so each dataset gets a thin
driver script that extracts these into a common array representation and
calls the functions here — the numerical pipeline itself is written once.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

# Allow `from geometry import ...` to work whether src/ is on sys.path or not.
_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from geometry import (ctg_label_permutation_null, ctg_content_permutation_null,
                      temporal_stability_tau, spatiotemporal_participation_ratio,
                      geometric_drift)

N_PC_DEFAULT = 8
BIN_MS_DEFAULT = 100
SMOOTH_MS_DEFAULT = 200

# QC floors documented in Daume et al. 2024 (Neuron; DANDI 001187 source
# paper): 2/48 sessions excluded for behavioural accuracy below this floor,
# 67/883 neurons (7.1%) excluded for firing rate below this floor. Applied
# uniformly across all three Rutishauser-lineage Sternberg cohorts sharing
# this pipeline (000469, 001187, 000673) for cross-cohort QC consistency,
# even where a given cohort's own paper does not separately restate them.
MIN_SESSION_ACCURACY = 0.55
MIN_UNIT_FIRING_RATE_HZ = 0.1

# Region-stratified fitting (anatomy is a level of inference, never a
# nuisance variable pooled away -- a standing human ruling): declared
# independently of any fitted estimate, before any region-level result is
# computed. "pooled" reproduces the pre-stratification behavior (every unit
# in the session) and is retained as the superseded baseline, never removed.
ANATOMICAL_REGIONS = ("hippocampus", "amygdala", "pre_sma", "dacc", "vmpfc")
REGIONS_WITH_POOLED = ("pooled",) + ANATOMICAL_REGIONS
MIN_UNITS_PER_REGION = 8

# DANDI 000574 (Boran) single-unit region set: every structure the anatomical
# census (results/anatomical_census.json) found with n_units > 0 in this
# corpus. Distinct from ANATOMICAL_REGIONS because Boran's Brainnetome-hybrid
# labels resolve a different, wider set of structures than the Rutishauser
# suffix convention -- the pooled DANDI 000574 fit mixes all of these into
# one state today, which a region-stratified refit still needs to correct.
BORAN_ANATOMICAL_REGIONS = (
    "hippocampus", "amygdala", "entorhinal_parahippocampal",
    "inferior_temporal_gyrus", "middle_temporal_gyrus",
    "superior_temporal_gyrus", "vtc", "unspecific",
)
BORAN_REGIONS_WITH_POOLED = ("pooled",) + BORAN_ANATOMICAL_REGIONS


def filter_units_by_region(
    items: list[NDArray], unit_regions: NDArray, region: str
) -> list[NDArray]:
    """Restrict a per-unit list (spike times, waveforms, ...) to one region.

    ``region == "pooled"`` returns every unit unfiltered -- the
    pre-stratification behavior. ``unit_regions`` must be aligned with
    ``items`` in unit order (e.g. :func:`resolve_unit_regions`'s ``"region"``
    array before any firing-rate mask is applied).
    """
    if region == "pooled":
        return list(items)
    return [item for item, r in zip(items, unit_regions) if r == region]


def load_spike_times(f) -> list[NDArray]:
    """Return list of per-unit spike-time arrays from an NWB ``units`` table."""
    times_flat = f["units/spike_times"][:]
    index = f["units/spike_times_index"][:]
    units, prev = [], 0
    for idx in index:
        units.append(times_flat[prev:idx])
        prev = idx
    return units


# Controlled vocabulary for anatomical region resolution (anatomy is a level
# of inference, never a nuisance variable pooled away). One fine-grained structure name per row; every
# raw electrode/channel label observed across every corpus in
# config/datasets.json (results/anatomical_census.json) maps into this same
# table, through a per-corpus parser selected by that corpus's DECLARED
# ``label_convention`` (config/datasets.json), never sniffed at runtime.
# Anything unmapped falls through to "other" and must be surfaced explicitly
# by the caller, with the raw string preserved, rather than silently
# absorbed. White-matter and "unspecific"/unlabelled sites get their own
# status rather than being folded into "other" or dropped.
REGION_VOCABULARY: dict[str, str] = {
    "hippocampus": "hippocampus",
    "amygdala": "amygdala",
    "pre_supplementary_motor_area": "pre_sma",
    "dorsal_anterior_cingulate_cortex": "dacc",
    "ventral_medial_prefrontal_cortex": "vmpfc",
    "ventral_temporal_cortex": "vtc",
}

# Coarse grouping for every fine structure this project resolves, so a
# downstream analysis can choose its grain and state which it chose.
# Declared once here rather than re-derived per caller.
# "cingulate" and "subcortical_other" are kept distinct from
# "medial_frontal" / "MTL" because RAM's whole-cortex parcellation contacts
# (anterior cingulate gyrus, basal ganglia, ...) are not the same recording
# targets as the Rutishauser-lineage microwire ROIs of the same rough
# neighborhood, and collapsing them would misrepresent what was pooled.
COARSE_REGION_GROUP: dict[str, str] = {
    "hippocampus": "MTL", "amygdala": "MTL", "entorhinal_parahippocampal": "MTL",
    "vtc": "MTL",
    "middle_temporal_gyrus": "lateral_temporal", "superior_temporal_gyrus": "lateral_temporal",
    "inferior_temporal_gyrus": "lateral_temporal", "temporal_pole": "lateral_temporal",
    "posterior_superior_temporal_sulcus": "lateral_temporal",
    "pre_sma": "medial_frontal", "dacc": "medial_frontal", "vmpfc": "medial_frontal",
    "anterior_cingulate_gyrus": "cingulate", "posterior_cingulate": "cingulate",
    "rostral_anterior_cingulate": "cingulate",
    "dlpfc": "lateral_frontal", "ofc": "lateral_frontal",
    "inferior_frontal_gyrus": "lateral_frontal", "superior_frontal_gyrus": "lateral_frontal",
    "middle_frontal_gyrus": "lateral_frontal", "frontal_pole": "lateral_frontal",
    "precentral": "lateral_frontal",
    "parietal": "parietal", "postcentral": "parietal", "supramarginal_gyrus": "parietal",
    "angular_gyrus": "parietal", "paracentral_lobule": "parietal",
    "occipital": "occipital",
    "insula": "insula",
    "basal_ganglia": "subcortical_other", "ventricle": "subcortical_other",
    "white_matter": "white_matter", "unspecific": "unspecific", "unlabelled": "unlabelled",
    "scalp": "scalp", "other": "other",
}

# ``ind.region`` values from the FreeSurfer/Desikan-Killiany surface pipeline
# used by the RAM releases (ds005489, ds005557), lower-cased, no hemisphere.
_DK_APARC_MAP: dict[str, str] = {
    "bankssts": "superior_temporal_gyrus", "caudalanteriorcingulate": "anterior_cingulate_gyrus",
    "caudalmiddlefrontal": "dlpfc", "cuneus": "occipital", "entorhinal": "entorhinal_parahippocampal",
    "frontalpole": "frontal_pole", "fusiform": "vtc", "inferiorparietal": "parietal",
    "inferiortemporal": "inferior_temporal_gyrus", "insula": "insula",
    "isthmuscingulate": "posterior_cingulate", "lateraloccipital": "occipital",
    "lateralorbitofrontal": "ofc", "lingual": "occipital", "medialorbitofrontal": "ofc",
    "middletemporal": "middle_temporal_gyrus", "paracentral": "paracentral_lobule",
    "parahippocampal": "entorhinal_parahippocampal", "parsopercularis": "inferior_frontal_gyrus",
    "parsorbitalis": "inferior_frontal_gyrus", "parstriangularis": "inferior_frontal_gyrus",
    "pericalcarine": "occipital", "postcentral": "postcentral",
    "posteriorcingulate": "posterior_cingulate", "precentral": "precentral",
    "precuneus": "parietal", "rostralanteriorcingulate": "rostral_anterior_cingulate",
    "rostralmiddlefrontal": "dlpfc", "superiorfrontal": "superior_frontal_gyrus",
    "superiorparietal": "parietal", "superiortemporal": "superior_temporal_gyrus",
    "supramarginal": "supramarginal_gyrus", "temporalpole": "temporal_pole",
    "transversetemporal": "superior_temporal_gyrus",
}

# Hand/subcortical MTL annotations (``das.region``/``stein.region``) and
# whole-brain subcortical labels (``wb.region``) from the same RAM releases --
# finer-grained than the surface pipeline for exactly the structures (hippo-
# campus, amygdala, entorhinal/perirhinal cortex) this project cares most
# about, so the census prefers these columns over ``ind.region`` when present.
# Matched as a case-insensitive substring against the label with any leading
# "left "/"right " hemisphere token stripped first.
_DK_HAND_ANNOTATION_MAP: dict[str, str] = {
    "ca1": "hippocampus", "ca3": "hippocampus", "dg": "hippocampus", "sub": "hippocampus",
    "hippocampus": "hippocampus",
    "amy": "amygdala", "amygdala": "amygdala",
    "phc": "entorhinal_parahippocampal", "erc": "entorhinal_parahippocampal",
    "ba35": "entorhinal_parahippocampal", "ba36": "entorhinal_parahippocampal",
    "prc": "entorhinal_parahippocampal", "ent": "entorhinal_parahippocampal",
    "parahippocampal": "entorhinal_parahippocampal",
    "mtl wm": "white_matter", "cerebral white matter": "white_matter",
    "misc": "unspecific",
    "mtg": "middle_temporal_gyrus", "middle temporal gyrus": "middle_temporal_gyrus",
    "itg": "inferior_temporal_gyrus", "inferior temporal gyrus": "inferior_temporal_gyrus",
    "stg": "superior_temporal_gyrus", "superior temporal gyrus": "superior_temporal_gyrus",
    "tmp": "temporal_pole", "temporal pole": "temporal_pole",
    "fug": "vtc", "fusiform": "vtc",
    "smg": "supramarginal_gyrus", "supramarginal": "supramarginal_gyrus",
    "ang": "angular_gyrus", "angular gyrus": "angular_gyrus",
    "sfg": "superior_frontal_gyrus", "superior frontal": "superior_frontal_gyrus",
    "mfg": "middle_frontal_gyrus", "middle frontal": "middle_frontal_gyrus",
    "acgg": "anterior_cingulate_gyrus", "anterior cingulate": "anterior_cingulate_gyrus",
    "pcg": "posterior_cingulate", "posterior cingulate": "posterior_cingulate",
    "ains": "insula", "pins": "insula", "insula": "insula", "co": "insula",
    "prg": "precentral", "precentral": "precentral",
    "pog": "postcentral", "postcentral": "postcentral",
    "putamen": "basal_ganglia", "caudate": "basal_ganglia", "pallidum": "basal_ganglia",
    "mog": "occipital", "occipital": "occipital",
    "lateral ventricle": "ventricle", "inf lat vent": "ventricle",
    "pcu": "parietal", "precuneus": "parietal", "spl": "parietal",
    "superior parietal lobule": "parietal",
    "cun": "occipital", "cuneus": "occipital", "lig": "occipital", "lingual": "occipital",
    "morg": "ofc", "medial orbital gyrus": "ofc", "lorg": "ofc", "lateral orbital gyrus": "ofc",
    "porg": "ofc", "posterior orbital gyrus": "ofc", "aorg": "ofc", "anterior orbital gyrus": "ofc",
    "gre": "ofc", "gyrus rectus": "ofc",
    "ec": "entorhinal_parahippocampal",
    "pp": "superior_temporal_gyrus", "planum polare": "superior_temporal_gyrus",
    "ttg": "superior_temporal_gyrus", "transverse temporal gyrus": "superior_temporal_gyrus",
    "pt": "superior_temporal_gyrus", "planum temporale": "superior_temporal_gyrus",
    "mcgg": "posterior_cingulate", "middle cingulate gyrus": "posterior_cingulate",
    "frp": "frontal_pole", "frontal pole": "frontal_pole",
    "dlpfc": "dlpfc",
    "acg": "anterior_cingulate_gyrus",
    "clear label": "unlabelled",
}

# 10-20 scalp channel labels, used to recognize DANDI 000574's (Boran) EEG
# cap contacts by label alone when an NWB electrode-group cross-check isn't
# available to the caller (e.g. this module's own parser, called on a bare
# string).  scripts/build_anatomical_census.py additionally cross-checks
# against the NWB ``electrodes/group_name`` column before trusting this set.
SCALP_1020_LABELS = {
    "fp1", "fp2", "f7", "f3", "fz", "f4", "f8", "t3", "c3", "cz", "c4", "t4",
    "t5", "p3", "pz", "p4", "t6", "o1", "o2", "a1", "a2",
}

# Brainnetome-hybrid coarse codes shared verbatim by DANDI 000574 (Boran) and
# ds004752 -- both Sarnthein-lab Zurich verbal-Sternberg releases -- whose
# electrode labels read "{CoarseCode}, {Hemisphere} {CoarseName} {FineCode},
# {fine description}" (e.g. "Hipp, Left Hippocampus cHipp, caudal
# hippocampus") or the bare coarse code alone.
_BRAINNETOME_COARSE_MAP: dict[str, str] = {
    "hipp": "hippocampus", "amyg": "amygdala", "mtg": "middle_temporal_gyrus",
    "stg": "superior_temporal_gyrus", "itg": "inferior_temporal_gyrus", "fug": "vtc",
    "phg": "entorhinal_parahippocampal", "ins": "insula", "ipl": "parietal",
    "psts": "posterior_superior_temporal_sulcus", "bg": "basal_ganglia",
    "locc": "occipital", "org": "ofc", "pog": "postcentral", "prg": "precentral",
}


def _parse_rutishauser_suffix(raw: str) -> tuple[str, str | None]:
    """``{structure}_{left|right}`` -- DANDI 000469 / 001187 / 000673."""
    s = raw.lower().strip()
    hemisphere = None
    for suffix, hemi in (("_left", "left"), ("_right", "right")):
        if s.endswith(suffix):
            hemisphere = hemi
            s = s[: -len(suffix)]
            break
    return REGION_VOCABULARY.get(s, "other"), hemisphere


def _parse_brainnetome_hybrid(raw: str) -> tuple[str, str | None]:
    """"{CoarseCode}, {Hemisphere} ..." -- DANDI 000574 (Boran) / ds004752."""
    s = raw.strip()
    low = s.lower()
    if low in ("unspecific",):
        return "unspecific", None
    if low in ("no_label_found", "n/a", ""):
        return "unlabelled", None
    if s.lower() in SCALP_1020_LABELS:
        return "scalp", None
    coarse_code = s.split(",")[0].strip().lower()
    structure = _BRAINNETOME_COARSE_MAP.get(coarse_code, "other")
    hemisphere = None
    if "," in s:
        rest = s.split(",", 1)[1].strip().lower()
        if rest.startswith("left "):
            hemisphere = "left"
        elif rest.startswith("right "):
            hemisphere = "right"
    return structure, hemisphere


def _parse_desikan_killiany(raw: str) -> tuple[str, str | None]:
    """RAM ``ind.region`` (bare DK-aparc word) or ``das.region`` /
    ``stein.region`` / ``wb.region`` hand/subcortical annotation, with or
    without a leading hemisphere token -- ds005489 / ds005557.
    """
    s = raw.strip()
    low = s.lower()
    if low in ("n/a", ""):
        return "unlabelled", None
    if low == "misc":
        return "unspecific", None
    hemisphere = None
    body = low
    for prefix, hemi in (("left ", "left"), ("right ", "right")):
        if low.startswith(prefix):
            hemisphere = hemi
            body = low[len(prefix):]
            break
    if body in _DK_APARC_MAP:
        return _DK_APARC_MAP[body], hemisphere
    if body in _DK_HAND_ANNOTATION_MAP:
        return _DK_HAND_ANNOTATION_MAP[body], hemisphere
    for key in sorted(_DK_HAND_ANNOTATION_MAP, key=len, reverse=True):
        if key in body:
            return _DK_HAND_ANNOTATION_MAP[key], hemisphere
    return "other", hemisphere


# Per-corpus label parser dispatch, keyed by the ``label_convention`` each
# dataset declares in config/datasets.json -- selected declaratively, never
# sniffed from the raw string at runtime. One shared
# public entry point (:func:`normalize_region_label`); parsers stay private.
_LABEL_CONVENTION_PARSERS = {
    "nwb_structure_hemisphere_suffix": _parse_rutishauser_suffix,
    "nwb_boran_brainnetome_hybrid": _parse_brainnetome_hybrid,
    "bids_brainnetome_anatomical_location": _parse_brainnetome_hybrid,
    "bids_desikan_killiany_ind_region": _parse_desikan_killiany,
}


def normalize_region_label(
    raw: str, label_convention: str = "nwb_structure_hemisphere_suffix"
) -> tuple[str, str | None]:
    """Map one raw electrode/channel label to ``(structure, hemisphere)``.

    ``label_convention`` selects the per-corpus parser (declared in
    ``config/datasets.json``, not sniffed from ``raw``); it defaults to the
    original Rutishauser-lineage ``{structure}_{left|right}`` convention so
    every existing caller (000469/001187/000673) is unaffected. An
    unrecognized label maps to ``"other"`` with its raw string retained by
    the caller -- never silently absorbed into a known structure. Use
    :data:`COARSE_REGION_GROUP` to get the coarse grouping for the returned
    structure.
    """
    parser = _LABEL_CONVENTION_PARSERS.get(label_convention)
    if parser is None:
        raise ValueError(f"unknown label_convention {label_convention!r}")
    return parser(raw)


def resolve_unit_regions(
    f, label_convention: str = "nwb_structure_hemisphere_suffix"
) -> dict[str, NDArray]:
    """Per-unit anatomical region for every unit in an NWB ``units`` table.

    Resolves ``units/electrodes`` -> ``general/extracellular_ephys/electrodes/location``
    and normalizes each raw string through :func:`normalize_region_label`.
    Handles both the flat integer-column layout (``units/electrodes`` already
    one electrode-row index per unit, as used by 000469/001187/000673) and the
    ragged VectorIndex layout (``units/electrodes_index`` present, one or more
    electrode rows per unit — a unit spanning electrodes with disagreeing
    locations is region ``"other"`` with its raw strings joined, since a
    single unit cannot be assigned to two regions).

    Returns a dict of parallel arrays, one entry per unit, in unit order:
    ``region`` (controlled vocabulary), ``raw`` (original string(s), joined by
    ``"|"`` if more than one), ``hemisphere`` (``"left"``/``"right"``/``""``).

    Raises the underlying exception (annotated with the file name) rather than
    swallowing it — a file that cannot resolve region metadata must fail
    loudly, never be silently skipped or bucketed as unknown.
    """
    fname = getattr(getattr(f, "file", f), "filename", "<unknown file>")
    try:
        locations_raw = f["general/extracellular_ephys/electrodes/location"][:]
        locations = np.array(
            [x.decode() if isinstance(x, bytes) else x for x in locations_raw]
        )
        n_units = f["units/id"].shape[0]
        if "units/electrodes_index" in f:
            elec_flat = f["units/electrodes"][:]
            elec_index = f["units/electrodes_index"][:]
            region, raw, hemisphere = [], [], []
            prev = 0
            for idx in elec_index:
                rows = elec_flat[prev:idx]
                prev = idx
                locs = [locations[r] for r in rows]
                regions_hemis = [normalize_region_label(loc, label_convention) for loc in locs]
                uniq_regions = {r for r, _ in regions_hemis}
                if len(uniq_regions) == 1:
                    region.append(next(iter(uniq_regions)))
                    hemisphere.append(regions_hemis[0][1] or "")
                else:
                    region.append("other")
                    hemisphere.append("")
                raw.append("|".join(locs))
            return {
                "region": np.array(region),
                "raw": np.array(raw),
                "hemisphere": np.array(hemisphere),
            }
        else:
            elec_rows = f["units/electrodes"][:]
            locs = locations[elec_rows]
            resolved = [normalize_region_label(loc, label_convention) for loc in locs]
            return {
                "region": np.array([r for r, _ in resolved]),
                "raw": np.array(locs),
                "hemisphere": np.array([h or "" for _, h in resolved]),
            }
    except Exception as exc:
        raise RuntimeError(
            f"resolve_unit_regions failed on {fname!r}: {exc!r}"
        ) from exc


def unit_mean_firing_rates(
    spike_lists: list[NDArray],
    onsets: NDArray,
    window_s: float,
) -> NDArray:
    """Each unit's mean firing rate (Hz) across all trial windows
    (onset, onset + window_s), pooling spikes across trials."""
    onsets = np.asarray(onsets)
    total_time = len(onsets) * window_s
    rates = np.zeros(len(spike_lists))
    if total_time <= 0:
        return rates
    for u, spk in enumerate(spike_lists):
        n_spikes = sum(int(np.sum((spk >= t0) & (spk < t0 + window_s))) for t0 in onsets)
        rates[u] = n_spikes / total_time
    return rates


def low_rate_unit_mask(
    spike_lists: list[NDArray],
    onsets: NDArray,
    window_s: float,
    min_rate_hz: float = MIN_UNIT_FIRING_RATE_HZ,
) -> NDArray:
    """Boolean keep-mask: True where a unit's mean firing rate across all
    trial windows (onset, onset+window_s) is >= min_rate_hz."""
    if len(onsets) * window_s <= 0:
        return np.zeros(len(spike_lists), dtype=bool)
    return unit_mean_firing_rates(spike_lists, onsets, window_s) >= min_rate_hz
    return keep


def build_psth(
    spike_lists: list[NDArray],
    onsets: NDArray,
    bin_ms: float = BIN_MS_DEFAULT,
    smooth_ms: float = SMOOTH_MS_DEFAULT,
    window_s: float = 3.0,
) -> NDArray:
    """Smoothed per-trial firing-rate matrix aligned to `onsets`.

    Returns (n_trials, n_units, n_bins) in spikes/s.
    """
    bin_s = bin_ms / 1000.0
    edges = np.arange(0.0, window_s + bin_s, bin_s)
    n_bins = len(edges) - 1
    psth = np.zeros((len(onsets), len(spike_lists), n_bins), dtype=np.float32)
    for tr, t0 in enumerate(onsets):
        for u, spk in enumerate(spike_lists):
            counts, _ = np.histogram(spk - t0, bins=edges)
            psth[tr, u, :] = counts / bin_s
    if smooth_ms > 0:
        psth = gaussian_filter1d(psth, sigma=smooth_ms / bin_ms, axis=2)
    return psth


def zscore_psth(psth: NDArray) -> NDArray:
    """Legacy time-varying z-score, retained only for historical replay.

    This transform uses a separate mean/scale at every time bin and must not
    be used for current dynamics or cross-temporal inference.  New analyses
    should fit :class:`FrozenPSTHTransform` on each outer training fold.
    """
    mu = psth.mean(axis=0, keepdims=True)
    sd = psth.std(axis=0, keepdims=True) + 1e-8
    return (psth - mu) / sd


class FrozenPSTHTransform:
    """One fixed unit-wise transform fit on outer-training trials only.

    The same mean and scale are applied to every time bin, preserving the
    time coordinate required by a fixed state-space model.  ``fit`` may use a
    pre-specified baseline interval; otherwise it uses all train-fold bins.
    """

    def __init__(self, baseline_bins: NDArray | slice | None = None, eps: float = 1e-8):
        self.baseline_bins = baseline_bins
        self.eps = eps
        self.mean_: NDArray | None = None
        self.scale_: NDArray | None = None

    def fit(self, psth_train: NDArray) -> "FrozenPSTHTransform":
        values = np.asarray(psth_train, dtype=float)
        if values.ndim != 3:
            raise ValueError("psth_train must have shape (trials, units, time)")
        if self.baseline_bins is not None:
            values = values[:, :, self.baseline_bins]
        self.mean_ = values.mean(axis=(0, 2), keepdims=True)
        scale = values.std(axis=(0, 2), keepdims=True)
        # A constant training unit carries no standardized variance.  Give it
        # a neutral scale rather than dividing held-out deviations by epsilon.
        self.scale_ = np.where(scale > self.eps, scale, 1.0)
        return self

    def transform(self, psth: NDArray) -> NDArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FrozenPSTHTransform must be fit on training trials first")
        values = np.asarray(psth, dtype=float)
        if values.ndim != 3:
            raise ValueError("psth must have shape (trials, units, time)")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, psth_train: NDArray) -> NDArray:
        return self.fit(psth_train).transform(psth_train)


def fit_pca_psth(psth: NDArray, n_comp: int = N_PC_DEFAULT) -> tuple[NDArray, NDArray, float]:
    """PCA on (n_trials, n_units, n_bins) -> (n_trials, n_bins, n_comp) latent + loadings."""
    N, U, T = psth.shape
    X = psth.transpose(0, 2, 1).reshape(-1, U)
    X = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:n_comp].T
    Z_flat = X @ V
    Z = Z_flat.reshape(N, T, n_comp)
    var_total = np.linalg.norm(X) ** 2
    var_proj = np.linalg.norm(Z_flat) ** 2
    return Z, V, float(var_proj / (var_total + 1e-10))


def load_vs_load_ctg(
    psth_z: NDArray,
    loads: NDArray,
    low: int,
    high: int,
    n_components: int = N_PC_DEFAULT,
    ctg_step: int = 3,
    n_splits: int = 5,
    n_perm: int = 100,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Load-`low`-vs-load-`high` CTG with label-permutation null, or None if underpowered."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = (loads == low) | (loads == high)
    if min((loads[mask] == low).sum(), (loads[mask] == high).sum()) < 5:
        return None
    X = psth_z[mask]
    y = (loads[mask] == high).astype(int)
    t_idx = np.arange(0, psth_z.shape[2], ctg_step)
    res = ctg_label_permutation_null(
        X, y, t_idx, n_components=n_components, n_splits=n_splits, n_perm=n_perm, rng=rng
    )
    res["tau_info"] = temporal_stability_tau(res["auc_mat"])
    res["t_idx"] = t_idx
    return res


def item_identity_ctg(
    psth_z: NDArray,
    loads: NDArray,
    item_ids: NDArray,
    target_load: int = 1,
    min_trials: int = 20,
    n_components: int = N_PC_DEFAULT,
    ctg_step: int = 3,
    n_splits: int = 3,
    n_perm: int = 100,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Item-identity (content) CTG within a fixed load, or None if underpowered."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = loads == target_load
    if mask.sum() < min_trials:
        return None
    y = item_ids[mask]
    if len(np.unique(y)) < 2:
        return None
    class_counts = np.bincount(y - y.min())
    min_class = class_counts[class_counts > 0].min()
    n_splits_use = max(2, min(n_splits, int(min_class)))
    if min_class < 2:
        return None
    t_idx = np.arange(0, psth_z.shape[2], ctg_step)
    res = ctg_content_permutation_null(
        psth_z[mask], y, t_idx, n_components=min(n_components, mask.sum() - 2),
        n_splits=n_splits_use, n_perm=n_perm, rng=rng,
    )
    return res


def correct_error_drift(
    Z: NDArray,
    loads: NDArray,
    times: NDArray,
    window_s: float,
) -> NDArray:
    """Per-trial drift from the load-conditioned centroid (for pooled correct-vs-error tests).

    Returns drift for every trial; callers pair it with their own correct/error
    labels downstream (e.g. in a correct-vs-error LME), so no response-accuracy
    array is needed here.
    """
    return geometric_drift(Z, loads, times, maint_window=(0.0, window_s))


def pr_by_load(
    psth_z: NDArray,
    loads: NDArray,
    load_values: tuple[int, ...] = (1, 2, 3),
    min_trials: int = 5,
    n_splits: int = 2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cross-validated spatiotemporal PR (native unit space) per load level."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for ld in load_values:
        mask = loads == ld
        if mask.sum() < min_trials:
            continue
        out[ld] = spatiotemporal_participation_ratio(psth_z[mask], n_splits=n_splits, rng=rng)
    return out


def load_precomputed_session_geometry(results_dir, glob_pattern: str) -> dict[str, dict]:
    """Load every `*_geometry_{key}.npz` file already written by a per-dataset
    pipeline (e.g. run_000574_units_pipeline.py's `dandi000574_units_geometry_*`)
    into {key: {Z, correct, set_size, ...}}, keyed by session/subject string.

    Several downstream scripts (DPAD, DFINE-lite, NoMAD-lite dynamics checks)
    all need the same already-PCA-reduced per-trial latent trajectories and
    behavioral labels rather than raw spikes -- load once here instead of
    each re-deriving PCA from spike times.
    """
    from pathlib import Path
    sessions = {}
    for fp in sorted(Path(results_dir).glob(glob_pattern)):
        key = fp.stem.split("geometry_")[-1]
        with np.load(fp, allow_pickle=True) as d:
            sessions[key] = {k: d[k] for k in d.files}
    return sessions
