"""
wm_dynamics — Neural manifold geometry and dynamics of working memory maintenance,
including a closed-loop stimulation (LQR/DARE) control layer and causal-inference
tools for testing geometry-conditioned treatment effects.

Modules
-------
preprocessing  : iEEG/NWB signal processing and dataset loaders
spike_pipeline : shared single-unit Sternberg WM pipeline (Rutishauser-lab datasets)
geometry       : dimensionality, subspace angles, participation ratio, CTG, RSA
dynamics       : DMD/ensemble DMD, EDMD/Koopman, SINDy, tangling, flow divergence
control        : controllability, LQR/DARE, minimum-energy control
statistics     : permutation tests, bootstrap CI, mixed-effects models, forest meta-analysis
causal         : cross-fit nuisances, AIPW ATE/CATE, DR-Learner CATE
neuroai        : CKA, Procrustes alignment
io_utils       : locked concurrent updates to shared results JSON
visualization  : Nature-style publication figures
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wm_dynamics")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
