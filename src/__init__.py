"""
wm_dynamics — Neural manifold geometry of working memory maintenance.

Production library for the manuscript:
  'Manifold geometry predicts working memory maintenance load and representational
   conflict in human electrocorticography'

Modules
-------
preprocessing : iEEG signal processing pipeline
geometry      : dimensionality, subspace angles, participation ratio
dynamics      : DMD, trajectory tangling, system identification
control       : LQR rescue, controllability analysis
statistics    : permutation tests, bootstrap CI, mixed-effects models
visualization : Nature-style publication figures
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wm_dynamics")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
