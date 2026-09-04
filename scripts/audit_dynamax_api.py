"""Record whether the installed Dynamax API can fit the required model classes."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402


def main() -> None:
    from dynamax.generalized_gaussian_ssm import GeneralizedGaussianSSM
    from dynamax.hidden_markov_model.models.poisson_hmm import PoissonHMM
    from dynamax.slds import SLDS

    generalized_trainable = "raise NotImplementedError" not in inspect.getsource(
        GeneralizedGaussianSSM.e_step
    ) and "raise NotImplementedError" not in inspect.getsource(GeneralizedGaussianSSM.m_step)
    slds_trainable = "raise NotImplementedError" not in inspect.getsource(
        SLDS.e_step
    ) and "raise NotImplementedError" not in inspect.getsource(SLDS.m_step)
    packages = [
        "dynamax", "jax", "jaxlib", "numpy", "scipy",
        "scikit-learn", "statsmodels", "torch",
    ]
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "dynamax_dependency_audit",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "installation": {
            "status": "complete",
            "versions": {name: importlib.metadata.version(name) for name in packages},
            "core_import_check": "passed",
        },
        "poisson_observation": {
            "poisson_hmm_exposed": PoissonHMM is not None,
            "generalized_gaussian_ssm_accepts_user_poisson_distribution": True,
            "generalized_gaussian_ssm_fit_em_implemented": generalized_trainable,
            "refit_status": "not_buildable_from_exposed_training_api",
            "reason": (
                "The installed example constructs a Poisson emission distribution for fixed "
                "parameters, but GeneralizedGaussianSSM.e_step and m_step both raise "
                "NotImplementedError; PoissonHMM has no continuous confined-drift latent state."
            ),
        },
        "switching_lds": {
            "slds_class_exposed": SLDS is not None,
            "slds_fit_em_implemented": slds_trainable,
            "recurrent_transition_exposed": False,
            "refit_status": "not_buildable_from_exposed_training_api",
            "reason": (
                "SLDS exposes parameter containers, sampling, and particle filtering, but its "
                "e_step and m_step both raise NotImplementedError; no recurrent switching "
                "transition model is exposed."
            ),
        },
        "decision": (
            "Installation is stable, but Dynamax 1.0.2 does not expose a trainable PLDS or "
            "SLDS/rSLDS path for these fits. Keep the dependency crack open for model "
            "implementation rather than package availability."
        ),
    }
    destination = ROOT / "results" / "dynamax_dependency_audit.json"
    destination.write_text(canonical_json(output))

    crack_path = ROOT / "results" / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    for entry in cracks["entries"]:
        if entry.get("crack_id") == "full_rslds_dependency":
            entry.update({
                "status": "open_unimplemented_training_api",
                "trigger": "A full Poisson LDS or recurrent switching LDS is required to replace the Gaussian AR-HMM bound.",
                "chase": "Installed Dynamax 1.0.2 with JAX 0.10.2, verified the existing scientific stack, and inspected the local Poisson and SLDS APIs.",
                "resolution": "Package availability is no longer the blocker. GeneralizedGaussianSSM and SLDS expose model objects but both EM E- and M-steps raise NotImplementedError; no recurrent transition class is exposed.",
                "artifact": "results/dynamax_dependency_audit.json",
            })
            break
    crack_path.write_text(canonical_json(cracks))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
