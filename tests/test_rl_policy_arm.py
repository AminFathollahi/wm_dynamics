"""Self-check for scripts/run_rl_policy_arm.py's policy-gradient direction
search: on a synthetic unstable 2D plant with a KNOWN dominant eigenvector,
the learned steering direction should align with it (cos > 0.8) -- per
comments.txt 5A. This is the one required check; the Boran integration
(run_rl_arm_on_boran) needs real TES1/geometry data and is exercised by
actually running the script, not unit-tested here.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_rl_policy_arm import train_policy_direction, dominant_eigvec, _rollout_reward


def test_recovers_dominant_eigenvector_direction():
    # Unstable 2D plant: dominant eigenvalue 1.15 along [1,0], stable -0.3 along [0,1].
    A = np.array([[1.15, 0.0], [0.0, -0.3]])
    v_true = dominant_eigvec(A)
    assert np.allclose(np.abs(v_true), [1.0, 0.0], atol=1e-6)

    res = train_policy_direction(A, n_episodes=300, rng=np.random.default_rng(7))
    b = res["b"]
    cos = float(np.abs(b @ v_true))
    assert cos > 0.8, f"learned direction cos={cos:.3f} to true dominant eigenvector"


def test_reward_higher_when_aligned_with_instability():
    A = np.array([[1.2, 0.0], [0.0, -0.2]])
    rng = np.random.default_rng(0)
    x0 = np.array([0.4, 0.4])
    b_aligned = np.array([1.0, 0.0])
    b_orthogonal = np.array([0.0, 1.0])
    r_aligned = np.mean([_rollout_reward(A, b_aligned, 0.5, 25, 0.02, x0, 0.0, 2.0,
                                         np.random.default_rng(i)) for i in range(20)])
    r_orthogonal = np.mean([_rollout_reward(A, b_orthogonal, 0.5, 25, 0.02, x0, 0.0, 2.0,
                                            np.random.default_rng(i)) for i in range(20)])
    assert r_aligned > r_orthogonal


def test_reward_history_improves():
    A = np.array([[1.15, 0.0], [0.0, -0.3]])
    res = train_policy_direction(A, n_episodes=300, rng=np.random.default_rng(3))
    early = np.mean(res["reward_history"][:20])
    late = np.mean(res["reward_history"][-20:])
    assert late > early


if __name__ == "__main__":
    test_recovers_dominant_eigenvector_direction()
    test_reward_higher_when_aligned_with_instability()
    test_reward_history_improves()
    print("All rl_policy_arm self-checks passed.")
