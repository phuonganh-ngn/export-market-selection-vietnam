import numpy as np
import pytest

from experiments.decision_focused_xai_v1.src.decision_explainability import pairwise_contribution_difference


def test_pairwise_explanation_reconstructs_score_difference():
    phi_a = np.array([2.0, 1.0, 0.5])
    phi_b = np.array([1.0, 0.5, 0.0])
    diff, score_diff, reconstructed, error = pairwise_contribution_difference(phi_a, phi_b, 3.5, 1.5)
    assert diff.sum() == pytest.approx(score_diff)
    assert reconstructed == pytest.approx(2.0)
    assert error == pytest.approx(0.0)
