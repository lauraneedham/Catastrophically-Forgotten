import numpy as np
import pytest

from src.analysis.gradient_metrics import calculate_cosine_similarity, compute_SNR


def test_compute_SNR_known_value():
    # 3 items x 2 gradient dimensions.
    # dim 0: values [1, 2, 3] -> mean=2, std=sqrt(2/3)
    # dim 1: values [4, 4, 4] -> mean=4, std=0 (epsilon-guarded)
    data = np.array(
        [
            [1.0, 4.0],
            [2.0, 4.0],
            [3.0, 4.0],
        ]
    )
    epsilon = 1e-7
    expected_dim0 = 2.0 / (np.std([1.0, 2.0, 3.0]) + epsilon)
    expected_dim1 = 4.0 / (0.0 + epsilon)
    expected = np.mean([expected_dim0, expected_dim1])

    assert compute_SNR(data, epsilon=epsilon) == pytest.approx(expected)


def test_compute_SNR_zero_mean_gives_zero():
    # Symmetric values average to zero mean, so SNR should be ~0 regardless of spread.
    data = np.array([[-1.0], [1.0]])
    assert compute_SNR(data) == pytest.approx(0.0, abs=1e-6)


def test_calculate_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert calculate_cosine_similarity(v, v) == pytest.approx(1.0)


def test_calculate_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert calculate_cosine_similarity(a, b) == pytest.approx(0.0)


def test_calculate_cosine_similarity_opposite_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert calculate_cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_calculate_cosine_similarity_flattens_multidimensional_input():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0], [0.0, 4.0]])
    assert calculate_cosine_similarity(a, b) == pytest.approx(1.0)
