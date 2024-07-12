import numpy as np

from pykda.normalizers import (
    normalization_same_eigenvec_centr,
    normalization_with_self_loops,
    standard_row_normalization,
)


def test_standard_row_normalization():
    A = np.array([[1, 0, 1], [0.5, 0.5, 0], [0, 0, 100]])
    P = standard_row_normalization(A)
    assert np.allclose(P, np.array([[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0, 1]]))


def test_normalization_with_self_loops():
    for a in [0, 0.1, 0.5, 1, 10]:
        A = np.array([[0, 1, 0], [0, 0, 2 + a], [1, 1, 0]])
        P = normalization_with_self_loops(A)
        expected_P = np.array([[1 + a, 1, 0], [0, 0, 2 + a], [1, 1, a]]) / (
            2 + a
        )
        assert np.allclose(P, expected_P)


def test_normalization_same_eigenvec_centr():
    A = np.array([[0, 1, 0], [0, 0, 2], [1, 1, 0]])
    expected_P = np.array(
        [
            [0, 0.231, 0.361, 0.408],
            [0.5, 0, 0.5, 0],
            [0, 0, 0, 1],
            [0, 0.5, 0.5, 0],
        ]
    )
    P = np.round(normalization_same_eigenvec_centr(A), 3)  # know precision
    assert np.allclose(P, expected_P)
