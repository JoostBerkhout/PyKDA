import numpy as np

from pykda.loaders import (
    load_predefined_transition_matrix,
    load_transition_matrix,
)


def test_load_transition_matrix():
    A = [[1, 0, -1], [0.5, 0.5, 0], [0, 0, 0]]
    P = load_transition_matrix(A)
    assert np.allclose(P, np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]))


def test_load_predefined_transition_matrix():
    P1 = np.array([[0.5, 0.25, 0.25], [0.5, 0, 0.5], [0.25, 0.25, 0.5]])
    P2 = load_predefined_transition_matrix("land_of_Oz")
    assert np.allclose(P1, P2)
