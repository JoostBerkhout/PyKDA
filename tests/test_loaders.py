import numpy as np

from pykda.loaders import load_transition_matrix


def test_load_transition_matrix():
    A = [[1, 0, -1], [0.5, 0.5, 0], [0, 0, 0]]
    P = load_transition_matrix(A)
    assert np.allclose(P, np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]))
