import numpy as np
from pytest import approx

from pykda.utilities import (
    create_graph_dict,
    eigenvec_centrality,
    expand_matrix_with_row_and_column,
    has_positive_row_sums,
    is_nonnegative_matrix,
    is_stochastic_matrix,
    row_sums_are_1,
)


def test_is_stochastic_matrix(example_unichains, example_multichains):

    for MC in example_unichains:
        assert is_stochastic_matrix(MC["P"])

    for MC in example_multichains:
        assert is_stochastic_matrix(MC["P"])

    assert not is_stochastic_matrix(np.array([[0.5, 0.5], [0.5, 0.4]]))
    assert not is_stochastic_matrix(np.array([[0.5, 0.5], [-0.5, 0.5]]))
    assert not is_stochastic_matrix(np.array([[0.5, 0.5], [-1, 2]]))


def test_is_nonnegative_matrix(example_unichains, example_multichains):

    for MC in example_unichains:
        assert is_nonnegative_matrix(MC["P"])

    for MC in example_multichains:
        assert is_nonnegative_matrix(MC["P"])

    assert is_nonnegative_matrix(np.array([[0.5, 0.5], [0.5, 10]]))
    assert is_nonnegative_matrix(np.array([[0.5, 0.5], [10, 0.5]]))
    assert not is_nonnegative_matrix(np.array([[0.5, 0.5], [-1, 2]]))
    assert not is_nonnegative_matrix(np.array([[0.5, -0.5], [1, 2]]))


def test_has_positive_row_sums(example_unichains, example_multichains):

    for MC in example_unichains:
        assert has_positive_row_sums(MC["P"])

    for MC in example_multichains:
        assert has_positive_row_sums(MC["P"])

    assert not has_positive_row_sums(np.array([[0, 0], [0.6, 0.4]]))
    assert has_positive_row_sums(np.array([[0, 0, 0.1], [0.6, 0.4, 0.3]]))
    assert not has_positive_row_sums(np.array([[0, 0, 0.1], [0, 0, 0]]))
    assert not has_positive_row_sums(np.array([[0, 0, 0], [0, 0, 0]]))
    assert not has_positive_row_sums(
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )


def test_row_sums_are_1(example_unichains, example_multichains):

    for MC in example_unichains:
        assert is_nonnegative_matrix(MC["P"])

    for MC in example_multichains:
        assert is_nonnegative_matrix(MC["P"])

    assert row_sums_are_1(np.array([[0.5, 0.5], [0.6, 0.4]]))
    assert row_sums_are_1(np.array([[0.67, 0.33], [0, 1]]))
    assert row_sums_are_1(np.array([[-0.25, 1.25], [0, 1]]))
    assert row_sums_are_1(np.array([[0.5, 0.5], [-1, 2]]))
    assert not row_sums_are_1(np.array([[0.5, 0.5], [0.5, 10]]))
    assert not row_sums_are_1(np.array([[0.5, -0.5], [1, 2]]))
    assert not row_sums_are_1(np.array([[0.5, 0.5], [0.99, 0.02]]))


def test_eigenvec_centrality(example_unichains):

    for MC in example_unichains:
        eigvec_centr, eigval = eigenvec_centrality(MC["P"])
        scaled_eigvec_centr = eigvec_centr / np.sum(eigvec_centr)
        assert approx(eigval) == 1
        assert approx(scaled_eigvec_centr) == MC["stationary distribution"]

    for seed in range(10):
        rng = np.random.default_rng(seed)
        A_random = rng.random((5, 5))
        eigvec_centr, eigval = eigenvec_centrality(A_random)
        assert approx(eigvec_centr.T.dot(A_random)) == eigval * eigvec_centr.T


def test_expand_matrix_with_row_and_column(
    example_unichains, example_multichains
):

    for MC in example_unichains:
        P_extended = expand_matrix_with_row_and_column(MC["P"])
        assert np.all(P_extended[:, 0] == 0)
        assert np.all(P_extended[0, :] == 0)
        assert np.all(P_extended[1:, 1:] == MC["P"])


def test_create_graph_dict():

    assert create_graph_dict(np.array([[1, 0], [0, 1]])) == {0: [0], 1: [1]}
    assert create_graph_dict(np.array([[0, 0], [0, 0]])) == {0: [], 1: []}
    assert create_graph_dict(np.array([[0, 1], [1, 0]])) == {0: [1], 1: [0]}
    assert create_graph_dict(np.array([[0, 1], [1, 0], [0, 0]])) == {
        0: [1],
        1: [0],
        2: [],
    }
