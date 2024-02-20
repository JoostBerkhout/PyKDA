"""
Contains a collection of matrix normalizers, i.e., functions that take a matrix
as input and return a normalized/stochastic matrix version of it.
"""
import numpy as np

from pykda.utilities import (
    eigenvec_centrality,
    expand_matrix_with_row_and_column,
    is_nonnegative_matrix,
)


def standard_row_normalization(A: np.ndarray) -> np.ndarray:
    """
    Normalize the rows of a given matrix so that they sum up to one.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be normalized.

    Returns
    -------
    np.ndarray
        Normalized matrix.

    """

    assert is_nonnegative_matrix(A), "Ensure the matrix elements are >= 0."

    return A / A.sum(axis=1, keepdims=True)


def normalization_with_self_loops(A: np.ndarray) -> np.ndarray:
    """
    Add self-loops to each node so that the row sums of A are equal. Afterward
    apply standard row normalization.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be normalized.

    Returns
    -------
    np.ndarray
        Normalized matrix.

    """

    assert is_nonnegative_matrix(A), "Ensure the matrix elements are >= 0."

    rows_sums = A.sum(axis=1, keepdims=True)
    max_row_sum = np.max(rows_sums)
    self_loops_matrix = np.diag(max_row_sum - rows_sums)

    return standard_row_normalization(A + self_loops_matrix)


def normalization_same_eigenvec_centr(A: np.ndarray) -> np.ndarray:
    """
    Add a dummy node to the matrix and normalize it so that the row sums are
    equal to one. The dummy node is added in such a way that the first
    eigenvector of the normalized matrix is the same as the first eigenvector
    of the original matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be normalized.

    Returns
    -------
    np.ndarray
        Normalized matrix.

    """

    rows_sums = A.sum(axis=1, keepdims=True)
    max_row_sum = np.max(rows_sums)

    A_with_dummy = expand_matrix_with_row_and_column(A)  # added at beginning
    eigvec_centr, eigval = eigenvec_centrality(A)
    A_with_dummy[0, 1:] = eigvec_centr.T
    A_with_dummy[1:, 0] = max_row_sum - rows_sums

    return standard_row_normalization(A_with_dummy)
