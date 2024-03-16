"""
Contains general utility functions for the package.
"""
from typing import Tuple

import numpy as np
from numpy import bool_

from pykda import constants


def is_stochastic_matrix(A: np.ndarray) -> bool:
    """
    Check if a given matrix is a stochastic matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        True if P is a stochastic matrix, False otherwise.

    """
    return is_nonnegative_matrix(A) and row_sums_are_1(A)


def is_nonnegative_matrix(A: np.ndarray) -> bool_:
    """
    Check if a given matrix is non-negative.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        True if A is non-negative, False otherwise.

    """
    return (A >= 0).all()


def has_positive_row_sums(A: np.ndarray) -> bool_:
    """
    Check if the row sums of a given matrix are positive.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        True if the row sums of A are positive, False otherwise.

    """
    return (A.sum(axis=1) > constants.VALUE_ZERO).all()


def row_sums_are_1(A: np.ndarray) -> bool:
    """
    Check if the row sums of a given matrix are equal to one.

    Parameters
    ----------
    A : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        True if the row sums of A are equal to one, False otherwise.

    """
    return np.all(np.abs(A.sum(axis=1) - 1) < constants.VALUE_ZERO)


def eigenvec_centrality(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the eigenvector centrality of a given non-negative adjacency matrix.

    I am assuming the matrix A contains one connected component.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix to calculate the eigenvector centrality of.

    Returns
    -------
    np.ndarray
        The eigenvector centrality of A.
    float
        The eigenvector centrality of A.
    """

    assert is_nonnegative_matrix(A), "Ensure the matrix elements are >= 0."

    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    max_idx = np.argmax(eigenvalues)

    return eigenvectors[:, [max_idx]], eigenvalues[max_idx]


def expand_matrix_with_row_and_column(A: np.ndarray) -> np.ndarray:
    """
    Expands the given matrix with an extra row and column of zeros at the start.

    Parameters
    ----------
    A : np.ndarray
        Matrix to which to add first row and column.

    Returns
    -------
    np.ndarray
        A where a first row and column of zeros is added at the start.

    """

    extra_column = np.zeros((A.shape[0], 1))
    A = np.hstack([extra_column, A])
    extra_row = np.zeros((1, A.shape[1]))
    return np.vstack([extra_row, A])


def create_graph_dict(A: np.ndarray) -> dict:
    """Creates a graph dictionary based upon adjacency matrix A, where each
    (i, j) for which A(i, j) > 0 is an edge by assumption.

    Parameters
    ----------
    A : np.ndarray
        An adjacency matrix.

    Returns
    -------
    graph : dict
        graph[i] gives a list of nodes that can be reached from node i.
    """

    return {
        i: np.where(A[i] > constants.VALUE_ZERO)[0].tolist()
        for i in range(len(A))
    }


def perturb_stochastic_matrix(
    P: np.ndarray, i: int, j: int, theta: float = 10 ** (-4)
) -> np.ndarray:
    """Perturbes P towards (i, j) with rate theta according to the method
    from Berkhout and Heidergott (2019) "Analysis of Markov influence graphs".

    Parameters
    ----------
    P : np.ndarray
        An adjacency matrix.
    i : int
        The row index of the perturbation.
    j : int
        The column index of the perturbation.
    theta : float
        The perturbation parameter.

    Returns
    -------
    np.ndarray
        P perturbed into the direction of (i, j) with rate theta.
    """

    P_perturbed = P.copy()
    P_perturbed[i, :] *= 1 - theta
    P_perturbed[i, j] += theta

    return P_perturbed
