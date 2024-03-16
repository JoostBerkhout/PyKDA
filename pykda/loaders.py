"""
Contains functions to load transition matrices from arrays or predefined csv's.
"""
from importlib.resources import read_text
from io import StringIO
from typing import List

import numpy as np

from pykda.constants import PRINT_NORMALIZATION_WARNINGS
from pykda.normalizers import normalizer_type, standard_row_normalization
from pykda.utilities import (
    has_positive_row_sums,
    is_nonnegative_matrix,
    is_stochastic_matrix,
)


def load_transition_matrix(
    A: np.ndarray | List[List],
    normalizer: normalizer_type = standard_row_normalization,
) -> np.ndarray:
    """
    Load a transition matrix from a given array. If the array is not a
    stochastic matrix, the function will try to normalize it.

    Parameters
    ----------
    A : Union[np.ndarray, List[List]]
        Array to be loaded as a Markov chain transition matrix.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Normalization function used to create a stochastic matrix from a matrix
        A, by default standard_row_normalization. See normalizers.py for
        pre-defined options.

    Returns
    -------
    np.ndarray
        Transition matrix of a Markov chain.

    """

    if not isinstance(A, np.ndarray):
        A = np.array(A)

    if is_stochastic_matrix(A):
        return A

    if PRINT_NORMALIZATION_WARNINGS:
        print("The given matrix is not stochastic: trying to normalize it.")

    if not is_nonnegative_matrix(A):
        if PRINT_NORMALIZATION_WARNINGS:
            print("Negative elements in matrix replaced by zeros.")
        A[A < 0] = 0

    if not has_positive_row_sums(A):
        if PRINT_NORMALIZATION_WARNINGS:
            print("Some row sums were zero: added self-loops.")
        rows_zero_sums = A.sum(axis=1) == 0
        A[rows_zero_sums, rows_zero_sums] = 1

    return normalizer(A)


def load_predefined_transition_matrix(
    name: str, normalizer: normalizer_type = standard_row_normalization
) -> np.ndarray:
    """
    Load a predefined csv file from the data folder as transition matrix. The
    data is normalized when needed.

    Parameters
    ----------
    name : str
        Name of the transition matrix to be loaded.
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Normalization function used to create a stochastic matrix from a matrix
        A, by default standard_row_normalization. See normalizers.py for
        pre-defined options.

    Returns
    -------
    np.ndarray
        Transition matrix of a Markov chain.

    """

    csv_string = read_text("pykda.data", f"{name}.csv")
    csv_file = StringIO(csv_string)
    data = np.genfromtxt(csv_file, delimiter=",")

    return load_transition_matrix(data)
