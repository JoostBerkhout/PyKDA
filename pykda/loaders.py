from typing import Callable, List, Union

import numpy as np

from pykda import constants, utilities
from pykda.normalizers import standard_row_normalization

_normalizer_type = Callable[[np.ndarray], np.ndarray]


def load_transition_matrix(
    A: Union[np.ndarray, List[List]],
    normalizer: _normalizer_type = standard_row_normalization,
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

    if utilities.is_stochastic_matrix(A):
        return A

    if constants.PRINT_WARNINGS:
        print("The given matrix is not stochastic: trying to normalize it.")

    return normalizer(A)
