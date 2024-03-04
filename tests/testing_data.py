import numpy
import numpy as np
import pytest

from pykda.normalizers import standard_row_normalization


@pytest.fixture
def example_unichains():
    """Returns a list of Markov unichains with their properties."""

    instances = [
        {
            "source": "Kemeny and Snell (1976) Page 76",
            "name": "Land of Oz",
            "P": np.array(
                [[0.5, 0.25, 0.25], [0.5, 0, 0.5], [0.25, 0.25, 0.5]]
            ),
            "stationary distribution": np.array([[0.4], [0.2], [0.4]]),
            "ergodic projector": np.array(
                [[0.4, 0.2, 0.4], [0.4, 0.2, 0.4], [0.4, 0.2, 0.4]]
            ),
            "fundamental matrix": np.array(
                [[86, 3, -14], [6, 63, 6], [-14, 3, 86]]
            )
            / 75,
            "deviation matrix": (
                np.array([[86, 3, -14], [6, 63, 6], [-14, 3, 86]]) / 75
                - np.array([[0.4, 0.2, 0.4], [0.4, 0.2, 0.4], [0.4, 0.2, 0.4]])
            ),  # fundamental matrix - ergodic projector
            "mean first passage matrix": np.array(
                [[2.5, 4, 10 / 3], [8 / 3, 5, 8 / 3], [10 / 3, 4, 5 / 2]]
            ),
            "variance first passage matrix": np.array(
                [
                    [67 / 12, 12, 62 / 9],
                    [56 / 9, 12, 56 / 9],
                    [62 / 9, 12, 67 / 12],
                ]
            ),
            "ergodic classes": [[0, 1, 2]],
            "transient classes": [],
            "strongly connected components": [[0, 1, 2]],
            "weakly connected components": [[0, 1, 2]],
        },
        {
            "source": "Own example",
            "name": "Smallest absorbing example",
            "P": np.array([[1.0]]),
            "stationary distribution": np.array([[1]]),
            "ergodic projector": np.array([[1]]),
            "fundamental matrix": np.array([[1]]),
            "deviation matrix": np.array([[0]]),
            "mean first passage matrix": np.array([[1]]),
            "variance first passage matrix": np.array([[0]]),
            "ergodic classes": [[0]],
            "transient classes": [],
            "strongly connected components": [[0]],
            "weakly connected components": [[0]],
        },
        {
            "source": "Own example",
            "name": "Smallest unichain with transient state",
            "P": np.array([[1, 0], [0.5, 0.5]]),
            "stationary distribution": np.array([[1], [0]]),
            "ergodic projector": np.array([[1, 0], [1, 0]]),
            "fundamental matrix": np.array([[1, 0], [-1, 2]]),
            "deviation matrix": np.array([[0, 0], [-2, 2]]),
            "mean first passage matrix": None,
            "variance first passage matrix": None,
            "ergodic classes": [[0]],
            "transient classes": [[1]],
            "strongly connected components": [[0], [1]],
            "weakly connected components": [[0, 1]],
        },
        {
            "source": "Example 3 with p = 0.5 from Kemeny & Snell (1976) pages"
            " 27 & 90",
            "name": "Example 3 from Kemeny and Snell (1976) with p = 0.5",
            "P": np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0.5, 0, 0.5, 0, 0],
                    [0, 0.5, 0, 0.5, 0],
                    [0, 0, 0.5, 0, 0.5],
                    [0, 0, 1, 0, 0],
                ]
            ),
            "stationary distribution": np.array(
                [[0.1], [0.2], [0.4], [0.2], [0.1]]
            ),
            "ergodic projector": np.tile([0.1, 0.2, 0.4, 0.2, 0.1], (5, 1)),
            "fundamental matrix": np.array(
                [
                    [0.88, -0.04, 0.32, -0.04, -0.12],
                    [0.33, 0.86, 0.12, -0.14, -0.17],
                    [-0.02, 0.16, 0.72, 0.16, -0.02],
                    [-0.17, -0.14, 0.12, 0.86, 0.33],
                    [-0.12, -0.04, 0.32, -0.04, 0.88],
                ]
            ),
            "deviation matrix": None,
            "mean first passage matrix": np.array(
                [
                    [10, 4.5, 1, 4.5, 10],
                    [5.5, 5, 1.5, 5, 10.5],
                    [9, 3.5, 2.5, 3.5, 9],
                    [10.5, 5, 1.5, 5, 5.5],
                    [10, 4.5, 1, 4.5, 10],
                ]
            ),
            "variance first passage matrix": np.array(
                [
                    [66, 12.75, 0, 12.75, 66],
                    [53.25, 13, 0.25, 13, 66.25],
                    [66, 12.75, 0.25, 12.75, 66],
                    [66.25, 13, 0.25, 13, 53.25],
                    [66, 12.75, 0, 12.75, 66],
                ]
            ),
            "ergodic classes": [[0, 1, 2, 3, 4]],
            "transient classes": [[]],
            "strongly connected components": [[0, 1, 2, 3, 4]],
            "weakly connected components": [[0, 1, 2, 3, 4]],
        },
        {
            "source": "Example 7 from Kemeny & Snell (1976) page 90",
            "name": "Example 7 from Kemeny and Snell (1976)",
            "P": np.tile([0.1, 0.2, 0.4, 0.2, 0.1], (5, 1)),
            "stationary distribution": np.array(
                [[0.1], [0.2], [0.4], [0.2], [0.1]]
            ),
            "ergodic projector": np.tile([0.1, 0.2, 0.4, 0.2, 0.1], (5, 1)),
            "fundamental matrix": np.eye(5),
            "deviation matrix": None,
            "mean first passage matrix": np.tile([10, 5, 2.5, 5, 10], (5, 1)),
            "variance first passage matrix": np.tile(
                [90, 20, 3.75, 20, 90], (5, 1)
            ),
            "ergodic classes": [[0, 1, 2, 3, 4]],
            "transient classes": [[]],
            "strongly connected components": [[0, 1, 2, 3, 4]],
            "weakly connected components": [[0, 1, 2, 3, 4]],
        },
    ]

    # add simple example with 2 states
    values_P_12 = [0.1, 0.25, 0.6, 0.7]
    values_P_21 = [0.3, 0.5, 0.7, 0.9]
    for P_12, P_21 in zip(values_P_12, values_P_21):
        instances.append(
            {
                "source": "Own example",
                "name": f"Simple 2 states example with P_12 = {P_12} and P_21 ="
                f" {P_21}",
                "P": np.array([[1 - P_12, P_12], [P_21, 1 - P_21]]),
                "stationary distribution": np.array(
                    [[P_21 / (P_12 + P_21)], [P_12 / (P_12 + P_21)]]
                ),
                "ergodic projector": np.array(
                    [
                        [P_21 / (P_12 + P_21), P_12 / (P_12 + P_21)],
                        [P_21 / (P_12 + P_21), P_12 / (P_12 + P_21)],
                    ]
                ),
                "fundamental matrix": None,
                "deviation matrix": None,  # fundamental m. - ergodic projector
                "mean first passage matrix": np.array(
                    [
                        [(1 - P_12) * 1 + P_12 * (1 + 1 / P_21), 1 / P_12],
                        [1 / P_21, (1 - P_21) * 1 + P_21 * (1 + 1 / P_12)],
                    ]
                ),
                "variance first passage matrix": None,
                "ergodic classes": [[0, 1]],
                "transient classes": [],
                "strongly connected components": [[0, 1]],
                "weakly connected components": [[0, 1]],
            }
        )

    return instances


@pytest.fixture
def example_multichains():
    """Returns a list of Markov multichains with their properties."""

    return [
        {
            "source": "Own example",
            "name": "Smallest multichain without transient states",
            "P": np.array([[1, 0], [0, 1]]),
            "stationary distribution": None,
            "ergodic projector": np.array([[1, 0], [0, 1]]),
            "fundamental matrix": np.array([[1, 0], [0, 1]]),
            "deviation matrix": np.array(
                [[0, 0], [0, 0]]
            ),  # fundamental matrix - ergodic projector
            "mean first passage matrix": None,
            "variance first passage matrix": None,
            "ergodic classes": [[0], [1]],
            "transient classes": [],
            "strongly connected components": [[0], [1]],
            "weakly connected components": [[0], [1]],
        },
        {
            "source": "Own example",
            "name": "Smallest multichain with transient state",
            "P": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]),
            "stationary distribution": None,
            "ergodic projector": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]),
            "fundamental matrix": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "deviation matrix": np.array(
                [[0, 0, 0], [0, 0, 0], [-1, 0, 1]]
            ),  # fundamental matrix - ergodic projector
            "mean first passage matrix": None,
            "variance first passage matrix": None,
            "ergodic classes": [[0], [1]],
            "transient classes": [[2]],
            "strongly connected components": [[0], [1], [2]],
            "weakly connected components": [[0, 2], [1]],
        },
        {
            "source": "Own example",
            "name": "Smallest multichain with transient state flipped",
            "P": np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]),
            "stationary distribution": None,
            "ergodic projector": np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]),
            "fundamental matrix": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "deviation matrix": np.array(
                [[1, -1, 0], [0, 0, 0], [0, 0, 0]]
            ),  # fundamental matrix - ergodic projector
            "mean first passage matrix": None,
            "variance first passage matrix": None,
            "ergodic classes": [[1], [2]],
            "transient classes": [[0]],
            "strongly connected components": [[0], [1], [2]],
            "weakly connected components": [[0, 1], [2]],
        },
        {
            "source": "Own example",
            "name": "Smallest multichain with transient state equal prob.",
            "P": np.array([[1, 0, 0], [0, 1, 0], [1 / 3, 1 / 3, 1 / 3]]),
            "stationary distribution": None,
            "ergodic projector": np.array(
                [[1, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 0]]
            ),
            "fundamental matrix": None,
            "deviation matrix": None,
            "mean first passage matrix": None,
            "variance first passage matrix": None,
            "ergodic classes": [[0], [1]],
            "transient classes": [[2]],
            "strongly connected components": [[0], [1], [2]],
            "weakly connected components": [[0, 1, 2]],
        },
    ]


@pytest.fixture
def random_stochastic_matrices():
    """Returns a list of random stochastic matrices P."""

    random_Ps = []
    size = 10
    numb_rand_Ps = 10

    for seed in range(numb_rand_Ps):
        rng = np.random.default_rng(seed)
        random_matrix = rng.random((size, size))
        random_Ps.append(standard_row_normalization(random_matrix))

    return random_Ps
