"""
Contains the MarkovChain class which captures a discrete time Markov chain with
given transition matrix P.
"""
import copy
from functools import cached_property
from typing import List, Tuple

import numpy as np
from pyvis.network import Network
from tarjan import tarjan

from pykda import constants
from pykda.loaders import (
    load_predefined_transition_matrix,
    load_transition_matrix,
)
from pykda.utilities import create_graph_dict


class MarkovChain:
    """
    Captures a discrete time Markov chain with given transition matrix P.

    It is set up using cached_property decorator so unnecessary recalculations
    are avoided (a lot of concepts depend on each other).

    Parameters
    ----------
    P : np.ndarray or list[list] or str
        Probability transition matrix. If a string is given, it is assumed it is
        the name of one of the predefined transition matrices from the folder
        data.

    """

    def __init__(self, P: np.ndarray | List[List] | str) -> None:
        self.P = P

    def __str__(self):  # pragma: no cover
        return (
            f"MC with {self.num_states} states.\n"
            f"Ergodic classes: {self.ergodic_classes}.\n"
            f"Transient classes: {self.transient_classes}."
        )

    def copy(self):
        return copy.deepcopy(self)

    @property
    def P(self):
        """Probability transition matrix."""

        return self._P

    @P.setter
    def P(self, P: np.ndarray | List[List] | str) -> None:
        """Setter for probability transition matrix."""

        if isinstance(P, str):
            self._P = load_predefined_transition_matrix(P)
        else:
            self._P = load_transition_matrix(P)

    @cached_property
    def num_states(self) -> int:
        """Number of states in the Markov chain."""

        return self.P.shape[0]

    @cached_property
    def eye(self) -> np.ndarray:
        """Identity matrix of the same shape as the transition matrix."""

        return np.eye(self.num_states)

    @cached_property
    def ones(self) -> np.ndarray:
        """Matrix of ones of the same shape as the transition matrix."""

        return np.ones((self.num_states, self.num_states))

    @cached_property
    def strongly_connected_components(self) -> list[list[int]]:
        """Strongly connected components of Markov chain as list of lists."""

        return tarjan(create_graph_dict(self.P))

    @cached_property
    def weakly_connected_components(self) -> list[list[int]]:
        """Weakly connected components of Markov chain as list of lists."""

        return tarjan(create_graph_dict(self.P + self.P.T))

    @cached_property
    def ergodic_classes(self) -> list[list[int]]:
        """Ergodic classes of the Markov chain as list of lists."""

        if hasattr(self, "_ergodic_classes"):
            return self._ergodic_classes

        # classify strongly connected components first
        self._classify_strongly_connected_components()
        return self._ergodic_classes

    @cached_property
    def transient_classes(self) -> list[list[int]]:
        """Transient classes of the Markov chain as list of lists. All states
        in the inner lists are transient and those within one inner list are
        strongly connected."""

        if hasattr(self, "_transient_classes"):
            return self._transient_classes

        # classify strongly connected components first
        self._classify_strongly_connected_components()
        return self._transient_classes

    @cached_property
    def ergodic_states(self) -> list:
        """Gives all ergodic states of the Markov chain as a list."""

        return [state for ec in self.ergodic_classes for state in ec]

    @cached_property
    def transient_states(self) -> list:
        """Gives all transient states of the Markov chain as a list."""

        return [state for tc in self.transient_classes for state in tc]

    @cached_property
    def num_strongly_connected_components(self) -> int:
        """Number of strongly connected components in the Markov chain."""

        return len(self.strongly_connected_components)

    @cached_property
    def num_weakly_connected_components(self) -> int:
        """Number of weakly connected components in the Markov chain."""

        return len(self.weakly_connected_components)

    @cached_property
    def num_ergodic_classes(self) -> int:
        """Number of ergodic classes in the Markov chain."""

        return len(self.ergodic_classes)

    @cached_property
    def num_transient_classes(self) -> int:
        """Number of transient classes in the Markov chain."""

        return len(self.transient_classes)

    @cached_property
    def num_ergodic_states(self) -> int:
        """Number of ergodic states in the Markov chain."""

        return len(self.ergodic_states)

    @cached_property
    def num_transient_states(self) -> int:
        """Number of transient states in the Markov chain."""

        return len(self.transient_states)

    @cached_property
    def is_unichain(self) -> bool:
        """True if the Markov chain is unichain (i.e., contains only one
        ergodic class), False otherwise."""

        return len(self.ergodic_classes) == 1

    @cached_property
    def is_multichain(self) -> bool:
        """True if the Markov chain is multichain (i.e., contains more than
        one ergodic class), False otherwise."""

        return len(self.ergodic_classes) > 1

    @cached_property
    def has_transient_states(self) -> bool:
        """True if the Markov chain has transient states, False otherwise."""

        return len(self.transient_states) > 0

    @cached_property
    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution (if it exists)."""

        assert self.is_unichain, "Stationary distribution does not exist."

        Z = self.P - self.eye
        Z[:, 0] = 1
        v = np.zeros((self.num_states, 1))
        v[0] = 1

        return np.linalg.solve(Z.T, v)

    @cached_property
    def Google_page_rank(self, d: float = 0.85) -> np.ndarray:
        """Google's PageRank of the Markov chain.

        Parameters
        ----------
        d : float
            Damping factor in (0, 1). Default is 0.85.

        """

        new_P = d * self.P + (1 - d) * self.ones / self.num_states

        return MarkovChain(new_P).stationary_distribution

    @cached_property
    def deviation_matrix_transient_part(self) -> np.ndarray:
        """Deviation matrix of the transient part of the Markov chain.

        The (i, j)th element in this matrix gives the number of expected visits
        to state j before leaving the transient part of the chain when starting
        in state i."""

        assert self.has_transient_states, "No transient states."

        tr_I = np.eye(self.num_transient_states)
        tr_P = self.P[np.ix_(self.transient_states, self.transient_states)]

        return np.linalg.inv(tr_I - tr_P)

    @cached_property
    def ergodic_projector(self) -> np.ndarray:
        """Ergodic projector of the Markov chain."""

        if self.is_unichain:
            return np.tile(self.stationary_distribution.T, (self.num_states, 1))

        assert self.is_multichain

        erg_proj = np.zeros((self.num_states, self.num_states))  # init

        for ec in self.ergodic_classes:

            ec_idxs = np.ix_(ec, ec)
            ec_P = self.P[ec_idxs]
            ec_stat_distr = MarkovChain(ec_P).stationary_distribution
            erg_proj[ec_idxs] = np.tile(ec_stat_distr.T, (len(ec), 1))

            if not self.has_transient_states:
                continue

            tr_to_ec_idxs = np.ix_(self.transient_states, ec)
            tr_to_ec_P = self.P[tr_to_ec_idxs]
            ec_erg_proj = erg_proj[ec_idxs]
            tr_dm = self.deviation_matrix_transient_part
            erg_proj[tr_to_ec_idxs] = tr_dm.dot(tr_to_ec_P).dot(ec_erg_proj)

        return erg_proj

    @cached_property
    def fundamental_matrix(self):
        """Fundamental matrix of the Markov chain."""

        return np.linalg.inv(self.eye - self.P + self.ergodic_projector)

    @cached_property
    def deviation_matrix(self):
        """Deviation matrix of the Markov chain."""

        return self.fundamental_matrix - self.ergodic_projector

    @cached_property
    def mean_first_passage_matrix(self):
        """Mean first passage matrix of the Markov chain. Element (i, j)
        gives the expected number of steps to reach state j from state i."""

        if self.is_multichain or self.has_transient_states:
            raise Exception(
                "Mean first passage matrix not defined. Note that "
                "you can calculate the mean first passage matrix "
                "per ergodic class."
            )

        # using notation of Kemeny and Snell (1976) Page 79
        Z = self.fundamental_matrix
        EZ_dg = np.tile(np.diag(Z), (self.num_states, 1))
        D = np.diag(1 / self.stationary_distribution.flatten())

        return (self.eye - Z + EZ_dg).dot(D)

    @cached_property
    def variance_first_passage_matrix(self):
        """Variance first passage matrix of the Markov chain. Element (i, j)
        gives the variance of the number of steps to reach state j from
        state i."""

        if self.is_multichain or self.has_transient_states:
            raise Exception(
                "Variance first passage matrix not defined. Note"
                " that you can calculate the variance first passage"
                " matrix per ergodic class."
            )

        # using notation of Kemeny and Snell (1976) Page 79
        M = self.mean_first_passage_matrix
        Z = self.fundamental_matrix
        Z_dg = np.diag(np.diag(Z))
        D = np.diag(1 / self.stationary_distribution.flatten())
        E_br_ZM_dg = np.tile(np.diag(Z.dot(M)), (self.num_states, 1))
        M_sq = M * M
        eye = self.eye

        return M.dot(2 * Z_dg.dot(D) - eye) + 2 * (Z.dot(M) - E_br_ZM_dg) - M_sq

    @cached_property
    def Kemeny_constant(self):
        """Kemeny constant of the Markov chain."""
        return np.trace(self.deviation_matrix) + 1

    @cached_property
    def Kemeny_constant_derivatives(self):
        """Kemeny constant derivatives of all Markov chain transitions. See
        Berkhout and Heidergott (2019) "Analysis of Markov influence graphs" for
        calculation details."""

        dev_mat_sq = np.linalg.matrix_power(self.deviation_matrix, 2)
        diag_P_dev_mat_sq = np.diag(self.P.dot(dev_mat_sq))[:, np.newaxis]

        return dev_mat_sq.T - np.tile(diag_P_dev_mat_sq, (1, self.num_states))

    def _classify_strongly_connected_components(self) -> None:
        """Classify the strongly connected components of the Markov chain
        into ergodic classes and transient classes and store them."""

        self._ergodic_classes = []
        self._transient_classes = []

        for scc in self.strongly_connected_components:
            mass_out = np.abs(np.sum(self.P[np.ix_(scc, scc)]) - len(scc))
            if mass_out < constants.VALUE_ZERO:
                # no 'outgoing' probabilities: scc is an ergodic class
                self._ergodic_classes.append(scc)
            else:
                self._transient_classes.append(scc)

    def sorted_edges(
        self, existing_edges_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the edges/transitions in order of how connecting they are
        from least to most connecting.

        How connecting they are is measure by the Kemeny constant derivatives.

        Parameters
        ----------
        existing_edges_only : bool
            If True (default), only existing edges are considered.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The ordered row and column indices, respectively, of the
            edges/connections in order of smallest Kemeny constant derivatives.
        """

        ordered_idxs = np.argsort(self.Kemeny_constant_derivatives, axis=None)

        if existing_edges_only:
            existing_idxs = np.flatnonzero(self.P > constants.VALUE_ZERO)
            ordered_idxs = [x for x in ordered_idxs if x in existing_idxs]

        row_idxs, col_idxs = np.unravel_index(ordered_idxs, self.P.shape)

        return row_idxs, col_idxs

    def most_connecting_edges(
        self, num_edges: int, only_existing_edges: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the num_edges most connecting edges.

        Parameters
        ----------
        num_edges : int
            Number of edges to return.
        only_existing_edges : bool
            If True, only existing edges are considered.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The num_edges row and column indices, respectively, of the
            edges/connections with the smallest Kemeny constant derivatives.
        """

        row_idxs, col_idxs = self.sorted_edges(only_existing_edges)

        if len(row_idxs) < num_edges:
            print(
                "Warning: # edges requested exceeds # edges in the matrix. "
                "Returned as many as possible."
            )
            num_edges = len(row_idxs)

        return row_idxs[:num_edges], col_idxs[:num_edges]

    def edges_below_threshold(
        self, threshold: float, only_existing_edges=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the edges with Kemeny constant derivatives < threshold.

        Parameters
        ----------
        threshold : float
            Threshold for selected edges.
        only_existing_edges : bool
            If True, only existing edges are considered.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The num_edges row and column indices, respectively, of the
            edges/connections with the smallest Kemeny constant derivatives.
        """

        row_idxs, col_idxs = self.sorted_edges(only_existing_edges)

        # find the num_edges edges with Kemeny constant derivative < threshold
        num_edges = 0
        for (i, j) in zip(row_idxs, col_idxs):
            if self.Kemeny_constant_derivatives[i, j] >= threshold:
                break
            num_edges += 1

        return row_idxs[:num_edges], col_idxs[:num_edges]

    def plot(
        self,
        file_name: str | None = None,
        labels: list[str] | None = None,
        hover_text: list[str] | None = None,
        notebook: bool = False,
        **kwargs,
    ) -> None:  # pragma: no cover
        """Plots the Markov chain as a directed graph.

        Parameters
        ----------
        file_name : str
            File name for the html file to be saved.
        labels : list[str]
            Labels for the states.
        hover_text : list[str]
            Text for the states which are visible when hovered over.
        notebook : bool
            If True, the graph is plotted in a Jupyter notebook, e.g., using
            Google Colab. Default is False.
        kwargs
            Additional keyword arguments to be passed to pyvis.

        """

        if file_name is None:
            file_name = "mygraph"

        if labels is None:
            labels = [str(i) for i in range(self.num_states)]

        if hover_text is None:
            hover_text = labels

        if notebook:
            net = Network(
                directed=True,
                notebook=True,
                cdn_resources="in_line",  # option needed in Google Colab
            )
        else:
            net = Network(directed=True)

        for i in range(self.num_states):
            net.add_node(
                i,
                value=self.Google_page_rank[i, 0],
                label=labels[i],
                title=hover_text[i],
            )

        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.P[i, j] > constants.VALUE_ZERO:
                    net.add_edge(
                        i,
                        j,
                        title=self.P[i, j],
                        weight=self.P[i, j],
                        arrows="to",
                    )

        net.force_atlas_2based()
        net.show(file_name + ".html", notebook=notebook)


if __name__ == "__main__":  # pragma: no cover

    P = np.array(
        [
            [0.0, 0.52493781, 0.47506219],
            [0.0, 0.66666667, 0.33333333],
            [0.06666667, 0.33333333, 0.6],
        ]
    )
    MC = MarkovChain(P)

    MC.plot(
        labels=["1", "2", "3"],
        hover_text=["A", "B", "C"],
    )

    MC = MarkovChain("land_of_Oz")
    MC.plot("land of Oz")

    MC = MarkovChain("Courtois_matrix")
    MC.plot("Courtois matrix")

    print(MC.ergodic_classes)
    print(MC.transient_classes)

    P = np.array(
        [
            [0, 0, 1, 0, 0],
            [0.5, 0, 0.5, 0, 0],
            [0, 0.5, 0, 0.5, 0],
            [0, 0, 0.5, 0, 0.5],
            [0, 0, 1, 0, 0],
        ]
    )
    MC = MarkovChain(P)
