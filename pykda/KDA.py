"""
Contains the Kemeny Decomposition Algorithm (KDA) class.
"""
import numpy as np

from pykda.Markov_chain import MarkovChain
from pykda.loaders import load_transition_matrix
from pykda.normalizers import normalizer_type, standard_row_normalization


class KDA:
    """
    Kemeny Decomposition Algorithm (KDA) class.

    KDA will iteratively cut 'edges' to decompose an original Markov chain (MC).
    An edge corresponds to a MC transition probability. The notation of Berkhout
    and Heidergott (2019) is used.

    Attributes
    ----------
    MC : MarkovChain
        Current Markov chain during KDA.
    log : dict
        Dictionary which logs the edges cut by KDA during the iterations. It
        also logs the Markov chains after each iteration. Each iteration of
        the inner-loop is stored by appending the list log['edges cut']. It is
        initialized with [None] to make the indexing easier. The Markov chains
        are stored in log['Markov chains'], where the original/initial MC is
        stored at index 0.

    Methods
    -------
    run()
        This will run KDA.
    cut_edges(*args)
        Allows one to cut manual edges in the current Markov chain to create
        a new Markov chain after normalization

    """

    def __init__(
        self,
        original_MC: MarkovChain,
        CO_A: str = "CO_A_1(1)",
        CO_B: str = "CO_B_3(0)",
        symmetric_cut: bool = False,
        verbose: bool = False,
        normalizer: normalizer_type = standard_row_normalization,
    ):
        """
        Parameters
        ----------
        original_MC : MarkovChain
            Original Markov chain object to which KDA will be applied.
        CO_A : str
            Condition of how often the Kemeny constant derivatives are being
            recalculated (outer while loop in KDA). The options are:

                - 'CO_A_1(i)' = Number of times performed < i
                - 'CO_A_2(E)' = Number of ergodic classes in current MC is < E
                - 'CO_A_3(C)' = Number of strongly connected components in
                                current MC is < C
        CO_B : str
            Condition of how many edges are being cut per iteration (inner while
            loop in KDA). The options are:

                - 'CO_B_1(e)' = Number of edges cut is < e
                - 'CO_B_2(E)' = Number of ergodic classes in MC is < E
                - 'CO_B_3(q)' = Not all edges with MC.KDer < q are cut
        symmetric_cut : bool
            If True, cutting (i, j) will also cut (j, i). If False, only (i, j).
        verbose : bool
            If true, information will be printed, else not.
        normalizer : normalizer_type
            Normalizer used to create a stochastic matrix from a matrix.
        """

        self.MC = original_MC.copy()
        self.log = {  # also log original MC after "iteration 0"
            "edges cut": [[None]],
            "Markov chains": [self.MC.copy()],
        }
        self.CO_A = CO_A
        self.CO_A_type = CO_A[:6]
        self.num_in_CO_A = self.get_num_in_str_brackets(CO_A)
        self.CO_B = CO_B
        self.CO_B_type = CO_B[:6]
        self.num_in_CO_B = self.get_num_in_str_brackets(CO_B)
        self.symmetric_cut = symmetric_cut
        self.normalizer = normalizer
        self.verbose = verbose

    @staticmethod
    def get_num_in_str_brackets(s):
        """Returns the number given between brackets in string s."""

        return int(s[s.find("(") + 1 : s.find(")")])

    def condition_A(self) -> bool:
        """Returns whether condition A is True or False."""

        if self.CO_A_type == "CO_A_1":
            return self.iterations_count < self.num_in_CO_A

        if self.CO_A_type == "CO_A_2":
            return self.MC.num_ergodic_classes < self.num_in_CO_A

        if self.CO_A_type == "CO_A_3":
            return self.MC.num_strongly_connected_components < self.num_in_CO_A

        raise Exception("Unknown condition A chosen (CO_A).")

    def condition_B(self) -> bool:
        """Returns whether condition B is True or False."""

        if self.CO_B_type == "CO_B_2":
            return self.MC.num_ergodic_classes < self.num_in_CO_B

        raise Exception("Unknown condition B chosen (CO_B).")

    def run(self) -> None:
        """Runs KDA with conditions CO_A and CO_B."""

        self.iterations_count = 0  # of the outer while loop of KDA

        # start cutting till condition A fails
        while self.condition_A():

            self.cut_till_condition_B_fails()
            self.iterations_count += 1

    def cut_till_condition_B_fails(self):
        """Cuts edges till condition B fails."""

        if self.CO_B_type == "CO_B_1":

            edges = self.MC.most_connecting_edges(self.num_in_CO_B)
            self.cut_edges(*edges)

        elif self.CO_B_type == "CO_B_2":

            row_indexes, col_indexes = self.MC.sorted_edges()
            num_cut = 0

            while self.condition_B():  # inner while loop
                self.cut_edges(
                    row_indexes[num_cut : num_cut + 1],
                    col_indexes[num_cut : num_cut + 1],
                )
                num_cut += 1

        elif self.CO_B_type == "CO_B_3":

            edges = self.MC.edges_below_threshold(self.num_in_CO_B)
            self.cut_edges(*edges)

        else:
            raise Exception("Unknown condition B chosen (CO_B).")

    def cut_edges(self, *args):
        """Cut given edges in the Markov chain and normalize afterward.

        There are different options to specify which edges to cut via args.
        When self.symmetric_cut = True, also the reversed edges are cut.

        Parameters
        ----------
        args :
            There are three options for args:
                1. One tuple of length 2 indicating which edge to cut.
                2. One list of tuples of edges which to cut.
                3. Two lists or np.ndarrays indicating which edges to cut.

        """

        if len(args) == 1:
            if isinstance(args[0], tuple):
                self._cut_edges([args[0][0]], [args[0][1]])
            elif isinstance(args[0], list):
                assert all(isinstance(x, tuple) for x in args[0])
                row_indexes, col_indexes = map(list, zip(*args[0]))
                self._cut_edges(row_indexes, col_indexes)
            else:
                raise Exception(
                    "Expected list of tuples or tuple in case "
                    "one argument is given in KDA.cut_edges()."
                )
        elif len(args) == 2:
            self._cut_edges(args[0], args[1])
        else:
            raise Exception("Expected 1 or 2 arguments in KDA.cut_edges().")

    def _cut_edges(
        self, row_indexes: np.ndarray | list, col_indexes: np.ndarray | list
    ) -> None:
        """Cut given edges in the Markov chain and normalize afterward.

        When self.symmetric_cut = True, also the reversed edges are cut.

        Parameters
        ----------
        row_indexes : np.ndarray or list
            Row indexes of the edges to be cut.
        col_indexes : np.ndarray or list
            Columns indexes of the edges to be cut.

        """

        if self.symmetric_cut:
            row_indexes, col_indexes = np.concatenate(
                (row_indexes, col_indexes)
            ), np.concatenate((col_indexes, row_indexes))

        self.MC.P[row_indexes, col_indexes] = 0  # cut edges
        new_P = load_transition_matrix(self.MC.P, self.normalizer)
        self.MC = MarkovChain(new_P)
        self.log_edges(row_indexes, col_indexes)
        self.check_for_infinite_loops()

    def log_edges(
        self, row_indexes: np.ndarray | list, col_indexes: np.ndarray | list
    ) -> None:
        """Logs the edges that have been cut.

        Parameters
        ----------
        row_indexes : np.ndarray or list
            Row indexes of the edges that are cut.
        col_indexes : np.ndarray or list
            Columns indexes of the edges that are cut.
        """

        edges = [(x, y) for x, y in zip(row_indexes, col_indexes)]
        self.log["edges cut"].append(edges)
        self.log["Markov chains"].append(self.MC.copy())

    def report(self):  # pragma: no cover
        """Prints a report of KDA."""

        # init
        edges_cut = self.log["edges cut"]
        MCs = self.log["Markov chains"]

        print("\nKDA report:")
        print("===========\n")

        print("KDA settings:")
        print(f"- Condition A: {self.CO_A} ({self.condition_A_translator()})")
        print(f"- Condition B: {self.CO_B} ({self.condition_B_translator()})")
        print(f"- Symmetric cut: {self.symmetric_cut}\n")

        print("Initial Markov chain (MC) is:")
        print(f"{MCs[0]}\n")

        print("KDA progress is:")
        for iteration_num, (edges_cut, MC) in enumerate(zip(edges_cut, MCs)):
            print(f"* Iteration {iteration_num}:")
            print(f"   - Edges cut: {edges_cut}")
            print(f"   - Ergodic classes: {MC.ergodic_classes}")
            print(f"   - Transient classes: {MC.transient_classes}")
            scc = MC.strongly_connected_components
            print(f"   - Strongly connected components: {scc}")
            wcc = MC.weakly_connected_components
            print(f"   - Weakly connected components: {wcc}")

    def check_for_infinite_loops(self):
        """Due to fixes for normalization, it may happen that the same edge is
        cut over and over again. This check raises an error in that case.
        """

        if len(self.log["edges cut"]) > 2:
            edges_cut_now = set(self.log["edges cut"][-1])
            edges_cut_prev = set(self.log["edges cut"][-2])
            edges_cut_prev2 = set(self.log["edges cut"][-3])
            same_edges_cut = edges_cut_now & edges_cut_prev & edges_cut_prev2
            if len(same_edges_cut) > 0:
                raise Warning(
                    "Possible infinite KDA loop detected: the following edges "
                    f"may be cut over and over again: {same_edges_cut}. "
                    f"If you see this message more than two times, try using"
                    f" another KDA setting that cuts less edges to avoid this."
                )

    @staticmethod
    def get_num_in_brackets(s):
        """Gets the number given between brackets in string s."""

        return int(s[s.find("(") + 1 : s.find(")")])

    def condition_A_translator(self):  # pragma: no cover
        """Translates the condition A into something readable."""

        if self.CO_A_type == "CO_A_1":
            return f"continue till {self.num_in_CO_A} times performed"

        if self.CO_A_type == "CO_A_2":
            return f"continue till # ergodic classes is {self.num_in_CO_A}"

        if self.CO_A_type == "CO_A_3":
            return (
                f"continue till # strongly connected components "
                f"{self.num_in_CO_A}"
            )

        raise Exception("Unknown condition A chosen (CO_A).")

    def condition_B_translator(self):  # pragma: no cover
        """Translates the condition B into something readable."""

        if self.CO_B_type == "CO_B_1":
            return f"continue till {self.num_in_CO_B} edge(s) cut"

        if self.CO_B_type == "CO_B_2":
            return f"continue till # ergodic classes is {self.num_in_CO_B}"

        if self.CO_B_type == "CO_B_3":
            return (
                f"continue till all edges with Kemeny constant derivative"
                f" < {self.num_in_CO_B} are cut"
            )

        raise Exception("Unknown condition B chosen (CO_B).")

    def plot_progress(self, **kwargs):  # pragma: no cover
        """Plots the Markov chains in the log. The kwargs are passed to the
        plot method of the Markov chain. Refer to MarkovChain.plot() for more
        details."""

        for i, MC in enumerate(self.log["Markov chains"]):
            MC.plot(**kwargs)

    def plot(self, **kwargs):  # pragma: no cover
        """Plots the Markov chain after KDA. Refer to MarkovChain.plot() for
        more details."""

        self.MC.plot(**kwargs)


if __name__ == "__main__":  # pragma: no cover

    # the following is used for testing purposes

    # MC = MarkovChain(
    #     np.array([[0.5, 0.25, 0.25], [0.5, 0, 0.5], [0.25, 0.25, 0.5]])
    # )
    # # MC = MarkovChain(
    # #     np.array(
    # #         [[1, 0], [0.5, 0.5]]
    # #         )
    # #     )
    #
    # kda = KDA(MC, 'CO_A_1(1)', f'CO_B_3(0)', False, True)
    # kda.cut_edges([0, 1], [1, 2])
    #
    MC = MarkovChain("Courtois_matrix")
    kda = KDA(
        original_MC=MC, CO_A="CO_A_1(1)", CO_B="CO_B_3(0)", symmetric_cut=False
    )
    kda.run()
    kda.plot(file_name="Courtois_matrix_after_KDA_1_0")

    kda2 = KDA(
        original_MC=MC, CO_A="CO_A_2(3)", CO_B="CO_B_1(1)", symmetric_cut=False
    )
    kda2.run()
    kda2.plot(file_name="Courtois_matrix_after_KDA_2_1")

    name = "Zacharys_karate_club"
    MC = MarkovChain(name)  # load the pre-defined Courtois matrix
    MC.plot(file_name=name)

    kda = KDA(
        original_MC=MC, CO_A="CO_A_1(1)", CO_B="CO_B_3(0)", symmetric_cut=False
    )
    kda.run()
    kda.plot(file_name="Zachary_after_KDA_1_0")
