import numpy as np
from pytest import approx

from pykda.Markov_chain import MarkovChain
from pykda.utilities import perturb_stochastic_matrix


def test_MarkovChain_stationary_distribution(example_unichains):
    for instance in example_unichains:
        MC = MarkovChain(instance["P"])
        assert np.allclose(
            MC.stationary_distribution, instance["stationary distribution"]
        )
        assert np.allclose(
            MC.stationary_distribution.T.dot(MC.P),
            MC.stationary_distribution.T,
        )
        assert MC.stationary_distribution.shape == (MC.num_states, 1)
        assert approx(np.sum(MC.stationary_distribution)) == 1
        assert np.min(MC.stationary_distribution) >= 0


def compare_lists_of_lists(list1, list2):
    """Helper function that compares two lists of lists and ignores order by
    sorting the inner and outer lists."""

    # sort the inner lists
    sorted_list1 = [sorted(inner_list) for inner_list in list1]
    sorted_list2 = [sorted(inner_list) for inner_list in list2]

    # sort the outer lists
    sorted_list1.sort()
    sorted_list2.sort()

    # compare the sorted lists
    return sorted_list1 == sorted_list2


def test_strongly_connected_components(example_unichains, example_multichains):
    for instance in example_unichains + example_multichains:
        ssc = MarkovChain(instance["P"]).strongly_connected_components
        assert compare_lists_of_lists(
            ssc, instance["strongly connected components"]
        )


def test_ergodic_and_transient_classes(example_multichains):
    for instance in example_multichains:
        MC = MarkovChain(instance["P"])
        assert compare_lists_of_lists(
            MC.ergodic_classes, instance["ergodic classes"]
        )
        assert compare_lists_of_lists(
            MC.transient_classes, instance["transient classes"]
        )
        assert len(MC.ergodic_classes) == MC.num_ergodic_classes
        assert len(MC.transient_classes) == MC.num_transient_classes


def test_ergodic_and_transient_states(example_unichains, example_multichains):
    for instance in example_unichains + example_multichains:
        ergodic_states = [s for ec in instance["ergodic classes"] for s in ec]
        transient_states = [
            s for tc in instance["transient classes"] for s in tc
        ]
        ergodic_states = sorted(ergodic_states)
        transient_states = sorted(transient_states)
        MC = MarkovChain(instance["P"])
        assert sorted(MC.ergodic_states) == ergodic_states
        assert sorted(MC.transient_states) == transient_states
        assert len(MC.ergodic_states) == MC.num_ergodic_states
        assert len(MC.transient_states) == MC.num_transient_states


def test_is_uni_or_multichain(example_unichains, example_multichains):
    for MC in example_unichains:
        assert MarkovChain(MC["P"]).is_unichain
        assert not MarkovChain(MC["P"]).is_multichain

    for MC in example_multichains:
        assert not MarkovChain(MC["P"]).is_unichain
        assert MarkovChain(MC["P"]).is_multichain


def test_ergodic_projector(example_unichains, example_multichains):
    for instance in example_unichains + example_multichains:
        MC = MarkovChain(instance["P"])
        assert np.allclose(MC.ergodic_projector, instance["ergodic projector"])


def test_fundamental_matrix(example_unichains, example_multichains):
    for instance in example_unichains + example_multichains:

        if instance["fundamental matrix"] is None:
            continue  # fundamental matrix is not specified

        MC = MarkovChain(instance["P"])
        assert np.allclose(
            MC.fundamental_matrix, instance["fundamental matrix"]
        )


def test_fundamental_matrix_via_inv(random_stochastic_matrices):
    for P in random_stochastic_matrices:
        MC = MarkovChain(P)
        inv_matrix = MC.eye - P + MC.ergodic_projector
        assert np.allclose(MC.fundamental_matrix.dot(inv_matrix), MC.eye)


def test_deviation_matrix(example_unichains, example_multichains):
    for instance in example_unichains + example_multichains:

        if instance["deviation matrix"] is None:
            continue  # deviation matrix is not specified

        MC = MarkovChain(instance["P"])
        assert np.allclose(MC.deviation_matrix, instance["deviation matrix"])


def test_mean_first_passage_matrix(example_unichains):
    for instance in example_unichains:

        if instance["mean first passage matrix"] is None:
            continue  # mean first passage matrix is not specified

        MC = MarkovChain(instance["P"])
        assert np.allclose(
            MC.mean_first_passage_matrix, instance["mean first passage matrix"]
        )


def test_variance_first_passage_matrix(example_unichains):
    for instance in example_unichains:

        if instance["variance first passage matrix"] is None:
            continue  # variance first passage matrix is not specified

        MC = MarkovChain(instance["P"])
        assert np.allclose(
            MC.variance_first_passage_matrix,
            instance["variance first passage matrix"],
        )


def test_Kemeny_constant_derivatives(example_unichains):
    for instance in example_unichains:

        P = instance["P"]
        MC = MarkovChain(P)

        if MC.has_transient_states:
            continue

        for i in range(MC.num_states):
            for j in range(MC.num_states):

                if P[i, j] == 0:
                    continue

                # calculate a finite difference (FD) approximation
                diff = 10 ** (-6)
                P_perturbed = perturb_stochastic_matrix(P, i, j, diff)
                MC_pert = MarkovChain(P_perturbed)
                FD_approx = (
                    MC_pert.Kemeny_constant - MC.Kemeny_constant
                ) / diff

                abs_diff = 100 * diff
                assert (
                    approx(MC.Kemeny_constant_derivatives[i, j], abs=abs_diff)
                    == FD_approx
                )
