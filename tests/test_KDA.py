from math import ceil

import numpy as np
from pytest import approx, raises

from pykda.KDA import KDA
from pykda.Markov_chain import MarkovChain


def test_get_num_in_str_brackets():

    assert KDA.get_num_in_str_brackets("CO_A_1(2)") == 2
    assert KDA.get_num_in_str_brackets("CO_A_2(4)") == 4
    assert KDA.get_num_in_str_brackets("askjdfk289ur32inwefsf(4)ashdkfh=") == 4
    assert KDA.get_num_in_str_brackets("ask(100)ashdkfh=") == 100


def test_KDA_cut_edges():

    MC = MarkovChain(np.ones((5, 5)))

    # test cut_edges with args a tuple
    # ================================

    # asymmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", False, True)
    kda.cut_edges((0, 1))
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) == (0, 1):
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 1

    # symmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", True, True)
    kda.cut_edges((0, 1))
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) in {(0, 1), (1, 0)}:
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 2

    # test cut_edges with list of tuples
    # ==================================

    # asymmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", False, True)
    kda.cut_edges([(0, 2), (1, 3)])
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) in {(0, 2), (1, 3)}:
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 2

    # symmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", True, True)
    kda.cut_edges([(0, 2), (1, 3)])
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) in {(0, 2), (2, 0), (1, 3), (3, 1)}:
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 4

    # test cut_edges with args as two np.ndarrays / lists
    # ===================================================

    # asymmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", False, True)
    kda.cut_edges(np.array([0, 1]), np.array([2, 3]))
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) in {(0, 2), (1, 3)}:
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 2

    # symmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", True, True)
    kda.cut_edges([0, 1], [2, 3])
    for i in range(MC.num_states):
        for j in range(MC.num_states):
            if (i, j) in {(0, 2), (2, 0), (1, 3), (3, 1)}:
                assert approx(kda.MC.P[i, j]) == 0
                continue
            assert kda.MC.P[i, j] > 0
    assert len(kda.log["edges cut"]) == 2
    assert len(kda.log["edges cut"][1]) == 4


def test_KDA_criteria(example_unichains):
    for instance in example_unichains:

        MC = MarkovChain(instance["P"])

        # tests condition A
        # =================

        # condition CO_A_1
        for num_times in range(1, ceil(MC.num_states / 2)):

            # symmetric cut test
            kda = KDA(MC, f"CO_A_1({num_times})", "CO_B_1(1)", True, True)
            kda.run()
            assert len(kda.log["edges cut"]) == num_times + 1
            for i in range(num_times):
                assert len(kda.log["edges cut"][i + 1]) == 2

            # asymmetric cut test
            kda = KDA(MC, f"CO_A_1({num_times})", "CO_B_1(1)", False, True)
            kda.run()
            assert len(kda.log["edges cut"]) == num_times + 1
            for i in range(num_times):
                assert len(kda.log["edges cut"][i + 1]) == 1

        # condition CO_A_2
        for num_ergodic_classes in range(1, ceil(MC.num_states / 2)):

            # symmetric cut test
            kda = KDA(
                MC, f"CO_A_2({num_ergodic_classes})", "CO_B_1(1)", True, True
            )
            kda.run()
            assert kda.MC.num_ergodic_classes == num_ergodic_classes

            # asymmetric cut test
            kda = KDA(
                MC, f"CO_A_2({num_ergodic_classes})", "CO_B_1(1)", False, True
            )
            kda.run()
            assert kda.MC.num_ergodic_classes == num_ergodic_classes

        # condition CO_A_3
        for num_scc in range(1, ceil(MC.num_states / 2)):

            # symmetric cut test
            kda = KDA(MC, f"CO_A_3({num_scc})", "CO_B_1(1)", True, True)
            kda.run()
            assert kda.MC.num_strongly_connected_components == num_scc

            # asymmetric cut test
            kda = KDA(MC, f"CO_A_3({num_scc})", "CO_B_1(1)", False, True)
            kda.run()
            assert kda.MC.num_strongly_connected_components == num_scc

        # tests condition B
        # =================

        # condition CO_B_1
        for num_times in range(1, ceil(MC.num_states / 2)):

            # symmetric cut test
            kda = KDA(MC, "CO_A_1(1)", f"CO_B_1({num_times})", True, True)
            kda.run()
            assert len(kda.log["edges cut"]) == 2
            assert len(kda.log["edges cut"][1]) == num_times * 2

            # asymmetric cut test
            kda = KDA(MC, "CO_A_1(1)", f"CO_B_1({num_times})", False, True)
            kda.run()
            assert len(kda.log["edges cut"]) == 2
            assert len(kda.log["edges cut"][1]) == num_times

        # condition CO_B_2
        for num_ergodic_classes in range(1, ceil(MC.num_states / 2)):

            # symmetric cut test
            kda = KDA(
                MC, "CO_A_1(1)", f"CO_B_2({num_ergodic_classes})", True, True
            )
            kda.run()
            assert kda.MC.num_ergodic_classes == num_ergodic_classes

            # asymmetric cut test
            kda = KDA(
                MC, "CO_A_1(1)", f"CO_B_2({num_ergodic_classes})", False, True
            )
            kda.run()
            assert kda.MC.num_ergodic_classes == num_ergodic_classes

        # condition CO_B_3
        num_edges_smaller_than_0 = np.sum(
            np.logical_and(MC.Kemeny_constant_derivatives < 0, MC.P > 0)
        )
        kda = KDA(MC, "CO_A_1(1)", "CO_B_3(0)", False, True)
        kda.run()
        assert len(kda.log["edges cut"]) == 2
        assert len(kda.log["edges cut"][1]) == num_edges_smaller_than_0


def test_check_for_infinite_loops():
    MC = MarkovChain([[1, 0], [0, 1]])

    # test for infinite loop
    kda = KDA(MC, "CO_A_1(3)", "CO_B_1(1)", False, True)
    with raises(Warning):
        kda.run()


def test_wrong_conditions():
    MC = MarkovChain([[1, 0], [0, 1]])

    with raises(Exception):
        kda = KDA(MC, "CO_1A_1(3)", "CO_B_1(1)", False, True)
        kda.run()

    with raises(Exception):
        kda = KDA(MC, "CO_A_1(3)", " CO_B_1(1)", False, True)
        kda.run()

    with raises(Exception):
        kda = KDA(MC, "CO_A_1(3)", " CO_B_1(1)", False, True)
        kda.condition_B()


def test_wrong_cut_edges():

    MC = MarkovChain(np.ones((5, 5)))

    # test cut_edges with args a tuple
    # ================================

    # asymmetric cut test
    kda = KDA(MC, "CO_A_2(0)", "CO_B_1(1)", False, True)

    with raises(Exception):
        kda.cut_edges(np.array([0, 1]))

    with raises(Exception):
        kda.cut_edges(0, 1)

    with raises(Exception):
        kda.cut_edges([0, 1], [2, 3], [2, 3])
