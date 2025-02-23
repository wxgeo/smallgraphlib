from smallgraphlib.markov_chain import MarkovChain, gaussian_elimination


def test_markov_chain_from_string():
    # https://github.com/wxgeo/smallgraphlib/issues/2
    g = MarkovChain.from_string("A:B=0.6 B:A=0.9 A:A=0.4 B:B=0.1")
    assert g.nodes == ("A", "B")
    assert sorted(g.edges) == [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    assert g.probability("A", "B") == 0.6
    assert g.probability("B", "A") == 0.9
    assert g.probability("A", "A") == 0.4
    assert g.probability("B", "B") == 0.1
    assert g.transition_matrix == ((0.4, 0.6), (0.9, 0.1))
    assert g.stable_state == (0.6, 0.4)


def test_gaussian_elimination():
    matrix = [[1, 2, 3], [2, 2, 1]]
    gaussian_elimination(matrix)
    assert matrix == [[1, 0, -2], [0, 1, 2.5]]
