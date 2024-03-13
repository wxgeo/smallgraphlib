import math

import pytest

from smallgraphlib import WeightedGraph, WeightedDirectedGraph, InvalidGraphAttribute


def test_weighted_graph():
    g = WeightedGraph([1, 2, 3, 4, 5], (1, 2, 10.0), (2, 3, 9.0), (2, 4, 7.0), (4, 5, 8.0))
    assert g.successors(1) == {2}
    assert g.weight(2, 1) == 10.0
    assert g.total_weight == 10.0 + 9.0 + 7.0 + 8.0
    assert g.copy() == g


def test_weighted_graph_from_matrix():
    oo = math.inf
    M = (
        (0, 2, oo, oo, 17, oo, 5),
        (4, 0, 5, 4, oo, oo, 4),
        (7, 8, 0, 13, oo, 8, 15),
        (17, oo, oo, 0, 7, 9, oo),
        (6, 8, 14, oo, 0, 21, 9),
        (6, 10, 8, 9, 7, 0, oo),
        (oo, 15, 5, 7, 18, oo, 0),
    )
    g = WeightedDirectedGraph.from_matrix(M)
    assert g.weight(1, 5) == 17
    assert g.weight(1, 7) == 5
    assert g.weight(7, 1) == oo

    def convert(val):
        return 0 if (math.isinf(val) or val == 0) else 1

    assert g.adjacency_matrix == tuple(tuple(convert(value) for value in line) for line in M)


def test_weighted_graph_from_sympy_matrix():
    import sympy

    oo = sympy.oo
    M = sympy.Matrix(
        (
            (0, 2, oo, oo, 17, oo, 5),
            (4, 0, 5, 4, oo, oo, 4),
            (7, 8, 0, 13, oo, 8, 15),
            (17, oo, oo, 0, 7, 9, oo),
            (6, 8, 14, oo, 0, 21, 9),
            (6, 10, 8, 9, 7, 0, oo),
            (oo, 15, 5, 7, 18, oo, 0),
        )
    )
    g = WeightedDirectedGraph.from_matrix(M, nodes_names="ABCDEFG")
    assert g.weight("A", "E") == 17
    assert g.nodes == tuple("ABCDEFG")
    assert g.distance("A", "A") == 0
    assert g.distance("A", "B") == 2
    assert g.distance("A", "G") == 5
    assert g.distance("A", "D") == 6
    assert g.distance("A", "C") == 7
    assert g.distance("A", "E") == 13
    assert g.distance("A", "F") == 15


def test_minimum_spanning_tree():
    g = WeightedGraph([1])
    assert g.minimum_spanning_tree() == g
    g = WeightedGraph([1, 2, 3, 4, 5], (1, 2, 10.0), (2, 3, 9.0), (2, 4, 7.0), (4, 5, 8.0))
    assert g.minimum_spanning_tree() == g
    g = WeightedGraph.from_dict(
        AB=15,
        AE=8,
        AG=6,
        AF=13,
        BC=10,
        BD=13,
        BG=14,
        CD=12,
        CF=11,
        DE=11,
        DF=5,
        DG=12,
        EF=5,
        EG=10,
    )
    assert g.degree == 14
    assert g.minimum_spanning_tree().total_weight == 45
    g = WeightedGraph.from_dict(
        AB=12,
        AC=20,
        AD=9,
        BF=13,
        CD=8,
        CF=2,
        CG=11,
        DG=21,
        EF=9,
        EG=3,
        FG=5,
    )
    assert g.degree == 11
    assert g.minimum_spanning_tree().total_weight == 39
    # Not connected graph
    g = WeightedGraph((1, 2))
    with pytest.raises(InvalidGraphAttribute):
        g.minimum_spanning_tree()


def test_shortest_paths():
    g = WeightedGraph.from_dict(
        AB=1,
        BG=9,
        FC=1,
        FD=5,
        AG=9,
        BC=8,
        AF=5,
        BF=7,
        BD=12,
        CD=3,
        FE=2,
        ED=7,
        GE=1,
        GD=2,
        AE=3,
    )
    assert {node: g.node_degree(node) for node in g.nodes} == {
        "A": 4,
        "B": 5,
        "C": 3,
        "D": 5,
        "E": 4,
        "F": 5,
        "G": 4,
    }
    assert g.shortest_paths("D", "B") == (7, [["D", "G", "E", "A", "B"]])
    assert g.shortest_paths("D", "A") == (6, [["D", "G", "E", "A"]])
    assert g.shortest_paths("D", "C") == (3, [["D", "C"]])
    assert g.shortest_paths("D", "E") == (3, [["D", "G", "E"]])
    assert g.shortest_paths("D", "F") == (4, [["D", "C", "F"]])
    assert g.shortest_paths("D", "G") == (2, [["D", "G"]])
    assert g.shortest_paths("D", "D") == (0, [["D"]])
