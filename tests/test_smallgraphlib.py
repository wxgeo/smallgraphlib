import math
import random

import pytest

from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.core import InvalidGraphAttribute
from smallgraphlib.utilities import Multiset

from smallgraphlib import (
    __version__,
    Graph,
    random_graph,
    complete_graph,
    complete_bipartite_graph,
    graph,
)


def test_version():
    version = __version__.split(".")
    assert len(version) == 3


def test_properties():
    g = DirectedGraph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
    assert g.is_simple
    assert not g.is_complete
    assert g.is_directed
    assert g.adjacency_matrix == ((0, 1, 0), (1, 0, 1), (0, 0, 0))
    assert g.degree
    assert g.order
    assert g.is_connected
    assert not g.is_strongly_connected


def test_modified_graphs():
    g = DirectedGraph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DA")
    assert g.degree == 7
    assert g.reversed_graph.degree == 7
    assert g.undirected_graph.degree == 7
    assert g.order == 5
    assert g.reversed_graph.order == 5
    assert g.undirected_graph.order == 5
    assert list(g.nodes) == list(g.reversed_graph.nodes)
    assert list(g.nodes) == list(g.undirected_graph.nodes)


def test_strongly_connected():
    g = DirectedGraph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DA")
    assert g.is_strongly_connected
    g = DirectedGraph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DB")
    assert not g.is_strongly_connected


def test_connected():
    g = Graph("ABCDE", {"A", "B"}, {"B", "C"}, {"C", "D"}, {"C", "E"})
    assert g.is_connected
    g = Graph("ABCDE", {"A", "B"}, {"C", "D"}, {"C", "E"})
    assert not g.is_connected


def test_remove_nodes():
    g = DirectedGraph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
    g.remove_nodes("A")
    assert g.nodes_set == {"B", "C"}
    assert len(g.edges) == 1
    assert g.edges == (("B", "C"),)


def test_levels_and_kernel():
    g = DirectedGraph(
        (1, 2, 3, 4, 5, 6, 7),
        (1, 2),
        (1, 3),
        (2, 7),
        (2, 6),
        (2, 5),
        (4, 3),
        (4, 5),
        (6, 5),
        (7, 3),
        (7, 4),
        (7, 6),
    )
    assert g.order == 7
    assert g.degree == 11
    assert g.out_degree(1) == 2
    assert g.in_degree(1) == 0
    assert g.levels == ({3, 5}, {4, 6}, {7}, {2}, {1})
    assert g.kernel == {3, 5}
    g = DirectedGraph((1, 2, 3), (1, 2), (2, 3), (3, 1))
    with pytest.raises(InvalidGraphAttribute):
        _ = g.levels


def test_kernel():
    g = DirectedGraph(
        (1, 2, 3, 4, 5, 6, 7),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 5),
        (3, 6),
        (5, 3),
        (7, 4),
    )
    assert not g.has_cycle
    assert g.kernel == {4, 5, 6}


def test_cycle():
    g = DirectedGraph("ABCD", "AB", "BC", "CA", "AD")
    assert g.has_cycle
    g.remove_nodes("C")
    assert not g.has_cycle


def test_greedy_coloring():
    g = Graph("ABCDE", "AB", "AE", "AD", "BD", "BC", "BE", "CD", "DE")
    coloring = g.greedy_coloring
    assert coloring["B"] == 0
    assert coloring["D"] == 1
    assert coloring["A"] == 2
    assert coloring["E"] == 3
    assert coloring["C"] == 2


def test_complete_eulerian():
    K = {}
    for n in range(2, 8):
        K[n] = complete_graph(n)
        assert K[n].order == n
        assert K[n].degree == n * (n - 1) / 2
        assert all(K[n].node_degree(node) == n - 1 for node in K[n].nodes)
        assert K[n].is_eulerian == (n % 2 == 1)
        assert K[n].is_semi_eulerian == (n == 2)


def test_graph_from_string():
    g = DirectedGraph.from_string("A:B,C B:C C")
    assert g.nodes_set == {"A", "B", "C"}
    assert g.degree == 3
    assert g.edges_set == {("A", "B"), ("A", "C"), ("B", "C")}


def test_simple_random_graph():
    for seed in range(10):
        random.seed(seed)
        g = random_graph(4, 5, directed=False, simple=True)
        assert not g.is_directed
        assert g.order == 4
        assert g.degree == 5
        assert g.is_simple
        g = random_graph(4, 5, directed=True, simple=True)
        assert g.is_directed
        assert g.order == 4
        assert g.degree == 5
        assert g.is_simple


def test_random_graph():
    max_multiple_edges = 5
    max_multiple_loops = 2
    for directed in (False, True):
        for seed in range(10, 20):
            random.seed(seed)
            g = random_graph(
                4,
                20,
                directed=directed,
                tikz_export_supported=False,
                max_multiple_edges=max_multiple_edges,
                max_multiple_loops=max_multiple_loops,
            )
            M = g.adjacency_matrix
            assert max(elt for line in M for elt in line) <= max_multiple_edges, g
            # loops count twice in undirected graphs !
            assert max(M[i][i] for i, _ in enumerate(M)) <= (
                max_multiple_loops if directed else 2 * max_multiple_loops
            ), g


def test_random_graph2():
    max_multiple_edges = 5
    max_multiple_loops = 2
    for directed in (False, True):
        for seed in range(10, 20):
            random.seed(seed)
            g = random_graph(
                4,
                20,
                directed=directed,
                tikz_export_supported=False,
                max_multiple_edges=max_multiple_edges,
                max_multiple_loops=max_multiple_loops,
            )
            M = g.adjacency_matrix
            assert max(elt for line in M for elt in line) <= max_multiple_edges, g
            # loops count twice in undirected graphs !
            assert max(M[i][i] for i, _ in enumerate(M)) <= (
                max_multiple_loops if directed else 2 * max_multiple_loops
            ), g


def test_random_graph_node_names():
    g = random_graph(4, 6, nodes_names="ABCD")
    assert g.nodes_set == set("ABCD")


def test_remove_edges():
    g = Graph("ABCDE", "AB", "AB", "AC", "AD", "EA", "EC")
    assert g.degree == 6
    g.remove_edges("AB")
    assert g.degree == 5
    assert g.edges_set == {
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"A", "D"}),
        frozenset({"E", "A"}),
        frozenset({"E", "C"}),
    }


def test_simple():
    g = Graph("ABCDE", "AB", "BA", "AC", "AD", "EA", "EC")
    assert not g.is_simple
    g.remove_edges("AB")
    assert g.is_simple


def test_complete_bipartite():
    g = complete_bipartite_graph(3, 4)
    assert g.order == 7
    assert g.degree == 12
    assert not g.is_directed
    assert g.is_simple
    assert g.is_complete_bipartite
    assert g.diameter == 2


def test_graph_string():
    g = graph("A:B,C B:C C")
    assert g.degree == 3
    assert not g.is_directed
    assert g.is_simple
    g = graph(AB=5, BC=7, AC=8, directed=True)
    assert g.degree == 3
    assert g.weight("A", "B") == 5
    assert g.weight("B", "C") == 7
    assert g.weight("A", "C") == 8
    assert g.is_directed
    g = graph("A:B=5,C=8 B:C=inf C", directed=True)
    assert g.is_directed
    assert g.diameter == math.inf
    g = graph("A:B='some text with space',C=text_without_space B:C=2.5 C D")
    assert not g.is_directed
    assert g.labels("A", "B") == ["some text with space"]


def test_graph_from_dict():
    g = DirectedGraph.from_dict({1: (2, 2, 3), 4: []})
    assert sorted(g.nodes) == [1, 2, 3, 4]
    assert sorted(g.edges) == [(1, 2), (1, 2), (1, 3)]


def test_single_node_renaming():
    g = DirectedGraph("ABCDE", "AB", "AC", "BC", "CD", "DE", "EA", "ED")
    g.rename_node("A", "F")
    assert g.nodes_set == set("FBCDE")
    assert g.edges_set == {
        ("F", "B"),
        ("F", "C"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("E", "D"),
    }
    assert g.successors("F") == {"B", "C"}
    assert g.predecessors("F") == {"E"}
    assert g.predecessors("C") == {"F", "B"}
    assert g.successors("E") == {"F", "D"}


def test_simultaneous_nodes_renaming():
    g = DirectedGraph("ABCDE", "AB", "AC", "BC", "CD", "DE", "EA", "ED")
    with pytest.raises(ValueError):
        g.rename_nodes({"A": "E", "B": "E"})
    assert g.nodes_set == set("ABCDE")
    with pytest.raises(ValueError):
        g.rename_nodes({"A": "B", "B": "C"})
    assert g.nodes_set == set("ABCDE")
    with pytest.raises(ValueError):
        g.rename_nodes({"A": "G", "F": "C"})

    g.rename_nodes({"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"})
    assert g.nodes_set == set("abcde")

    g = DirectedGraph("ABCDE", "AB", "AC", "BC", "CD", "DE", "EA", "ED")
    g_copy = g.copy()
    translate = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}
    g.rename_nodes(translate)
    assert g.nodes_set == set("ABCDE")
    assert g.edges_set == {
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
        ("D", "E"),
        ("E", "A"),
        ("A", "B"),
        ("A", "E"),
    }
    assert g != g_copy
    reverse_translate = {v: k for k, v in translate.items()}
    g.rename_nodes(reverse_translate)
    assert g == g_copy


def test_Multiset():
    with pytest.raises(ValueError) as _:
        # Negative counts are not allowed.
        Multiset({"a": 2, "b": 4, "c": 4, "d": 0, "e": -1})
    s = Multiset({"a": 2, "b": 4, "c": 4, "d": 0})
    # Automatically remove key 'd', since its count is zero.
    assert set(s) == {"a", "b", "c"}
    s["a"] -= 1
    assert set(s) == {"a", "b", "c"}
    s["a"] -= 1
    assert set(s) == {"b", "c"}
    with pytest.raises(ValueError) as _:
        s["a"] -= 1


def test_isomorphic_basic_case():
    g1 = Graph((1, 2, 3), (1, 2))
    g2 = Graph((4, 5, 6), (5, 6))
    assert g1.is_isomorphic_to(g2)


def test_non_isomorphic_with_same_degrees():
    k33 = complete_bipartite_graph(3, 3)
    isomorphic_to_k33 = graph("s1:s2,s3,s6 s2:s4,s5 s3:s4,s5 s4:s6 s5:s6 s6")
    other_graph_with_same_degrees = graph("t1:t2,t5,t4 t2:t6,t3 t3:t6,t4 t4:t5 t5:t6 t6")
    assert k33.order == isomorphic_to_k33.order == other_graph_with_same_degrees.order
    assert k33.degree == isomorphic_to_k33.degree == other_graph_with_same_degrees.degree
    assert not isomorphic_to_k33.is_isomorphic_to(other_graph_with_same_degrees)
    assert isomorphic_to_k33.is_isomorphic_to(k33)
    assert not other_graph_with_same_degrees.is_isomorphic_to(k33)


def test_graph_from_matrix():
    for seed in range(10):
        random.seed(seed)
        g = random_graph(5, 8, directed=False)
        assert Graph.from_matrix(g.adjacency_matrix) == g

        g = random_graph(5, 8, directed=True)
        assert DirectedGraph.from_matrix(g.adjacency_matrix) == g


def test_transitivity():
    M = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]]
    g = DirectedGraph.from_matrix(M)
    assert not g.is_transitive
    assert g.transitive_closure_matrix == (
        (0, 1, 1, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 0),
        (0, 0, 1, 0),
    )
    assert DirectedGraph.from_matrix(g.transitive_closure_matrix).is_transitive
    M = [[0, 1], [1, 0]]
    g = DirectedGraph.from_matrix(M)
    assert not g.is_transitive
    assert g.transitive_closure_matrix == ((1, 1), (1, 1))
    assert DirectedGraph("ABCD", "AB", "BA", "AC", "AD", "AA", "BB", "BC", "BD").is_transitive
