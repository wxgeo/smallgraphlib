import math
import random

from smallgraphlib import (
    random_graph,
    complete_graph,
    complete_bipartite_graph,
    graph,
)


def test_complete_eulerian():
    K = {}
    for n in range(2, 8):
        K[n] = complete_graph(n)
        assert K[n].order == n
        assert K[n].degree == n * (n - 1) / 2
        assert all(K[n].node_degree(node) == n - 1 for node in K[n].nodes)
        assert K[n].is_eulerian == (n % 2 == 1)
        assert K[n].is_semi_eulerian == (n == 2)


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
    g = graph("A:B=5,C=8 B:C=inf C")
    assert g.nodes == ("A", "B", "C")
    assert g.weight("B", "C") == float("inf")


def test_non_isomorphic_with_same_degrees():
    k33 = complete_bipartite_graph(3, 3)
    isomorphic_to_k33 = graph("s1:s2,s3,s6 s2:s4,s5 s3:s4,s5 s4:s6 s5:s6 s6")
    other_graph_with_same_degrees = graph("t1:t2,t5,t4 t2:t6,t3 t3:t6,t4 t4:t5 t5:t6 t6")
    assert k33.order == isomorphic_to_k33.order == other_graph_with_same_degrees.order
    assert k33.degree == isomorphic_to_k33.degree == other_graph_with_same_degrees.degree
    assert not isomorphic_to_k33.is_isomorphic_to(other_graph_with_same_degrees)
    assert isomorphic_to_k33.is_isomorphic_to(k33)
    assert not other_graph_with_same_degrees.is_isomorphic_to(k33)
