import random

from smallgraphlib.graph import DirectedGraph

from smallgraphlib import (
    __version__,
    Graph,
    WeightedGraph,
    random_graph,
    complete_graph,
    complete_bipartite_graph,
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
    assert set(g.nodes) == {"B", "C"}
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
    assert set(g.nodes) == {"A", "B", "C"}
    assert g.degree == 3
    assert set(g.edges) == {("A", "B"), ("A", "C"), ("B", "C")}


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


def test_remove_edges():
    g = Graph("ABCDE", "AB", "AB", "AC", "AD", "EA", "EC")
    assert g.degree == 6
    g.remove_edges("AB")
    assert g.degree == 5
    assert set(g.edges) == {
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


def test_shortest_paths():
    g = WeightedGraph.from_dict(
        AB=1, BG=9, FC=1, FD=5, AG=9, BC=8, AF=5, BF=7, BD=12, CD=3, FE=2, ED=7, GE=1, GD=2, AE=3
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


def test_complete_bipartite():
    g = complete_bipartite_graph(3, 4)
    assert g.order == 7
    assert g.degree == 12
    assert not g.is_directed
    assert g.is_simple
    assert g.is_complete_bipartite
    assert g.diameter == 2
