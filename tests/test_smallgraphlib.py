from smallgraphlib import __version__, Graph


def test_version():
    assert __version__ == "0.1.0"


def test_properties():
    from smallgraphlib import Graph

    g = Graph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
    assert g.is_simple
    assert not g.is_complete
    assert g.is_directed
    assert g.adjacency_matrix == [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    assert g.degree
    assert g.order
    assert g.is_connected
    assert not g.is_strongly_connected


def test_modified_graphs():
    g = Graph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DA")
    assert g.degree == 7
    assert g.reversed_graph.degree == 7
    assert g.undirected_graph.degree == 7
    assert g.order == 5
    assert g.reversed_graph.order == 5
    assert g.undirected_graph.order == 5
    assert g.nodes == g.reversed_graph.nodes
    assert g.nodes == g.undirected_graph.nodes


def test_strongly_connected():
    g = Graph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DA")
    assert g.is_strongly_connected
    g = Graph("ABCDE", "AB", "BC", "CD", "DB", "BE", "ED", "DB")
    assert not g.is_strongly_connected


def test_connected():
    g = Graph("ABCDE", {"A", "B"}, {"B", "C"}, {"C", "D"}, {"C", "E"})
    assert g.is_connected
    g = Graph("ABCDE", {"A", "B"}, {"C", "D"}, {"C", "E"})
    assert not g.is_connected


def test_remove_nodes():
    g = Graph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
    g.remove_nodes("A")
    assert g.nodes == {"B", "C"}
    assert len(g.edges) == 1
    assert g.edges.pop() == ("B", "C")


def test_levels_and_kernel():
    g = Graph(
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
    assert g.levels == [{3, 5}, {4, 6}, {7}, {2}, {1}]
    assert g.kernel == {3, 5}


def test_cycle():
    g = Graph("ABCD", "AB", "BC", "CA", "AD")
    assert g.has_cycle
    g.remove_nodes("C")
    assert not g.has_cycle


def test_greedy_coloring():
    g = Graph("ABCDE", "AB", "AE", "AD", "BD", "BC", "BE", "CD", "DE", directed=False)
    coloring = g.greedy_coloring
    assert coloring["B"] == 0
    assert coloring["D"] == 1
    assert coloring["A"] == 2
    assert coloring["E"] == 3
    assert coloring["C"] == 2


def test_complete_eulerian():
    K = {}
    for n in range(2, 8):
        K[n] = Graph.complete_graph(n)
        assert K[n].order == n
        assert K[n].degree == n * (n - 1) / 2
        assert all(K[n].node_degree(node) == n - 1 for node in K[n].nodes)
        assert K[n].is_eulerian == (n % 2 == 1)
        assert K[n].is_semi_eulerian == (n == 2)


def test_graph_from_string():
    g = Graph.from_string("A:B,C B:C C")
    assert g.nodes == {"A", "B", "C"}
    assert g.degree == 3
    assert set(g.edges) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_random_graph():
    g = Graph.random_graph(4, 5, simple=True, directed=False)
    assert not g.is_directed
    assert g.order == 4
    assert g.degree == 5
    assert g.is_simple
    g = Graph.random_graph(4, 5, simple=True, directed=True)
    assert g.is_directed
    assert g.order == 4
    assert g.degree == 5
    assert g.is_simple


def test_remove_edges():
    g = Graph("ABCDE", "AB", "AB", "AC", "AD", "EA", "EC", directed=False)
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
    g = Graph("ABCDE", "AB", "BA", "AC", "AD", "EA", "EC", directed=False)
    assert not g.is_simple
    g.remove_edges("AB")
    assert g.is_simple
