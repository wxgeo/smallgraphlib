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
