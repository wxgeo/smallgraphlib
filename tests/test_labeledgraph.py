from smallgraphlib import LabeledDirectedGraph, LabeledGraph


def test_LabeledDirectedGraph_from_string():
    g = LabeledDirectedGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
    assert g.nodes == ("A", "B", "C", "D")
    assert g.edges_set == {("A", "B"), ("A", "C"), ("B", "C"), ("D", "C")}
    assert g.labels("A", "B") == ["label"]
    assert g.labels("A", "C") == ["other label"]
    assert g.labels("B", "C") == ["5"]
    assert g.labels("D", "C") == [""]


def test_LabeledGraph_from_string():
    def f(*nodes):
        return frozenset(nodes)

    g = LabeledGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
    assert g.nodes == ("A", "B", "C", "D")
    assert g.edges_set == {f("A", "B"), f("A", "C"), f("B", "C"), f("D", "C")}
    assert g.labels("A", "B") == ["label"]
    assert g.labels("A", "C") == ["other label"]
    assert g.labels("B", "C") == ["5"]
    assert g.labels("D", "C") == [""]


def test_dict_and_copy():
    g = LabeledDirectedGraph.from_dict({"AB": 5, "BC": "hello"})
    assert g.copy() == g
    assert g.from_dict(g.as_dict()) == g
    h = LabeledGraph.from_dict({"AB": 5, "BC": "hi"})
    assert h.from_dict(h.as_dict()) == h
    assert h.copy() == h
    # Directed and undirected graphs should never be equal.
    assert g != h
