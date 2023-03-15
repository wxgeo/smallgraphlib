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
