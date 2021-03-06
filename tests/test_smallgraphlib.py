import math
import random

import pytest

from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.core import Traversal, InvalidGraphAttribute
from smallgraphlib.utilities import Multiset, segments_intersection

from smallgraphlib import (
    __version__,
    Graph,
    LabeledDirectedGraph,
    WeightedGraph,
    random_graph,
    complete_graph,
    complete_bipartite_graph,
    LabeledGraph,
    graph,
    perfect_binary_tree,
    WeightedDirectedGraph,
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


def test_complete_bipartite():
    g = complete_bipartite_graph(3, 4)
    assert g.order == 7
    assert g.degree == 12
    assert not g.is_directed
    assert g.is_simple
    assert g.is_complete_bipartite
    assert g.diameter == 2


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


def test_graph_constructor():
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


def test_dfs():
    g = Graph(range(1, 6), (3, 4), (3, 2), (2, 5), (2, 1))
    assert g.is_a_tree
    assert g.is_acyclic
    assert not g.is_directed
    assert g.is_connected
    assert g.diameter == 3
    assert tuple(g.depth_first_search(3, order=Traversal.PREORDER)) == (3, 4, 2, 5, 1)
    assert tuple(g.depth_first_search(3, order=Traversal.POSTORDER)) == (4, 5, 1, 2, 3)
    assert tuple(g.depth_first_search(3, order=Traversal.INORDER)) == (4, 3, 5, 2, 1)
    g = Graph(range(7), (0, 1), (0, 4), (1, 2), (1, 3), (4, 5), (4, 6))
    assert tuple(g.depth_first_search(order=Traversal.PREORDER)) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert tuple(g.depth_first_search(order=Traversal.POSTORDER)) == (
        2,
        3,
        1,
        5,
        6,
        4,
        0,
    )
    assert tuple(g.depth_first_search(order=Traversal.INORDER)) == (2, 1, 3, 0, 5, 4, 6)


def test_bfs():
    g = Graph(range(7), (0, 1), (0, 4), (1, 2), (1, 3), (4, 5), (4, 6))
    assert tuple(g.breadth_first_search()) == (0, 1, 4, 2, 3, 5, 6)


def test_binary_tree_and_dfs():
    g = perfect_binary_tree(4)
    assert g.nodes_set == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    assert g.edges_set == set(
        frozenset(s)
        for s in (
            {1, 2},
            {1, 3},
            {2, 4},
            {2, 5},
            {3, 6},
            {3, 7},
            {8, 4},
            {9, 4},
            {11, 5},
            {10, 5},
            {12, 6},
            {13, 6},
            {14, 7},
            {15, 7},
        )
    )
    assert list(g._iterative_depth_first_search()) == [
        1,
        2,
        4,
        8,
        9,
        5,
        10,
        11,
        3,
        6,
        12,
        13,
        7,
        14,
        15,
    ]
    assert g.is_a_tree
    assert g.is_acyclic
    g.add_edges((1, 15))
    assert g.order == 15
    assert g.degree == 15
    assert not g.is_a_tree
    assert not g.is_acyclic
    g.remove_edges((1, 15))
    g.add_edges((4, 14))
    assert g.order == 15
    assert g.degree == 15
    assert not g.is_a_tree
    assert not g.is_acyclic


def test_dfs_bfs():
    g = Graph("ABCDEFG", "AB", "BC", "BD", "AE", "EF", "EG")
    assert "".join(g.depth_first_search(start="A", order=Traversal.PREORDER)) == "ABCDEFG"
    assert "".join(g.depth_first_search(start="A", order=Traversal.POSTORDER)) == "CDBFGEA"
    assert "".join(g.depth_first_search(start="A", order=Traversal.INORDER)) == "CBDAFEG"


def test_weighted_graph():
    g = WeightedGraph([1, 2, 3, 4, 5], (1, 2, 10.0), (2, 3, 9.0), (2, 4, 7.0), (4, 5, 8.0))
    assert g.successors(1) == {2}
    assert g.weight(2, 1) == 10.0
    assert g.total_weight == 10.0 + 9.0 + 7.0 + 8.0


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


def test_intersection():
    A = (-4.191374663072777, -0.4986522911051212)
    B = (-0.41778975741239854, 1.495956873315364)
    C = (-3.4905660377358494, 2.035040431266846)
    D = (0.8760107816711589, -0.12129380053908356)
    expected_intersection = (-1.374693295677149, 0.9901650030897102)
    M = segments_intersection((A, B), (C, D))
    assert math.hypot(M[0] - expected_intersection[0], M[1] - expected_intersection[1]) < 10**-8
    A = (-2.0572283216093616, 1.544030635724147)
    assert segments_intersection((A, B), (C, D)) is None


def test_tikz_support():
    for seed in range(100):
        random.seed(seed)
        g = random_graph(4, 6, directed=True)
        g.as_tikz()


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
