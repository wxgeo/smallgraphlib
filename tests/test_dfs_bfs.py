from smallgraphlib import Graph, Traversal, perfect_binary_tree


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
