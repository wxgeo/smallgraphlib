from smallgraphlib import __version__


def test_version():
    version = __version__.split(".")
    assert len(version) == 3


def test_imports():
    import smallgraphlib

    required = {
        "__version__",
        "Graph",
        "DirectedGraph",
        "WeightedGraph",
        "WeightedDirectedGraph",
        "LabeledGraph",
        "LabeledDirectedGraph",
        "random_graph",
        "complete_graph",
        "graph",
        "complete_bipartite_graph",
        "perfect_binary_tree",
        "Traversal",
        "InvalidGraphAttribute",
        "Acceptor",
        "Transducer",
        "FlowNetwork",
    }
    assert required.issubset(set(vars(smallgraphlib)))
