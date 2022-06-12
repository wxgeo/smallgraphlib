from importlib import metadata

from smallgraphlib.basic_graphs import Graph, DirectedGraph
from smallgraphlib.labeled_graphs import (
    WeightedGraph,
    WeightedDirectedGraph,
    LabeledGraph,
    LabeledDirectedGraph,
)
from smallgraphlib.graphs_constructors import (
    complete_graph,
    complete_bipartite_graph,
    graph,
    random_graph,
    perfect_binary_tree,
)
from smallgraphlib.core import Traversal

__version__ = metadata.version(__package__)

__all__ = [
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
]
