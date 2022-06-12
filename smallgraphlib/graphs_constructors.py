import random
from typing import List, Tuple

from smallgraphlib import (
    Graph,
    WeightedDirectedGraph,
    LabeledDirectedGraph,
    LabeledGraph,
    WeightedGraph,
)
from smallgraphlib.basic_graphs import (
    DirectedGraph,
)
from smallgraphlib.core import (
    _TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT,
    _TIKZ_EXPORT_MAX_MULTIPLE_LOOPS_SUPPORT,
    AbstractGraph,
    Edge,
)
from smallgraphlib.utilities import Multiset


def graph(nodes=None, *edges, directed=False, **labeled_edges):
    """Factory function to create graphs with various syntaxes.

    >>> graph("A:B,C B:C C")
    >>> graph(AB=5, BC=7, AC=8, directed=True)
    >>> graph("A:B=5,C=8 B:C=inf", directed=True)
    >>> graph("A:B='some text with space',C=text_without_space B:C=2.5 D")

    This is intended mainly for quick interactive use (or one-time-use scripts),
    since its interface is not very clean and may change frequently.

    You should use the constructors provided by the various graph classes
    if you need a stable API.
    """
    if isinstance(nodes, str) and not edges:
        classes = (DirectedGraph, WeightedDirectedGraph, LabeledDirectedGraph)
        if not directed:
            classes = (Graph, WeightedGraph, LabeledGraph)
        error = None
        for cls in classes:
            try:
                return cls.from_string(nodes)
            except ValueError as e:
                error = e
        raise ValueError(f"Invalid argument value: {nodes!r}.") from error
    elif isinstance(nodes, dict) or nodes is None and labeled_edges:
        if isinstance(nodes, dict):
            labeled_edges.update(nodes)
        if all(isinstance(val, (float, int)) for val in labeled_edges.values()):
            cls = WeightedDirectedGraph if directed else WeightedGraph
        else:
            cls = LabeledDirectedGraph if directed else LabeledGraph
        return cls.from_dict(labeled_edges)
    cls = DirectedGraph if directed else Graph
    return cls(nodes, *edges)


def complete_graph(n: int):
    """Return complete graph K(n)."""
    nodes = list(range(1, n + 1))
    edges = ((i, j) for i in nodes for j in nodes if i < j)
    return Graph(nodes, *edges)


def complete_bipartite_graph(n: int, m: int):
    """Return complete bipartite graph K(n, m)."""
    return Graph(
        range(1, m + n + 1),
        *((i, j) for i in range(1, n + 1) for j in range(n + 1, n + 1 + m)),
    )


def perfect_binary_tree(height: int):
    """Return a perfect binary tree of given height.

    In a perfect binary tree, each node has two children, except in last level.
    """
    nodes: List[int] = []
    edges: List[Edge] = []
    i = 1
    for level in range(height):
        nodes.extend(range(1 << level, 1 << level + 1))
    for level in range(height - 1):
        for i in range(1 << level, 1 << level + 1):
            edges.append((i, i << 1))
            edges.append((i, (i << 1) + 1))

    return Graph(nodes, *edges)


def random_graph(
    order: int,
    degree: int,
    *,
    directed=True,
    simple=False,
    tikz_export_supported=True,
    max_multiple_edges: int = None,
    max_multiple_loops: int = None,
    shuffle_nodes=False,
) -> AbstractGraph:
    """Create a random graph satisfying given constraints.

    Raise a ValueError if contraints can't be satisfied.

    If `simple` is True, ignore remaining keyword arguments.
    Similarly, if `tikz_export_supported` is True, all remaining keyword arguments will be ignored.
    """
    # Test for feasibility: is it possible to satisfy all constraints ?
    if simple:
        max_degree_for_simple_graph = order * (order - 1)
        if not directed:
            max_degree_for_simple_graph //= 2
        if degree > max_degree_for_simple_graph:
            graph_type = "directed" if directed else "undirected"
            raise ValueError(
                f"Degree is at most {max_degree_for_simple_graph} "
                f"for a simple {graph_type} graph of order {order}."
            )
    if simple:
        if max_multiple_edges is not None or max_multiple_loops is not None:
            raise ValueError(
                "Conflicting arguments: `simple` must be set to False, "
                "else `max_multiple_edges` and `max_multiple_loops` will be ignored."
            )
        max_multiple_edges = 1
        max_multiple_loops = 0
    elif tikz_export_supported:
        if max_multiple_edges is not None or max_multiple_loops is not None:
            raise ValueError(
                "Conflicting arguments: `tikz_export_supported` must be set to False, "
                "else `max_multiple_edges` and `max_multiple_loops` will be ignored."
            )
        max_multiple_edges = _TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT
        max_multiple_loops = _TIKZ_EXPORT_MAX_MULTIPLE_LOOPS_SUPPORT
    else:
        if max_multiple_edges is None:
            max_multiple_edges = degree
        if max_multiple_loops is None:
            max_multiple_loops = degree

    if directed:
        max_degree = order * (max_multiple_loops + (order - 1) * max_multiple_edges)
    else:
        max_degree = order * max_multiple_loops + (order * (order - 1) * max_multiple_edges) // 2

    if degree > max_degree:
        raise ValueError(f"Degree must not exceed {max_degree} with given contraints.")

    nodes: List[int] = list(range(1, order + 1))
    edges: List[Tuple[int, int]] = []

    # Keep track of remaining edges possibilities, for each (start, end) nodes couple.
    counters = {
        start: counter
        for start in nodes
        if (
            counter := Multiset(
                {
                    end: (max_multiple_loops if start == end else max_multiple_edges)
                    for end in nodes
                    if directed or start <= end  # for undirected graph, only keep start <= end
                }
            )
        ).total()
        > 0
    }
    # Keep only nodes which accept edges (all except the last one
    # for undirected simple graph, since start < end).
    starts = list(counters)

    while len(edges) < degree:
        start = random.choice(starts)
        counter = counters[start]
        assert counter.total() > 0, counter
        end = random.choice(list(counter.elements()))
        counter[end] -= 1
        if counter.total() == 0:
            # Don't select anymore this node as start point: we can't add any other edge to it.
            starts.remove(start)
        edges.append((start, end))  # Loop will always end. :)
    g = graph(nodes, *edges, directed=directed)
    if shuffle_nodes:
        g.shuffle_nodes()
    return g
