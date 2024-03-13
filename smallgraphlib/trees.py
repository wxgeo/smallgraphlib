from numbers import Real
from typing import Iterable, Any, Set, Iterator, Generic

from smallgraphlib import Graph, WeightedGraph, LabeledGraph
from smallgraphlib.core import (
    AbstractGraph,
    InvalidGraphAttribute,
    Traversal,
)
from smallgraphlib.custom_types import Node, Edge


class Tree(Generic[Node]):
    """A rooted tree."""

    def __init__(
        self,
        nodes: Iterable[Node],
        root: Node = None,
        *edges: Edge,
        nodes_labels: dict[Node, Any] = None,
        edges_labels: dict[Edge, Any] = None,
    ):
        if root is None:
            root = next(iter(nodes))
        self._root = root
        if edges_labels is None:
            self._graph: Graph[Node] = Graph(nodes, *edges)
        else:
            edges_labels = edges_labels.copy()
            labeled_edges = []
            for edge in edges:
                labeled_edges.append((*edge, edges_labels.pop(edge, None)))
            if edges_labels:  # should be empty now
                raise ValueError("Unknown edges: " + ", ".join(str(edge) for edge in edges_labels))
            if all(isinstance(value, Real) for value in edges_labels.values()):
                self._graph = WeightedGraph(nodes, *labeled_edges)
            else:
                self._graph = LabeledGraph(nodes, *labeled_edges)

    @property
    def root(self) -> Node:
        return self._root

    def as_graph(self) -> AbstractGraph[Node]:
        return self._graph.copy()

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self._graph.nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        return self._graph.edges

    def children(self, node: Node) -> Set[Node]:
        return self._graph.successors(node)

    def parent(self, node: Node) -> Node:
        if node == self.root:
            raise InvalidGraphAttribute(f"Root node {self.root} has no parent.")
        predecessors = self._graph.predecessors(node)
        assert len(predecessors) == 1, predecessors
        return next(iter(predecessors))

    def depth_first_search(self, *, order: Traversal = Traversal.PREORDER) -> Iterator[Node]:
        """Recursive implementation of DFS (Depth First Search)."""
        return self._graph.depth_first_search(start=self.root, order=order)

    def breadth_first_search(self) -> Iterator[Node]:
        """Implementation of BFS (Breadth First Search)."""
        return self._graph.breadth_first_search(start=self.root)

    def as_tikz(self):
        raise NotImplementedError


class BinaryTree(Tree):
    ...
