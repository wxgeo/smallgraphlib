import math
from abc import ABC, abstractmethod
from math import inf
from numbers import Real
from typing import Generic, Iterable, Any, Sequence, Type, TypeVar

from smallgraphlib.core import InvalidGraphAttribute

from smallgraphlib.labeled_graphs import LabeledGraph, LabeledDirectedGraph
from smallgraphlib.custom_types import Node, Label, WeightedEdge, Edge
from smallgraphlib.labeled_graphs import AbstractLabeledGraph
from smallgraphlib.utilities import cached_property

_AbstractWeightedGraph = TypeVar("_AbstractWeightedGraph", bound="AbstractWeightedGraph")


class AbstractNumericGraph(AbstractLabeledGraph, ABC, Generic[Node, Label]):
    """Abstract class for all graphs labeled with positive numeric values, don't use it directly."""

    def __init__(
        self,
        nodes: Iterable[Node],
        *weighted_edges: WeightedEdge,
        sort_nodes: bool = True,
    ):
        super().__init__(nodes, *weighted_edges, sort_nodes=sort_nodes)
        for edge, labels in self._labels.items():
            for label in labels:
                if not self._is_positive_number(label):
                    raise ValueError(f"Edge {edge} weight must be a positive real number, not {label!r}.")

    @staticmethod
    def _is_positive_number(value: Any) -> bool:
        """Test if a value is a positive real number (including positive infinity)."""
        try:
            if isinstance(value, Real):
                return float(value) >= 0
            else:
                # We must support sympy.oo, but isinstance(sympy.oo, Real) return False.
                return float(value) == math.inf and value != "inf"  # do not accept string "inf" !
        except (TypeError, ValueError):
            return False

    def _edge_value(self, node1: Node, node2: Node, *, aggregator=min, default: float = inf) -> float:
        """
        Return the weight of the edge joining node1 and node2.

        If edge does not exist, return default value.

        Args:
            node1: Node
            node2: Node
            aggregator: function used to aggregate values if there are several edges
            between `node1` and `node2` (default is `min`)
            default: value returned if `node1` and `node2` are not adjacents (default is `inf`).
        Return:
            float | None
        """
        values = self._labels.get(self._edge(node1, node2), [default])
        return aggregator(values)


class AbstractWeightedGraph(AbstractNumericGraph, ABC, Generic[Node, Label]):
    """Abstract class for all weighted graphs, don't use it directly."""

    def __init__(
        self,
        nodes: Iterable[Node],
        *weighted_edges: WeightedEdge,
        sort_nodes: bool = True,
    ):
        super().__init__(nodes, *weighted_edges, sort_nodes=sort_nodes)
        for edge, weights in self.weights.items():
            for weight in weights:
                if not self._is_positive_number(weight):
                    raise ValueError(f"Edge {edge} weight must be a positive real number, not {weight!r}.")

    def weight(self, node1: Node, node2: Node) -> float:
        return 0 if node1 == node2 else self._edge_value(node1, node2, aggregator=min, default=inf)

    @property
    def weights(self) -> dict[Edge, list[float]]:
        return self._labels

    @cached_property
    def total_weight(self) -> float:
        """Return the sum of all edges weights."""
        return sum(sum(values) for values in self.weights.values())

    @staticmethod
    @abstractmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> list[tuple[int, int, float]]:
        ...

    @classmethod
    def from_matrix(
        cls: Type[_AbstractWeightedGraph],
        matrix: Iterable[Iterable[float]],
        nodes_names: Iterable[Node] = None,
    ) -> _AbstractWeightedGraph:
        """Construct the graph corresponding to the given adjacency matrix.

        Matrix must be a matrix of positive real numbers
        (`float` or any numeric type inheriting from `numbers.Real`).
        """
        # Convert iterable to matrix.
        M = cls._matrix_as_tuple_of_tuples(matrix)
        # Test if M is correct
        n = len(M)
        for line in M:
            for val in line:
                if not cls._is_positive_number(val):
                    raise ValueError(f"All matrix values must be positive real numbers, but {val!r} is not.")

        edges = cls._get_edges_from_weights_matrix(M)

        g = cls(range(1, n + 1), *edges)
        if nodes_names:
            g.rename_nodes(dict(enumerate(list(nodes_names)[: len(g.nodes)], start=1)))
        return g

    def get_path_weight(self, path: list[Node]):
        """Return the weight of the path, which is the sum of those of its edges."""
        return sum(self.weight(node1, node2) for node1, node2 in zip(path[:-1], path[1:]))


class WeightedGraph(AbstractWeightedGraph, LabeledGraph, Generic[Node]):
    """A weighted undirected graph, i.e. an undirected graph where all edges have a positive weight."""

    @staticmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> list[tuple[int, int, float]]:
        edges = []
        for i in range(len(matrix)):
            for j in range(i + 1):  # we must only deal with i <= j, since it is an undirected graph.
                weight = matrix[i][j]
                if i == j:
                    if weight != 0:
                        raise ValueError(
                            "The diagonal coefficients of the weights matrix must be nul, "
                            f"but matrix[{i}][{i}]={weight}."
                        )
                    # Don't append loops.
                else:
                    if matrix[i][j] != matrix[j][i]:
                        raise ValueError(
                            "The adjacency matrix of an undirected graph must be symmetric, "
                            f"but matrix[{i}][{j}]={matrix[i][j]} != matrix[{j}][{i}]={matrix[j][i]}"
                        )
                    # If weight is infinite, nodes are not adjacent.
                    if not math.isinf(weight):
                        edges.append((i + 1, j + 1, weight))
        return edges

    def minimum_spanning_tree(self) -> "WeightedGraph":
        """Use Prim's algorithm to return a minimum weight spanning tree.

        A spanning tree of a graph G is a subgraph of G who is a tree and contains all the nodes of G.

        If the graph is not connected, raise an `InvalidGraphAttribute`.
        """
        # Nodes and edges of the spanning tree.
        last_connected_node = self.nodes[0]
        connected_nodes: list[Node] = [last_connected_node]
        weighted_edges: list[WeightedEdge] = []

        # Nodes which may be connected, with the current cost of connection, i.e. the minimal edge's weight
        # enabling to connect it to the spanning tree.
        cheapest_cost: dict[Node, float] = {}
        cheapest_edge: dict[Node, Node] = {}
        unreached_nodes = set(self.nodes) - set(connected_nodes)

        while True:
            # Update frontier
            for successor in self.successors(last_connected_node):
                if successor not in connected_nodes:
                    cost = self.weight(last_connected_node, successor)
                    if cost < cheapest_cost.get(successor, inf):
                        # This a cheaper way to connect the node, so update.
                        cheapest_cost[successor] = cost
                        cheapest_edge[successor] = last_connected_node

            if not cheapest_cost:
                break

            # connect one more node at minimal cost
            last_connected_node = min(cheapest_cost, key=cheapest_cost.get)  # type: ignore
            connected_nodes.append(last_connected_node)
            unreached_nodes.remove(last_connected_node)
            weighted_edges.append(
                (
                    cheapest_edge[last_connected_node],
                    last_connected_node,
                    cheapest_cost[last_connected_node],
                )
            )
            cheapest_cost.pop(last_connected_node)
            cheapest_edge.pop(last_connected_node)

        if unreached_nodes:
            raise InvalidGraphAttribute("This graph has no spanning tree since it is not connected.")

        return WeightedGraph(connected_nodes, *weighted_edges)


class WeightedDirectedGraph(AbstractWeightedGraph, LabeledDirectedGraph, Generic[Node]):
    """A directed graph with weights (i.e. positive floats) associated to its edges."""

    @staticmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> list[tuple[int, int, float]]:
        edges = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                weight = matrix[i][j]
                if i == j:
                    if weight != 0:
                        raise ValueError(
                            "The diagonal coefficients of the weights matrix must be nul, "
                            f"but matrix[{i}][{i}]={weight}."
                        )

                elif not math.isinf(weight):
                    # If weight is infinite, nodes are not adjacent.
                    edges.append((i + 1, j + 1, weight))
        return edges
