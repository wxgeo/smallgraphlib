import ast
import math
import re
from abc import ABC, abstractmethod
from itertools import chain
from math import inf
from numbers import Real
from typing import Iterable, Tuple, Dict, List, TypeVar, Generic, Any, Sequence, Type

from smallgraphlib.basic_graphs import Graph, DirectedGraph
from smallgraphlib.core import AbstractGraph, InvalidGraphAttribute
from smallgraphlib.custom_types import Node, Edge
from smallgraphlib.utilities import cached_property

_AbstractLabeledGraph = TypeVar("_AbstractLabeledGraph", bound="AbstractLabeledGraph")
_AbstractWeightedGraph = TypeVar("_AbstractWeightedGraph", bound="AbstractWeightedGraph")
Label = TypeVar("Label")
LabeledEdge = Tuple[Node, Node, Label]
WeightedEdge = Tuple[Node, Node, float]


class AbstractLabeledGraph(AbstractGraph, ABC, Generic[Label]):
    """Abstract class for all labeled graphs, don't use it directly."""

    def __init__(
        self,
        nodes: Iterable[Node],
        *labeled_edges: LabeledEdge,
        sort_nodes: bool = True,
    ):
        edges: List[Edge] = []
        self._labels: Dict[Edge, List[Label]] = {}
        for *edge, label in labeled_edges:
            edges.append(edge)  # type: ignore
            self._labels.setdefault(self._edge(*edge), []).append(label)

        super().__init__(nodes, *edges, sort_nodes=sort_nodes)

    def __eq__(self, other: Any):
        return super().__eq__(other) and all(
            sorted(self.labels(*edge)) == sorted(other.labels(*edge)) for edge in self.edges
        )

    def __repr__(self):
        labeled_edges = (repr(labeled_edge) for labeled_edge in self.labeled_edges)
        return f"{self.__class__.__name__}({tuple(self.nodes)!r}, {', '.join(labeled_edges)})"

    @cached_property
    def labeled_edges(self) -> tuple[LabeledEdge, ...]:
        return tuple(  # type: ignore
            sorted(
                (
                    *edge,  # type: ignore
                    label,
                )
                for edge, labels in self._labels.items()
                for label in labels
            )
        )

    @classmethod
    def from_dict(
        cls: Type[_AbstractLabeledGraph], edge_label_dict: dict = None, /, **edge_label
    ) -> _AbstractLabeledGraph:
        """Construct a directed graph using a {edge_name: label} dictionnary (or keywords).

        All edges' names must be two letters strings (like "AB"), each letter representing a node.
        Nodes' names are automatically deduced from edges' names.

        >>> g1 = LabeledGraph.from_dict(AB=1, AC=3, BC=4)
        >>> g2 = LabeledGraph.from_dict({"AB": 1, "AC": 3, "BC": 4})
        >>> g1 == g2
        True
        >>> g1.nodes
        ('A', 'B', 'C')
        """
        if edge_label_dict is None:
            edge_label_dict = {}
        edge_label_dict.update(edge_label)
        nodes = set(chain(*(edge for edge in edge_label_dict)))
        return cls(nodes, *((*edge, label) for edge, label in edge_label_dict.items()))  # type: ignore

    @classmethod
    def from_string(cls: Type[_AbstractLabeledGraph], string: str) -> _AbstractLabeledGraph:
        """LabeledGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
        will generate a graph of 4 nodes, A, B, C and D, with edges A->B, A->C, B->C and C->D
        and respective labels 'label', 'other label', 5 and `None`.

        Note that labels containing a space must be surrounded by single or double quotes.
        Labels containing only digits will be converted to numbers (integers or floats),
        except if surrounded by quotes.

        If no label is given, `None` is stored by default.
        """
        # Convert spaces inside labels to null characters, to make splitting easier.
        string = re.sub("""'[^']*'|"[^"]*""", (lambda m: m.group().replace(" ", "\x00")), string)
        nodes: List[str] = []
        edges: List[Tuple[str, str, Any]] = []
        label: Any
        for substring in string.split():
            node, *remaining = substring.split(":", 1)
            nodes.append(node.strip())
            if remaining:
                for successor_and_label in remaining[0].split(","):
                    successor, *after_successor = successor_and_label.split("=", 1)
                    if after_successor:
                        # Convert back null characters inside labels to spaces.
                        label = after_successor[0].strip().replace("\x00", " ")
                        try:
                            label = ast.literal_eval(label)
                        except (ValueError, SyntaxError):
                            if label == "inf":
                                label = math.inf
                            elif label == "-inf":
                                label = -math.inf
                    else:
                        label = None
                    edges.append((node, successor.strip(), label))
        return cls(nodes, *edges)  # type: ignore

    def labels(self, node1: Node, node2: Node) -> List[str]:
        labels = self._labels.get(self._edge(node1, node2), [])
        assert len(labels) == self.count_edges(node1, node2, count_undirected_loops_twice=False)
        return [str(label if label is not None else "") for label in labels]

    def rename_nodes(self, node_names: Dict[Node, Node]) -> None:
        super().rename_nodes(node_names)
        # Rename nodes in self._labels dict.
        old_labels = self._labels.copy()
        self._labels.clear()
        for edge, label in old_labels.items():
            new_edge = self._edge(*(node_names[node] for node in edge))
            self._labels[new_edge] = label

        # # Rename nodes in self._labels dict.
        # edges_and_labels = [(list(edge), label) for edge, label in self._labels.items()]
        # for edge, label in edges_and_labels:
        #     for i, node in enumerate(edge):
        #         edge[i] = node_names[node]
        # self._labels.clear()
        # self._labels.update(*((self._edge(*edge), label) for edge, label in edges_and_labels))


class LabeledGraph(AbstractLabeledGraph, Graph):
    """A labeled undirected graph."""


class LabeledDirectedGraph(AbstractLabeledGraph, DirectedGraph):
    pass


class AbstractWeightedGraph(AbstractLabeledGraph, ABC):
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
                if not self._is_weight(weight):
                    raise ValueError(f"Edge {edge} weight must be a positive real number, not {weight!r}.")

    @staticmethod
    def _is_weight(value: Any) -> bool:
        """Test if a value is a positive real number (including positive infinity)."""
        try:
            if isinstance(value, Real):
                return float(value) >= 0
            else:
                # We must support sympy.oo, but isinstance(sympy.oo, Real) return False.
                return float(value) == math.inf and value != "inf"  # do not accept string "inf" !
        except (TypeError, ValueError):
            return False

    @property
    def weights(self) -> Dict[Edge, List[float]]:
        return self._labels

    def weight(self, node1: Node, node2: Node, *, aggregator=min, default: float = inf) -> float:
        """
        Return the weight of the edge joining node1 and node2.

        Args:
            node1: Node
            node2: Node
            aggregator: function used to aggregate values if there are several edges
            between `node1` and `node2` (default is `min`)
            default: value returned if `node1` and `node2` are not adjacents (default is `inf`).
        Return:
            float
        """
        if node1 == node2:
            return 0
        values = self.weights.get(self._edge(node1, node2), [default])
        return aggregator(values)

    def _tikz_labels(self, node1: Node, node2: Node) -> list[str]:
        """Overwrite this method to modify tikz value for some labels."""
        labels = self._labels.get(self._edge(node1, node2), [])

        def format_(label):
            # Note: math.isinf() supports `sympy.oo` too.
            if label is None:
                return ""
            elif math.isinf(label) and label > 0:
                return r"$\infty$"
            elif math.isinf(label) and label < 0:
                return r"$-\infty$"
            else:
                return str(label)

        return [format_(label) for label in labels]

    @cached_property
    def total_weight(self) -> float:
        """Return the sum of all edges weights."""
        return sum(sum(values) for values in self.weights.values())

    @staticmethod
    @abstractmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> List[Tuple[int, int, float]]:
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
                if not cls._is_weight(val):
                    raise ValueError(f"All matrix values must be positive real numbers, but {val!r} is not.")

        edges = cls._get_edges_from_weights_matrix(M)

        g = cls(range(1, n + 1), *edges)  # type: ignore
        if nodes_names:
            g.rename_nodes(dict(enumerate(list(nodes_names)[: len(g.nodes)], start=1)))  # type: ignore
        return g


class WeightedGraph(AbstractWeightedGraph, LabeledGraph):
    """A weighted undirected graph, i.e. an undirected graph where all edges have a positive weight."""

    @staticmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> List[Tuple[int, int, float]]:
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
        connected_nodes: List[Node] = [last_connected_node]  # type: ignore
        weighted_edges: List[WeightedEdge] = []

        # Nodes which may be connected, with the current cost of connection, i.e. the minimal edge's weight
        # enabling to connect it to the spanning tree.
        cheapest_cost: Dict[Node, float] = {}  # type: ignore
        cheapest_edge: Dict[Node, Node] = {}  # type: ignore
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


class WeightedDirectedGraph(AbstractWeightedGraph, LabeledDirectedGraph):
    """A directed graph with weights (i.e. positive floats) associated to its edges."""

    @staticmethod
    def _get_edges_from_weights_matrix(
        matrix: Sequence[Sequence[float]],
    ) -> List[Tuple[int, int, float]]:
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
