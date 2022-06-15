import ast
import math
import re
from abc import ABC
from itertools import chain
from math import inf
from numbers import Real
from typing import Iterable, Tuple, Dict, List, TypeVar, Generic, Any, Optional

from smallgraphlib.basic_graphs import Graph, DirectedGraph
from smallgraphlib.core import Node, Edge, AbstractGraph
from smallgraphlib.utilities import cached_property

Label = TypeVar("Label")
LabeledEdge = Tuple[Node, Node, Label]
WeightedEdge = Tuple[Node, Node, float]


class AbstractLabeledGraph(AbstractGraph, ABC, Generic[Label]):
    def __init__(
        self,
        nodes: Iterable[Node],
        *labeled_edges: LabeledEdge,
        sort_nodes: bool = True,
    ):
        edges: List[Edge] = []
        self.labels: Dict[Edge, List[Label]] = {}
        for *edge, label in labeled_edges:
            edges.append(edge)  # type: ignore
            self.labels.setdefault(self._edge(*edge), []).append(label)

        super().__init__(nodes, *edges, sort_nodes=sort_nodes)

    def __eq__(self, other: Any):
        return super().__eq__(other) and all(self.labels[edge] == other.labels[edge] for edge in self.edges)

    @classmethod
    def from_dict(cls, edge_label_dict: dict = None, /, **edge_label):
        """Construct a directed graph using a {edge_name: label} dictionnary (or keywords).

        All edges' names must be two letters strings (like "AB"), each letter representing a node.
        Nodes' names are automatically deduced from edges' names.

        >>> g1 = LabeledUndirectedGraph.from_dict(AB=1, AC=3, BC=4)
        >>> g2 = LabeledUndirectedGraph.from_dict({"AB": 1, "AC": 3, "BC": 4})
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
    def from_string(cls, string: str):
        """LabeledGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
        will generate a graph of 3 nodes, A, B and C, with edges A->B, A->C
        and B->C and respective labels 'label', 'other label' and 5.

        Note that labels containing a space must be surrounded by quotes.
        Labels containing only digits will be converted to numbers (integers or floats),
        except if surrounded by quotes.

        If no label is given, `None` is stored by default.
        """
        # Convert spaces inside labels to null character, to make splitting easier.
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
                        # convert back null character to space.
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


class LabeledGraph(AbstractLabeledGraph, Graph):
    pass


class LabeledDirectedGraph(AbstractLabeledGraph, DirectedGraph):
    pass


class AbstractWeightedGraph(AbstractLabeledGraph, ABC):
    def __init__(
        self,
        nodes: Iterable[Node],
        *weighted_edges: WeightedEdge,
        sort_nodes: bool = True,
    ):
        super().__init__(nodes, *weighted_edges, sort_nodes=sort_nodes)
        self.weights: Dict[Edge, List[float]] = self.labels  # type: ignore
        for weights in self.weights.values():
            for weight in weights:
                if not isinstance(weight, Real):
                    raise ValueError(f"Edge weight {weight!r} is not a real number.")

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
        values = self.weights[self._edge(node1, node2)]
        return aggregator(values) if values else default

    @cached_property
    def total_weight(self) -> float:
        """Return the sum of all edges weights."""
        return sum(sum(values) for values in self.weights.values())


class WeightedGraph(AbstractWeightedGraph, LabeledGraph):
    """A weighted graph, i.e. a graph where all edges have a weight."""

    def minimum_spanning_tree(self) -> Optional[Graph]:
        """Use Prim's algorithm to return a minimum weight spanning tree.

        A spanning tree of a graph G is a subgraph of G who is a tree and contains all the nodes of G.

        If the graph is not connected, return `None`.
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
                (cheapest_edge[last_connected_node], last_connected_node, cheapest_cost[last_connected_node])
            )
            cheapest_cost.pop(last_connected_node)
            cheapest_edge.pop(last_connected_node)

        return None if unreached_nodes else WeightedGraph(connected_nodes, *weighted_edges)


class WeightedDirectedGraph(AbstractWeightedGraph, LabeledDirectedGraph):
    pass
