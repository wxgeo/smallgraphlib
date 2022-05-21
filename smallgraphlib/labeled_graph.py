from abc import ABC
from itertools import chain
from numbers import Real
from typing import Iterable, Tuple, Dict, List, TypeVar, Generic

from smallgraphlib.graph import Graph, Node, UndirectedEdge, Edge, AbstractGraph, DirectedGraph
from math import inf

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
        self.labels: Dict[UndirectedEdge, List[Label]] = {}
        for *edge, label in labeled_edges:
            edges.append(edge)  # type: ignore
            self.labels.setdefault(frozenset(edge), []).append(label)

        super().__init__(nodes, *edges, sort_nodes=sort_nodes)

    @classmethod
    def from_dict(cls, edge_label_dict: dict = None, /, **edge_label):
        """LabeledUndirectedGraph.from_dict(AB=1, AC=3, BC=4)"""
        if edge_label_dict is None:
            edge_label_dict = {}
        edge_label_dict.update(edge_label)
        nodes = set(chain(*(edge for edge in edge_label_dict)))
        return cls(nodes, *((*edge, label) for edge, label in edge_label_dict.items()))  # type: ignore


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

    def weight(self, node1, node2) -> float:
        return min(self.weights[self._edge(node1, node2)], default=inf)

    def shortest_paths(self, start: Node, end: Node) -> Tuple[float, List[List[Node]]]:
        """Implementation of Dijkstra Algorithm."""
        for node in (start, end):
            if node not in self.nodes:
                raise ValueError(f"Unknown node {node!r}.")
        lengths: Dict[Node, float] = {node: (0 if node == start else inf) for node in self.nodes}
        last_step: Dict[Node, List[Node]] = {node: [] for node in self.nodes}
        never_selected_nodes = set(self.nodes)
        selected_node = start
        while selected_node != end:
            never_selected_nodes.remove(selected_node)
            for successor in self.successors(selected_node):
                weight = self.weight(selected_node, successor)
                if weight < 0:
                    raise ValueError("Can't find shortest paths with negative weights.")
                new_length = lengths[selected_node] + weight
                if new_length < lengths[successor]:
                    lengths[successor] = new_length
                    last_step[successor] = [selected_node]
                elif new_length == lengths[successor]:
                    last_step[successor].append(selected_node)
            selected_node = min(never_selected_nodes, key=(lambda node_: lengths[node_]))

        def generate_paths(path: List[Node]) -> List[List[Node]]:
            if path[0] == start:
                return [path]
            paths = []
            for predecessor in last_step[path[0]]:
                paths.extend(generate_paths([predecessor] + path))
            return paths

        return lengths[end], generate_paths([end])


class WeightedGraph(AbstractWeightedGraph, LabeledGraph):
    pass


class WeightedDirectedGraph(AbstractWeightedGraph, LabeledDirectedGraph):
    pass
