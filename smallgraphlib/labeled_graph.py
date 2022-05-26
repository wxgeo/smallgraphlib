import ast
import math
import re
from abc import ABC
from itertools import chain
from numbers import Real
from typing import Iterable, Tuple, Dict, List, TypeVar, Generic, Any

from smallgraphlib.graph import Graph, Node, Edge, AbstractGraph, DirectedGraph
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
        self.labels: Dict[Edge, List[Label]] = {}
        for *edge, label in labeled_edges:
            edges.append(edge)  # type: ignore
            self.labels.setdefault(self._edge(*edge), []).append(label)

        super().__init__(nodes, *edges, sort_nodes=sort_nodes)

    @classmethod
    def from_dict(cls, edge_label_dict: dict = None, /, **edge_label):
        """LabeledUndirectedGraph.from_dict(AB=1, AC=3, BC=4)"""
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

    def weight(self, node1, node2) -> float:
        return min(self.weights[self._edge(node1, node2)], default=inf)


class WeightedGraph(AbstractWeightedGraph, LabeledGraph):
    pass


class WeightedDirectedGraph(AbstractWeightedGraph, LabeledDirectedGraph):
    pass
