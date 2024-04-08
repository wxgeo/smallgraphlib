import ast
import math
import re
from abc import ABC
from itertools import chain
from typing import Iterable, TypeVar, Generic, Any, Type

from smallgraphlib.basic_graphs import Graph, DirectedGraph
from smallgraphlib.core import AbstractGraph
from smallgraphlib.custom_types import Node, Edge, Label, LabeledEdge
from smallgraphlib.tikz_export import TikzLabeledGraphPrinter
from smallgraphlib.utilities import cached_property

_AbstractLabeledGraph = TypeVar("_AbstractLabeledGraph", bound="AbstractLabeledGraph")


class AbstractLabeledGraph(AbstractGraph, ABC, Generic[Node, Label]):
    """Abstract class for all labeled graphs, don't use it directly."""

    printer = TikzLabeledGraphPrinter

    def __init__(
        self,
        nodes: Iterable[Node],
        *labeled_edges: LabeledEdge,
        sort_nodes: bool = True,
    ):
        edges: list[Edge] = []
        self._labels: dict[Edge, list[Label]] = {}
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

    def copy(self):
        return self.__class__(self.nodes, *self.labeled_edges)

    @cached_property
    def labeled_edges(self) -> tuple[LabeledEdge, ...]:
        # noinspection PyTypeChecker
        return tuple(
            sorted(
                (
                    *(
                        sorted(edge) if isinstance(edge, (set, frozenset)) else edge
                    ),  # Sorting nodes will help in making doctests deterministic.
                    label,
                )
                for edge, labels in self._labels.items()
                for label in labels
            )
        )

    def as_dict(self) -> dict[tuple[Node, Node], Label]:
        return {(node1, node2): label for node1, node2, label in self.labeled_edges}

    @classmethod
    def from_dict(
        cls: Type[_AbstractLabeledGraph],
        edge_label_dict: dict[str | tuple[Node, Node], Label] = None,  # type: ignore
        /,
        **edge_label: Label,  # type: ignore
    ) -> _AbstractLabeledGraph:
        """Construct a directed graph using a {edge_name: label} dictionnary (or keywords).

        All edges' names must be either:
            - two letters strings (like "AB"), each letter representing a node,
            - or a couple of nodes (like ("A", "B") or (1, 2)).
        Nodes' names are automatically deduced from edges' names.

        >>> g1 = LabeledGraph.from_dict(AB=1, AC=3, BC=4)
        >>> g2 = LabeledGraph.from_dict({"AB": 1, "AC": 3, "BC": 4})
        >>> g3 = LabeledGraph.from_dict({("A", "B"): 1, ("A", "C"): 3, ("B", "C"): 4})
        >>> g1 == g2 == g3
        True
        >>> g1.nodes
        ('A', 'B', 'C')
        """
        if edge_label_dict is None:
            edge_label_dict = {}
        edge_label_dict.update(edge_label)  # type: ignore
        nodes = set(chain(*(edge for edge in edge_label_dict)))
        return cls(nodes, *((*edge, label) for edge, label in edge_label_dict.items()))

    @classmethod
    def from_string(cls: Type[_AbstractLabeledGraph], string: str) -> _AbstractLabeledGraph:
        """LabeledGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
        will generate a graph of 4 nodes, A, B, C and D, with edges A->B, A->C, B->C and C->D
        and respective labels 'label', 'other label', 5 and `None`.

        >>> LabeledGraph.from_string("A:B=label,C='other label' B:C=5 C D:C")
        LabeledGraph(('A', 'B', 'C', 'D'), ('A', 'B', 'label'), ('A', 'C', 'other label'),
                     ('B', 'C', 5), ('C', 'D', None))

        Note that labels containing a space must be surrounded by single or double quotes.
        Labels containing only digits will be converted to numbers (integers or floats),
        except if surrounded by quotes.

        If no label is given, `None` is stored by default.
        """
        # Convert spaces inside labels to null characters, to make splitting easier.
        string = re.sub("""'[^']*'|"[^"]*""", (lambda m: m.group().replace(" ", "\x00")), string)
        nodes: list[str] = []
        edges: list[tuple[str, str, Any]] = []
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
        return cls(nodes, *edges)

    def labels(self, node1: Node, node2: Node) -> list[str]:
        labels = self._labels.get(self._edge(node1, node2), [])
        assert len(labels) == self.count_edges(node1, node2, count_undirected_loops_twice=False)
        return [str(label if label is not None else "") for label in labels]

    def rename_nodes(self, node_names: dict[Node, Node]) -> None:
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
    """A labeled directed graph."""
