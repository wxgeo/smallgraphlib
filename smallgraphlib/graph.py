#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 12:24:58 2022

@author: nicolas
"""
from collections import Counter
from functools import cached_property
from itertools import chain
from typing import TypeVar, Set, Union, Tuple, Dict, List, Iterable, FrozenSet, Counter as CounterType

Node = TypeVar("Node")
Edge = Union[Tuple[Node, ...], Set[Node], FrozenSet[Node]]


class Graph:
    def __init__(self, nodes: Iterable[Node], *edges: Edge):
        self._nodes: Set[Node] = set()
        self._edges: Set[Edge] = set()
        self._links: Dict[Node, CounterType[Node]] = {}
        # Nodes must be added before edges.
        self.add_nodes(*nodes)
        self.add_edges(*edges)

    def _test_edges(self, *edges: Edge) -> None:
        edges_nodes = set(chain(*edges))
        if not edges_nodes.issubset(self.nodes):
            raise RuntimeError(f"Undeclared nodes: {edges_nodes - self.nodes}")

    def add_nodes(self, *new_nodes: Node) -> None:
        for node in new_nodes:
            if node in self._links:
                raise RuntimeError(f"Node already present: {node!r}")
            self._links[node] = Counter()
        self._nodes.update(new_nodes)

    def add_edges(self, *new_edges: Edge) -> None:
        self._test_edges(*new_edges)
        edges_list: List[Edge] = list(new_edges)
        del new_edges
        for i, edge in enumerate(edges_list):
            if isinstance(edge, set):
                edges_list[i] = frozenset(edge)
            else:
                edges_list[i] = tuple(edge)
        self._edges.update(edges_list)
        for edge in edges_list:
            if isinstance(edge, (set, frozenset)):
                bidirectional_edge = True
                assert isinstance(edge, frozenset), edge
                assert 1 <= len(edge) <= 2
                if len(edge) == 1:
                    (start,) = edge
                    end = start
                else:
                    start, end = edge
            else:
                bidirectional_edge = False
                start, end = edge
            self._links[start][end] += 1
            if bidirectional_edge:
                # Note that bidirectional loops are added twice.
                self._links[end][start] += 1

    @property
    def directed(self) -> bool:
        return all(isinstance(edge, tuple) for edge in self._edges)

    @property
    def order(self):
        return len(self._nodes)

    @property
    def degree(self):
        return len(self._edges)

    @cached_property
    def nodes(self):
        return frozenset(sorted(self._nodes))

    @property
    def edges(self):
        return tuple(self._edges)

    def adjacents(self, node1: Node, node2: Node) -> bool:
        return node2 in self._links[node1] or node1 in self._links[node2]

    @property
    def has_loop(self) -> bool:
        return any(node in self._links[node] for node in self._links)

    @property
    def has_multiple_edges(self) -> bool:
        return any(len(self._links[node]) != len(set(self._links[node])) for node in self._links)

    @property
    def is_simple(self) -> bool:
        return not (self.has_loop or self.has_multiple_edges)

    @property
    def is_complete(self) -> bool:
        return self.is_simple and all(self.adjacents(i, j) for i in self.nodes for j in self.nodes if i != j)

    def count_edges(self, node1: Node, node2: Node):
        """Count the number of edges from node1 to node2. Note that undirected loops are counted twice."""
        # sum(1 for node in self._links[node1] if node == node2)
        return self._links[node1][node2]

    @property
    def adjacency_matrix(self):
        nodes = sorted(self.nodes)
        return [[self.count_edges(node1, node2) for node2 in nodes] for node1 in nodes]

    def as_tikz(self) -> str:
        lines: List[str] = [
            r"\begin{tikzpicture}["
            r"every node/.style = {draw, circle,font={\scriptsize},inner sep=2},"
            "directed/.style = {-{Stealth[scale=1.1]}},"
            "reversed/.style = {{Stealth[scale=1.1]}-},"
            "undirected/.style = {},"
            "]"
        ]
        theta = 360 / len(self.nodes)
        angles = {}
        for i, node in enumerate(self.nodes):
            angle = angles[node] = i * theta
            lines.append(rf"\node ({node}) at ({angle}:1cm) {{${node}$}};")
        nodes_pairs = {frozenset((node1, node2)) for node1 in self.nodes for node2 in self.nodes}
        pair: frozenset
        for pair in nodes_pairs:
            style = "directed" if self.directed else "undirected"
            if len(pair) == 1:
                (node,) = pair
                n = self._links[node][node]
                if not self.directed:
                    assert n % 2 == 0, n
                    n //= 2
                if n > 1:
                    raise NotImplementedError(n)
                if n == 1:
                    lines.append(
                        rf"\draw[{style}] ({node}) to "
                        f"[out={angles[node] - 45},in={angles[node] + 45},looseness=5] ({node});"
                    )
            else:
                node1, node2 = pair
                if self.directed:
                    n1 = self._links[node1][node2]
                    n2 = self._links[node2][node1]
                    n = n1 + n2
                    styles = n1 * ["directed"] + n2 * ["reversed"]
                else:
                    n = self._links[node1][node2]
                    styles = n * ["undirected"]
                if n == 0:
                    curves = []
                elif n == 1:
                    curves = [""]
                elif n == 2:
                    curves = ["bend left", "bend right"]
                elif n == 3:
                    curves = ["bend left", "", "bend right"]
                else:
                    raise NotImplementedError(n)
                for style, curve in zip(styles, curves):
                    lines.append(rf"\draw[{style}] ({node1}) to[{curve}] ({node2});")

        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)
