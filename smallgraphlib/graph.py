#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 12:24:58 2022

@author: nicolas
"""
from abc import abstractmethod
from collections import Counter
from itertools import chain
from typing import Set, Union, Tuple, Dict, List, Iterable, FrozenSet, Counter as CounterType
import typing


class Comparable(typing.Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other) -> bool:
        pass


# Node = TypeVar("Node", bound=typing.Hashable)  # too subtile for Pycharm
Node = typing.TypeVar("Node", bound=Comparable)
FrozenEdge = Union[Tuple[Node, ...], FrozenSet[Node]]
Edge = Union[FrozenEdge, Set[Node], Iterable[Node]]


class CycleFoundError(RuntimeError):
    pass


class Graph(typing.Generic[Node]):
    def __init__(self, nodes: Iterable[Node], *edges: Edge, directed: bool = True):
        """Create a graph of nodes, when nodes may be any hashable objects.

        Edges may be any iterable of two nodes. If edge
        """
        self._is_directed = directed
        self._links: Dict[Node, CounterType[Node]] = {}
        self._inverse_links: Dict[Node, CounterType[Node]] = {}
        # Nodes must be added before edges.
        self.add_nodes(*nodes)
        self.add_edges(*edges)

    def _test_edges(self, *edges: Edge) -> None:
        edges_nodes = set(chain(*edges))
        if not edges_nodes.issubset(self.nodes):
            raise ValueError(f"Undeclared nodes: {edges_nodes - self.nodes}")

    def add_nodes(self, *new_nodes: Node) -> None:
        for node in new_nodes:
            if node in self._links:
                raise RuntimeError(f"Node already present: {node!r}")
            self._links[node] = Counter()
            self._inverse_links[node] = Counter()

    def remove_nodes(self, *nodes: Node) -> None:
        for node in nodes:
            if node not in self.nodes:
                raise ValueError(f"Node {node} not found!")
        for node in nodes:
            successors = self._links.pop(node, ())
            for successor in successors:
                self._inverse_links.get(successor, {}).pop(node, None)  # type: ignore
            predecessors = self._inverse_links.pop(node, ())
            for predecessor in predecessors:
                self._links.get(predecessor, {}).pop(node, None)  # type: ignore

    def add_edges(self, *new_edges: Edge) -> None:
        self._test_edges(*new_edges)
        for edge in new_edges:
            nodes = iter(edge)
            start = next(nodes)
            try:
                end = next(nodes)
            except StopIteration:
                end = start
            try:
                next(nodes)
                raise ValueError(f"Too many values for an edge: {edge!r}")
            except StopIteration:
                pass
            self._links[start][end] += 1
            self._inverse_links[end][start] += 1
            if isinstance(edge, (set, frozenset)) or not self.is_directed:
                # This is a bidirectional edge.
                # Note that bidirectional loops are added twice too.
                self._links[end][start] += 1
                self._inverse_links[start][end] += 1

    @property
    def is_directed(self) -> bool:
        return self._is_directed

    @property
    def order(self) -> int:
        return len(self._links)

    @property
    def degree(self) -> int:
        return sum(self.node_degree(node) for node in self.nodes) // 2

    def __repr__(self):
        edges = (repr(set(edge) if isinstance(edge, frozenset) else list(edge)) for edge in self.edges)
        return f"Graph({self.nodes!r}, {', '.join(edges)})"

    @property
    def nodes(self) -> Set[Node]:
        return set(self._links)

    @property
    def edges(self) -> List[Edge]:
        edges: typing.Counter[Edge] = Counter()
        for node in self._links:
            successors: typing.Counter[Node] = self._links[node]
            for successor, count in successors.items():
                key: Edge = (node, successor)
                if not self.is_directed:
                    key = frozenset(key)
                edges[key] += count
        if not self.is_directed:
            for key in edges:
                edges[key] //= 2
        return list(edges.elements())

    def are_adjacents(self, node1: Node, node2: Node) -> bool:
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
        return self.is_simple and all(
            self.are_adjacents(i, j) for i in self.nodes for j in self.nodes if i != j
        )

    def _test_connection_from_node(self, node: Node) -> bool:
        """Test if every other node can be accessed from node `node`"""
        if self.order <= 1:
            return True
        connected_nodes = set()
        border = {node}
        while border:
            node = border.pop()
            connected_nodes.add(node)
            border |= set(self._links[node]) - connected_nodes
        return len(connected_nodes) == self.order

    @property
    def is_connected(self):
        if self.is_directed:
            return self.undirected_graph.is_connected
        return self._test_connection_from_node(next(iter(self.nodes)))

    @property
    def is_strongly_connected(self):
        if self.is_directed:
            node = next(iter(self.nodes))
            return self._test_connection_from_node(node) and self.reversed_graph._test_connection_from_node(
                node
            )
        return self.is_connected

    @property
    def reversed_graph(self):
        return Graph(
            self.nodes,
            *(((edge[1], edge[0]) if isinstance(edge, tuple) else edge) for edge in self.edges),
            directed=self.is_directed,
        )

    @property
    def undirected_graph(self):
        return Graph(self.nodes, *self.edges, directed=False)

    def count_edges(self, node1: Node, node2: Node):
        """Count the number of edges from node1 to node2. Note that undirected loops are counted twice."""
        # sum(1 for node in self._links[node1] if node == node2)
        return self._links[node1][node2]

    def node_degree(self, node: Node) -> int:
        if self.is_directed:
            return self.in_degree(node) + self.out_degree(node)
        return self.out_degree(node)

    def in_degree(self, node: Node) -> int:
        return sum(self._inverse_links[node].values())

    def out_degree(self, node: Node) -> int:
        return sum(self._links[node].values())

    def copy(self):
        return Graph(self.nodes, *self.edges)

    @property
    def levels(self) -> List[Set[Node]]:
        if not self.is_directed:
            raise CycleFoundError("An undirected graph has no levels.")
        graph = self.copy()
        levels = []
        while True:
            level = set()
            for node in graph.nodes:
                if graph.out_degree(node) == 0:  # is node a sink ?
                    level.add(node)
            if len(level) == 0:
                break
            graph.remove_nodes(*level)
            levels.append(level)
        if graph.nodes:
            raise CycleFoundError("Can't split the graph into levels, since it has a cycle.")
        return levels

    @property
    def has_cycle(self) -> bool:
        try:
            self.levels  # noqa
            return False
        except CycleFoundError:
            return True

    @property
    def kernel(self) -> Set[Node]:
        if not self.is_directed:
            raise CycleFoundError("An undirected graph has no kernel.")
        graph = self.copy()
        kernel = set()
        while True:
            nodes_to_remove = set()
            for node in graph.nodes:
                if graph.out_degree(node) == 0:  # is node a sink ?
                    kernel.add(node)
                    nodes_to_remove.add(node)
                    nodes_to_remove.update(self._inverse_links[node])

            if len(nodes_to_remove) == 0:
                break
            graph.remove_nodes(*nodes_to_remove)
        if graph.nodes:
            raise CycleFoundError("Can't split the graph into levels, since it has a cycle.")
        return kernel

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
        pair: FrozenSet[Node]
        for pair in nodes_pairs:
            style = "directed" if self.is_directed else "undirected"
            if len(pair) == 1:
                (node,) = pair
                n: int = self._links[node][node]
                if not self.is_directed:
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
                if self.is_directed:
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
