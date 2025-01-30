#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 12:24:58 2022

@author: nicolas
"""
from typing import (
    FrozenSet,
    Sequence,
    Generic,
    Self,
    Type,
)

from smallgraphlib.core import (
    AbstractGraph,
    InvalidGraphAttribute,
)
from smallgraphlib.custom_types import Node, DirectedEdge, UndirectedEdge
from smallgraphlib.utilities import (
    cached_property,
)


class Graph(AbstractGraph, Generic[Node]):
    """A graph with undirected edges.

    >>> G = Graph((1, 2, 3), {1, 3}, {1, 2}, {2, 1}, {1})
    """

    @classmethod
    def from_subgraphs(cls: "Type[Graph]", subgraphs: str) -> "Graph":
        """Generate a (simple) graph from a string listing complete subgraphs.

        Nodes in a subgraph are separated with commas, and subgraphs are separated with spaces (at least one).

        >>> Graph.from_subgraphs("A,B,C A,E D,F")
        Graph(('A', 'B', 'C', 'D', 'E', 'F'), {'A', 'B'}, {'A', 'C'}, {'A', 'E'}, {'B', 'C'}, {'D', 'F'})

        """
        subgraphs_list: list[list[Node]] = [
            [node.strip() for node in subgraph.split(",")] for subgraph in subgraphs.split()  # type: ignore
        ]
        nodes: set[Node] = {node for subgraph in subgraphs_list for node in subgraph}
        edges: set[tuple[Node, Node]] = set()
        for subgraph in subgraphs_list:
            for node1 in subgraph:
                for node2 in subgraph:
                    if node1 < node2:
                        edges.add((node1, node2))

        return Graph(nodes, *edges)

    @staticmethod
    def _get_edges_from_adjacency_matrix(
        matrix: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        edges = []
        for i in range(len(matrix)):
            for j in range(i + 1):  # we must only deal with i <= j, since it is an undirected graph.
                if i == j:
                    if matrix[i][i] % 2 == 1:
                        raise ValueError(
                            "The adjacency matrix of an undirected graph must have "
                            "even diagonal coefficients, "
                            f"but matrix[{i}][{i}]={matrix[i][i]}."
                        )
                    edge_multiplicity = matrix[i][i] // 2
                else:
                    if matrix[i][j] != matrix[j][i]:
                        raise ValueError(
                            "The adjacency matrix of an undirected graph must be symmetric, "
                            f"but matrix[{i}][{j}]={matrix[i][j]} != matrix[{j}][{i}]={matrix[j][i]}"
                        )
                    edge_multiplicity = matrix[i][j]
                edges.extend(edge_multiplicity * [(i + 1, j + 1)])
        return edges

    def is_isomorphic_to(self, other) -> bool:
        return super().is_isomorphic_to(other)

    def __repr__(self):
        # Sort nodes in edges for undirected graphs, so that doctests
        # can be deterministic.
        sorted_edges = sorted(sorted(edge) for edge in self.edges)

        def repr_edge(edge: list[Node]) -> str:
            if len(edge) == 2:
                node1, node2 = sorted(edge)
                return f"{{{node1!r}, {node2!r}}}"
            else:
                assert len(edge) == 1
                (node,) = edge
                return f"{{{node!r}}}"

        edges = ", ".join(repr_edge(edge) for edge in sorted_edges)
        return f"Graph({tuple(self.nodes)!r}, {edges})"

    def simplify(self, remove_loops: bool = True) -> Self:
        """Return the simplified graph: loops and duplicate edges are removed."""
        return self.__class__(
            self.nodes, *(edge for edge in self.edges_set if len(edge) == 2 or not remove_loops)
        )

    @property
    def is_directed(self):
        return False

    @staticmethod
    def _edge(node1: Node, node2: Node = None) -> UndirectedEdge:
        if node2 is None:
            node2 = node1
        return frozenset((node1, node2))

    def _count_odd_degrees(self):
        return sum(self.node_degree(node) % 2 for node in self.nodes)

    @cached_property
    def is_eulerian(self):
        return self._count_odd_degrees() == 0

    @cached_property
    def is_semi_eulerian(self):
        return self._count_odd_degrees() == 2

    @cached_property
    def is_connected(self):
        return self._test_connection_from_node(next(iter(self.nodes)))

    @cached_property
    def greedy_coloring(self) -> dict[Node, int]:
        coloring: dict[Node, int] = {}
        # Sort nodes by reversed degree, then alphabetically
        nodes = sorted(self.nodes, key=(lambda _node: (-self.node_degree(_node), _node)))
        for node in nodes:
            color_num = 0
            while any(coloring.get(adjacent) == color_num for adjacent in self.successors(node)):
                color_num += 1
            coloring[node] = color_num
        return coloring

    def are_adjacents(self, node1: Node, node2: Node) -> bool:
        return node2 in self.successors(node1)

    @property
    def as_weighted_graph(self):
        from smallgraphlib import WeightedGraph

        def weighted_edge(edge):
            start, end = self._get_edge_extremities(edge)
            return start, end, 1

        return WeightedGraph(self.nodes, *(weighted_edge(edge) for edge in self.edges))

    def is_subgraph_stable(self, *nodes: Node) -> bool:
        return not any(self.are_adjacents(node1, node2) for node1 in nodes for node2 in nodes)

    @cached_property
    def is_complete_bipartite(self) -> bool:
        """Is this graph a complete bipartite graph ?

        Example:
            >>> from smallgraphlib import complete_bipartite_graph
            >>> K33 = complete_bipartite_graph(3, 3)
            >>> K33.is_complete_bipartite
            True
        """
        nodes_group = self.successors(self.nodes[0])
        other_group = self.nodes_set - nodes_group
        if not self.is_subgraph_stable(*nodes_group):
            return False
        if not self.is_subgraph_stable(*other_group):
            return False
        return all(self.are_adjacents(node1, node2) for node1 in nodes_group for node2 in other_group)


class DirectedGraph(AbstractGraph, Generic[Node]):
    """A graph with directed edges.

    >>> G = DirectedGraph((1, 2, 3), (1, 3), (1, 2), (2, 1), (1, 1))
    """

    @staticmethod
    def _get_edges_from_adjacency_matrix(
        matrix: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        edges = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                edge_multiplicity = matrix[i][j]
                edges.extend(edge_multiplicity * [(i + 1, j + 1)])
        return edges

    def is_isomorphic_to(self, other) -> bool:
        return super().is_isomorphic_to(other)

    def __repr__(self):
        edges = sorted(repr(edge) for edge in self.edges)
        return f"DirectedGraph({tuple(self.nodes)!r}, {', '.join(edges)})"

    @property
    def is_directed(self):
        return True

    @staticmethod
    def _edge(node1: Node, node2: Node = None) -> DirectedEdge:
        if node2 is None:
            node2 = node1
        return node1, node2

    @cached_property
    def is_eulerian(self):
        return all(self.out_degree(node) == self.in_degree(node) for node in self.nodes)

    @cached_property
    def is_semi_eulerian(self):
        start_found = end_found = False
        for node in self.nodes:
            out_degree = self.out_degree(node)
            in_degree = self.in_degree(node)
            if out_degree == in_degree + 1 and not start_found:
                start_found = True
            elif out_degree == in_degree - 1 and not end_found:
                end_found = True
            elif out_degree != in_degree:
                return False
        return start_found and end_found

    @cached_property
    def sources(self) -> frozenset[Node]:
        return frozenset(node for node, in_degree in self.all_in_degrees.items() if in_degree == 0)

    @cached_property
    def sinks(self) -> frozenset[Node]:
        return frozenset(node for node, out_degree in self.all_out_degrees.items() if out_degree == 0)

    def simplify(self, remove_loops: bool = True) -> Self:
        """Return the simplified graph: loops and duplicate edges are removed."""
        return self.__class__(
            self.nodes, *(edge for edge in self.edges_set if edge[0] != edge[1] or not remove_loops)
        )

    @cached_property
    def levels(self) -> tuple[FrozenSet[Node], ...]:
        graph = self.copy()
        levels: list[FrozenSet] = []
        while True:
            level = set()
            for node in graph.nodes:
                if graph.out_degree(node) == 0:  # is node a sink ?
                    level.add(node)
            if len(level) == 0:
                break
            graph.remove_nodes(*level)
            levels.append(frozenset(level))
        if graph.order != 0:
            raise InvalidGraphAttribute("Can't split the graph into levels, since it has a closed path.")
        return tuple(levels)

    @cached_property
    def kernel(self) -> FrozenSet[Node]:
        graph = self.copy()
        kernel = set()
        while True:
            nodes_to_remove = set()
            for node in graph.nodes:
                if graph.out_degree(node) == 0:  # is node a sink ?
                    kernel.add(node)
                    nodes_to_remove.add(node)
                    nodes_to_remove |= graph.predecessors(node)

            if len(nodes_to_remove) == 0:
                break
            graph.remove_nodes(*nodes_to_remove)
        if graph.order != 0:
            raise InvalidGraphAttribute("Can't compute the graph's kernel, since it has a closed path.")
        return frozenset(kernel)

    @cached_property
    def has_cycle(self) -> bool:
        return not hasattr(self, "levels")

    @property
    def reversed_graph(self):
        return DirectedGraph(self.nodes, *(edge[::-1] for edge in self.edges))

    @property
    def undirected_graph(self):
        return Graph(self.nodes, *self.edges)

    @property
    def weighted_graph(self):
        from smallgraphlib import WeightedDirectedGraph

        return WeightedDirectedGraph(self.nodes, *(edge + (1,) for edge in self.edges))

    @cached_property
    def is_strongly_connected(self):
        node = next(iter(self.nodes))
        return self._test_connection_from_node(node) and self.reversed_graph._test_connection_from_node(node)

    @cached_property
    def is_connected(self):
        return self.undirected_graph.is_connected

    def are_adjacents(self, node1: Node, node2: Node) -> bool:
        return node2 in (self.successors(node1) | self.predecessors(node1))

    @staticmethod
    def _apply_transitivity_to(M):
        # Make a mutable copy of M
        M = [list(line) for line in M]
        n = len(M)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    M[i][j] = M[i][j] or (M[i][k] and M[k][j])
        return M

    @cached_property
    def transitive_closure_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Return the matrix of the transitive closure."""
        # Make a mutable copy of the adjacency matrix.
        M = [list(line) for line in self.adjacency_matrix]

        while True:
            new_M = self._apply_transitivity_to(M)
            if new_M == M:
                # Must be immutable, because of caching.
                return tuple(tuple(line) for line in M)
            M = new_M

    @cached_property
    def transitive_closure(self) -> Self:
        return self.__class__.from_matrix(self.transitive_closure_matrix, self.nodes)

    @cached_property
    def is_transitive(self) -> bool:
        """Test if the graph is transitive.

        A directed graph is transitive if for every i->j and j->k edges,
        there is also an i->k edge."""
        return self.adjacency_matrix == self.transitive_closure_matrix

    @cached_property
    def transitive_reduction(self) -> Self:
        """Compute the transitive reduction of the graph."""
        g = self.simplify(remove_loops=False)

        def filter_edges(_: Self, n1: Node, n2: Node) -> bool:
            return (node1, node2) != (n1, n2)

        for edge in g.edges:
            node1, node2 = edge
            if g.find_path(node1, node2, _filter_edges=filter_edges):
                g.remove_edges(edge)
        return g

    @cached_property
    def is_totally_ordered(self) -> bool:
        if len(self.sources) != 1:
            return False
        if len(self.sinks) != 1:
            return False
        g = self.transitive_reduction
        if g.degree != g.order - 1:
            return False
        already_seen = set(g.sources)
        assert len(already_seen) == 1, already_seen
        current = next(iter(g.sources))
        while g.out_degree(current) == 1:
            current = next(iter(g.successors(current)))
            if current in already_seen:
                return False
            already_seen.add(current)
        return len(already_seen) == g.order
