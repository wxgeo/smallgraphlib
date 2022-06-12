#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 12:24:58 2022

@author: nicolas
"""
from typing import (
    Tuple,
    Dict,
    List,
    FrozenSet,
)

from smallgraphlib.core import Node, DirectedEdge, UndirectedEdge, AbstractGraph
from smallgraphlib.utilities import (
    cached_property,
    CycleFoundError,
)


class Graph(AbstractGraph):
    """A graph with undirected edges.

    >>> G = Graph((1, 2, 3), {1, 3}, {1, 2}, {2, 1}, {1})
    """

    def is_isomorphic_to(self, other) -> bool:
        return super().is_isomorphic_to(other)

    def __repr__(self):
        edges = (repr(set(edge)) for edge in self.edges)
        return f"Graph({tuple(self.nodes)!r}, {', '.join(edges)})"

    @property
    def is_directed(self):
        return False

    @staticmethod
    def _edge(node1, node2) -> UndirectedEdge:
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
    def greedy_coloring(self) -> Dict[Node, int]:
        coloring: Dict[Node, int] = {}
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
    def weighted_graph(self):
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


class DirectedGraph(AbstractGraph):
    """A graph with directed edges.

    >>> G = DirectedGraph((1, 2, 3), (1, 3), (1, 2), (2, 1), (1, 1))
    """

    def is_isomorphic_to(self, other) -> bool:
        return super().is_isomorphic_to(other)

    def __repr__(self):
        edges = (repr(edge) for edge in self.edges)
        return f"DirectedGraph({tuple(self.nodes)!r}, {', '.join(edges)})"

    @property
    def is_directed(self):
        return True

    @staticmethod
    def _edge(node1, node2) -> DirectedEdge:
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
    def levels(self) -> Tuple[FrozenSet[Node], ...]:
        graph = self.copy()
        levels: List[FrozenSet] = []
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
            raise CycleFoundError("Can't split the graph into levels, since it has a closed path.")
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
                    nodes_to_remove |= self.predecessors(node)

            if len(nodes_to_remove) == 0:
                break
            graph.remove_nodes(*nodes_to_remove)
        if graph.order != 0:
            raise CycleFoundError("Can't compute the graph's kernel, since it has a closed path.")
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
