from typing import Iterable
from collections import deque

from smallgraphlib.labeled_graphs import WeightedDirectedGraph, WeightedEdge

from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.custom_types import Node, EdgeLike
from smallgraphlib.utilities import cached_property


class FlowNetwork(WeightedDirectedGraph):
    def __init__(
        self,
        nodes: Iterable[Node],
        *weighted_edges: WeightedEdge,
        sort_nodes: bool = True,
    ):
        self._initialized = False
        super().__init__(nodes, *weighted_edges, sort_nodes=sort_nodes)
        self._initialized = True
        self.on_clear_cache()

    def on_clear_cache(self):
        # Test existence of source and sink.
        if self._initialized:
            _ = self.source
            _ = self.sink

    @cached_property
    def source(self) -> Node:
        sources = [node for node, in_degree in self.all_in_degrees.items() if in_degree == 0]
        if len(sources) > 1:
            raise ValueError(f"Several sources detected: {', '.join(sources)}!")
        elif len(sources) == 0:
            raise ValueError("No source detected!")
        return sources[0]

    @cached_property
    def sink(self) -> Node:
        sinks = [node for node, out_degree in self.all_out_degrees.items() if out_degree == 0]
        if len(sinks) > 1:
            raise ValueError(f"Several sinks detected: {', '.join(sinks)}!")
        elif len(sinks) == 0:
            raise ValueError("No sink detected!")
        return sinks[0]

    def get_path(self, start: Node, end: Node) -> list[Node]:
        """Return a path between nodes `start` and `end`, if any, or an empty list."""
        previous: dict[Node, Node] = {}
        queue: deque[Node] = deque([start])
        while end not in previous and len(queue) > 0:
            # BFS
            node = queue.popleft()
            for successor in self.successors(node):
                # If successor was never seen before, append it to queue.
                if successor not in previous:
                    previous[successor] = node
                    queue.append(successor)
        if end in previous:
            path = [end]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return list(reversed(path))
        return []

    def get_max_flow(self) -> "FlowNetwork":
        flow = self.copy()

    # def get_residual_network(self, flow: "Flow") -> "Flow":
    #     ...
