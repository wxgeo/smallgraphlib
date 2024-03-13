from typing import Iterable, Generic, Callable, Self

from smallgraphlib.labeled_graphs import WeightedDirectedGraph

from smallgraphlib.custom_types import Node
from smallgraphlib.utilities import cached_property, clear_cache

CapacityEdge = tuple[Node, Node, int]


class FlowNetwork(WeightedDirectedGraph, Generic[Node]):
    def __init__(
        self,
        nodes: Iterable[Node],
        *weighted_edges: CapacityEdge,
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

    def get_residual(self, current_flow: "FlowNetwork[Node]") -> WeightedDirectedGraph[Node]:
        capacity = self.as_dict()
        current = current_flow.as_dict()
        if set(capacity) != set(current):
            raise ValueError("Both flows must have the same edges.")
        residual: dict[tuple[Node, Node], int] = {}
        for key in capacity:
            residual[key] = capacity[key] - current[key]
            residual[(key[1], key[0])] = current[key]
        return WeightedDirectedGraph.from_dict(residual)

    def find_path(
        self,
        start: Node,
        end: Node,
        _filter: Callable[[Self, Node, Node], bool] = (
            lambda self, node1, node2: self.weight(node1, node2) > 0
        ),
    ) -> list[Node]:
        return super().find_path(start, end, _filter=_filter)

    def get_max_flow(self) -> "FlowNetwork":
        def _filter(self, node1, node2):
            return self.weight(node1, node2) > 0

        flow: FlowNetwork[Node] = FlowNetwork.from_dict(dict.fromkeys(self.as_dict(), 0))
        while (
            len(
                path := (residual := self.get_residual(flow)).find_path(
                    self.source, self.sink, _filter=_filter
                )
            )
            > 0
        ):
            additional_capacity = residual.get_path_capacity(path)
            assert additional_capacity > 0
            for node1, node2 in zip(path[:-1], path[1:]):
                flow.set_capacity(node1, node2, flow.weight(node1, node2) + additional_capacity)
        return flow

    def get_max_flow_value(self) -> float:
        return sum(
            self.get_max_flow().weight(self.source, successor) for successor in self.successors(self.source)
        )

    @clear_cache
    def set_capacity(self, node1, node2, value: int) -> None:
        self._labels[(node1, node2)] = [value]

    # def get_residual_network(self, flow: "Flow") -> "Flow":
    #     ...
