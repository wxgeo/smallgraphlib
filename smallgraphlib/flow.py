from typing import Iterable, Generic, Callable, Self

from smallgraphlib.labeled_graphs import WeightedDirectedGraph

from smallgraphlib.custom_types import Node, WeightedEdge
from smallgraphlib.tikz_export import TikzFlowNetworkPrinter
from smallgraphlib.utilities import cached_property, clear_cache

# CapacityEdge = tuple[Node, Node, int]


class FlowNetwork(WeightedDirectedGraph, Generic[Node]):
    printer = TikzFlowNetworkPrinter  # type: ignore

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

    def get_residual(self, current_flow: "FlowNetwork[Node]") -> WeightedDirectedGraph[Node]:
        capacity = self.as_dict()
        current = current_flow.as_dict()
        if set(capacity) != set(current):
            raise ValueError("Both flows must have the same edges.")
        residual: dict[tuple[Node, Node], float] = {}
        for key in capacity:
            residual[key] = capacity[key] - current[key]
            residual[(key[1], key[0])] = current[key]
        # Residual network may not have a source or a sink,
        # so declare it as a WeightedDirectedGraph instead.
        return WeightedDirectedGraph.from_dict(residual)  # type: ignore

    def find_path(
        self,
        start: Node,
        end: Node,
        _filter_edges: Callable[[Self, Node, Node], bool] = (
            lambda self, node1, node2: self.weight(node1, node2) > 0
        ),
        _filter_nodes: Callable[[Self, Node], bool] | Iterable[Node] = (),
    ) -> list[Node]:
        return super().find_path(start, end, _filter_edges=_filter_edges, _filter_nodes=_filter_nodes)

    def get_max_flow(self) -> "FlowNetwork":
        def _filter_edges(self_: WeightedDirectedGraph, node1_: Node, node2_: Node) -> bool:
            return self_.weight(node1_, node2_) > 0

        flow: FlowNetwork[Node] = FlowNetwork.from_dict(dict.fromkeys(self.as_dict(), 0))
        while (
            len(
                path := (residual := self.get_residual(flow)).find_path(
                    self.source, self.sink, _filter_edges=_filter_edges
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
    def set_capacity(self, node1, node2, value: float) -> None:
        self._labels[(node1, node2)] = [value]
