import math
from typing import Iterable, Generic, Callable, Self

from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.weighted_graphs import AbstractNumericGraph

from smallgraphlib.custom_types import Node, WeightedEdge
from smallgraphlib.tikz_export import TikzFlowNetworkPrinter
from smallgraphlib.utilities import cached_property, clear_cache

# CapacityEdge = tuple[Node, Node, int]


class Network(AbstractNumericGraph, DirectedGraph, Generic[Node]):
    """A network is a directed graph whose edges are labeled by positive numbers.

    The value associated with each edge is called its capacity."""

    def capacity(self, node1: Node, node2: Node):
        return math.inf if node1 == node2 else self._edge_value(node1, node2, aggregator=sum, default=0)

    def get_path_capacity(self, path: list[Node]):
        """Return the capacity of the path, which is the smallest weight of any of its edges."""
        return min(self.capacity(node1, node2) for node1, node2 in zip(path[:-1], path[1:]))

    @clear_cache
    def set_capacity(self, node1, node2, value: float) -> None:
        self._labels[(node1, node2)] = [value]


class FlowNetwork(Network, Generic[Node]):
    """A network with a unique source and a unique sink."""

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

    def get_residual(self, current_flow: "FlowNetwork[Node]") -> Network[Node]:
        """Return the residual flow, i.e. the currently unused capacity of each edge.

        The residual flow also includes the removable capacity:
        for each currently used edge, the residual will contain the reversed edge,
        with the current flow as value.

            >>> from smallgraphlib.flow_networks import FlowNetwork
            >>> maximal_f = FlowNetwork(("A", "B"), ("A", "B", 5))
            >>> current_f = FlowNetwork(("A", "B"), ("A", "B", 3))
            >>> maximal_f.get_residual(current_f)
            Network(('A', 'B'), ('A', 'B', 2), ('B', 'A', 3))
        """
        capacity = self.as_dict()
        current = current_flow.as_dict()
        if set(capacity) != set(current):
            print(f"{capacity=}")
            print(f"{current=}")
            raise ValueError("Both flows must have the same edges.")
        residual: dict[tuple[Node, Node], float] = {}
        for key in capacity:
            if current[key] > capacity[key]:
                raise ValueError(
                    f"Current flow ({current[key]}) exceeds capacity ({capacity[key]}) for edge {key}."
                )
            residual[key] = capacity[key] - current[key]
            residual[(key[1], key[0])] = current[key]
        # Residual network may not have a source or a sink,
        # so declare it as a WeightedDirectedGraph instead.
        return Network.from_dict(residual)  # type: ignore

    def find_path(
        self,
        start: Node,
        end: Node,
        _filter_edges: Callable[[Self, Node, Node], bool] = (
            lambda self, node1, node2: self.capacity(node1, node2) > 0
        ),
        _filter_nodes: Callable[[Self, Node], bool] | Iterable[Node] = (),
    ) -> list[Node]:
        return super().find_path(start, end, _filter_edges=_filter_edges, _filter_nodes=_filter_nodes)

    def get_max_flow(self) -> "FlowNetwork":
        def _filter_edges(network: Network, node1_: Node, node2_: Node) -> bool:
            return network.capacity(node1_, node2_) > 0

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
            assert additional_capacity > 0 and not math.isinf(additional_capacity)
            for node1, node2 in zip(path[:-1], path[1:]):
                if (node1, node2) in flow.edges:
                    assert additional_capacity <= self.capacity(node1, node2)
                    flow.set_capacity(node1, node2, flow.capacity(node1, node2) + additional_capacity)
                    assert flow.capacity(node1, node2) <= self.capacity(node1, node2)
                else:
                    assert (node2, node1) in flow.edges
                    flow.set_capacity(node2, node1, flow.capacity(node2, node1) - additional_capacity)
                    assert flow.capacity(node1, node2) >= 0
        return flow

    def get_max_flow_value(self) -> float:
        return sum(
            self.get_max_flow().capacity(self.source, successor) for successor in self.successors(self.source)
        )
