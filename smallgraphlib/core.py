import math
import random
from abc import ABC, abstractmethod
from collections import Counter, deque
from enum import Enum
from itertools import chain
from numbers import Integral
from typing import (
    Tuple,
    FrozenSet,
    Set,
    Iterable,
    Any,
    Generic,
    Counter as CounterType,
    Optional,
    Iterator,
    Sequence,
    Type,
    Callable,
    Self,
)

from smallgraphlib.custom_types import _AbstractGraph, Node, Edge, EdgeLike
from smallgraphlib.printers.latex import latex_matrix, latex_degrees_table, latex_Dijkstra
from smallgraphlib.utilities import cached_property, Multiset, clear_cache
from smallgraphlib.printers.tikz import TikzPrinter


class Traversal(Enum):
    PREORDER = 0
    POSTORDER = 1
    INORDER = 2


class NodeAlreadyFoundError(RuntimeError):
    pass


class AbstractGraph(ABC, Generic[Node]):
    printer = TikzPrinter

    def __init__(
        self,
        nodes: Iterable[Node],
        *edges: EdgeLike,
        sort_nodes: bool = True,
    ):
        """Create a graph of nodes, when nodes may be any hashable objects.

        Edges may be any iterable of two nodes.

        If the graph is directed and the edge is a set {A, B} of two nodes,
        then two edges are added, one from A to B and one from B to A.
        Note that in that case, adding {A} will result in two edges too.
        """
        self._successors: dict[Node, CounterType[Node]] = {}
        if self.is_directed:
            self._predecessors: dict[Node, CounterType[Node]] = {}
        else:
            self._predecessors = self._successors
        # Nodes must be added before edges.
        if sort_nodes:
            nodes = sorted(nodes)
        self.add_nodes(*nodes)
        self.add_edges(*edges)

    # ------------------
    # Other constructors
    # ------------------

    @classmethod
    def from_dict(
        cls: Type[_AbstractGraph], d: dict[Node, Iterable[Node]] = None, /, **successors: Iterable[Node]
    ) -> _AbstractGraph:
        """Generate a graph from a dictionary of nodes, each node being associated with its successors.

        For example `DirectedGraph.from_dict({1: [2, 3], 2: [], 3: [1]})` will generate a graph of
        3 nodes, 1, 2 and 3, with edges 1->2, 1->3 and 3->1.

            >>> from smallgraphlib import DirectedGraph
            >>> DirectedGraph.from_dict({1: [2, 3], 2: [], 3: [1]})
            DirectedGraph((1, 2, 3), (1, 2), (1, 3), (3, 1))
            >>> DirectedGraph.from_dict(A="BC", B="", C="A")
            DirectedGraph(('A', 'B', 'C'), ('A', 'B'), ('A', 'C'), ('C', 'A'))
        """
        if d is None:
            d = {}
        d.update(successors)  # type: ignore
        nodes: set[Node] = set()
        edges: list[tuple[Node, Node]] = []
        for start, ends in d.items():
            nodes.add(start)
            nodes.update(ends)
            edges.extend((start, end) for end in ends)
        return cls(nodes, *edges)

    @classmethod
    def from_string(cls: Type[_AbstractGraph], string: str) -> _AbstractGraph:
        """DirectedGraph.from_string("A:B,C B:C C") will generate a graph of 3 nodes, A, B and C, with
        edges A->B, A->C and B->C."""
        nodes: list[str] = []
        edges: list[tuple[str, str]] = []
        for substring in string.split():
            node, *successors = substring.split(":", 1)
            nodes.append(node.strip())
            if successors:
                edges.extend((node, successor.strip()) for successor in successors[0].split(","))
        return cls(nodes, *edges)

    @staticmethod
    def _matrix_as_tuple_of_tuples(matrix: Iterable[Iterable]) -> tuple[Tuple, ...]:
        if hasattr(matrix, "tolist"):  # for numpy or sympy
            matrix = matrix.tolist()
        tuple_matrix = tuple(tuple(iterable) for iterable in matrix)
        # Test if M is correct
        n = len(tuple_matrix)
        if any(len(line) != n for line in tuple_matrix):
            raise ValueError("All matrix lines must be the same length.")
        return tuple_matrix

    @staticmethod
    @abstractmethod
    def _get_edges_from_adjacency_matrix(
        matrix: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        ...

    @classmethod
    def from_matrix(
        cls: Type[_AbstractGraph], matrix: Iterable[Iterable[int]], nodes_names: Iterable[Node] = None
    ) -> _AbstractGraph:
        """Construct the graph corresponding to the given adjacency matrix.

        Matrix must be a matrix of positive integers
        (`int` or any integer type inheriting from `numbers.Integral`).
        """
        # Convert iterable to matrix.
        tuple_matrix = cls._matrix_as_tuple_of_tuples(matrix)
        # Test if M is correct
        n = len(tuple_matrix)
        for line in tuple_matrix:
            for val in line:
                if not (isinstance(val, Integral) and int(val) >= 0):
                    raise ValueError(f"All matrix values must be positive integers, but {val!r} is not.")

        edges = cls._get_edges_from_adjacency_matrix(tuple_matrix)

        g = cls(range(1, n + 1), *edges)
        if nodes_names:
            g.rename_nodes(dict(enumerate(list(nodes_names)[: len(g.nodes)], start=1)))
        return g

    # ----------------
    #   Node methods
    # ================

    # Nodes must be ordered, to generate the matrix, so do *not* return a set.
    @cached_property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._successors)

    @cached_property
    def nodes_set(self) -> FrozenSet[Node]:
        return frozenset(self.nodes)

    def _add_node(self, node: Node) -> None:
        self._successors[node] = Multiset()
        self._predecessors[node] = Multiset()

    @clear_cache
    def add_nodes(self, *new_nodes: Node) -> None:
        for node in new_nodes:
            if node in self.nodes:
                raise RuntimeError(f"Node already present: {node!r}")
            self._add_node(node)

    @clear_cache
    def remove_nodes(self, *nodes: Node) -> None:
        for node in nodes:
            if node not in self.nodes:
                raise ValueError(f"Node {node} not found!")
        for node in nodes:
            node_successors = self._successors.pop(node, ())
            for successor in node_successors:
                self._predecessors.get(successor, {}).pop(node, None)  # type: ignore
            predecessors = self._predecessors.pop(node, ())
            for predecessor in predecessors:
                self._successors.get(predecessor, {}).pop(node, None)  # type: ignore

    @clear_cache
    def rename_node(self, old_name: Node, new_name: Node) -> None:
        """Rename node. New name must not be already present, else a `NameError` will be raised."""
        if new_name in self.nodes:
            raise NameError(f"Conflicting names: this graph already has a node named {new_name!r}")
        for dictionary in (self._successors, self._predecessors):
            for node, counter in list(dictionary.items()):  # make a copy, since we modify the dictionary.
                if node == old_name:
                    dictionary[new_name] = dictionary.pop(old_name)
                if old_name in counter:
                    counter[new_name] = counter.pop(old_name)

    @clear_cache
    def rename_nodes(self, node_names: dict[Node, Node]) -> None:
        if not node_names:
            return
        # First, we must assure the atomicity of the renaming operation:
        # it should not fail half-way if a node is not found, or if two renaming rules will conflict...
        # This is an incorrect renaming for example:
        # {A -> E, B -> E}
        # ({A -> E, A -> F} would be incorrect too, but can't occur with dict.)
        # Eventually, the following operation is correct if and only if A is an existing node and C is not:
        # {A -> C}
        # Note that, if A and C are both already existing nodes, this is correct:
        # {A -> C, C -> A}
        # but this alone is not:
        # {A -> C}
        # Considering a V1 -> V2 renaming operation, let's call V1 the **start node** and V2 the **end node**.
        # We'll use the following algorithm to detect those incorrect renaming operations.
        # The renaming operation is correct if and only if:
        #    1. All the **start** nodes exist.
        #    2. All the **end** nodes appear only once, and don't conflict with any existing unchanged nodes.
        unknown_nodes = set(node_names) - self.nodes_set
        if unknown_nodes:
            raise ValueError(f"Nodes not found: {unknown_nodes}")
        new_names = Counter(node_names.values())
        most_common, count = new_names.most_common(1)[0]
        if count > 1:
            raise ValueError(
                f"Conflict in renaming: trying to rename {count} different nodes to {most_common} !"
            )
        unchanged_nodes = self.nodes_set - set(node_names)
        conflicts = unchanged_nodes & set(new_names)
        if conflicts:
            raise ValueError(
                f"Names conflict in renaming, when trying to apply the following names {conflicts}."
            )

        # The renaming operation must look simultaneous:
        # for example, we must avoid that applying {A -> B, B -> C} will result in A −> C !
        # The solution is to use temporary names when renaming.
        # Something like {A -> tmp_B, B -> tmp_C} and then {tmp_B -> B, tmp_C -> C}.
        remaining_translation: list[Node] = []
        # Use list() to make a copy of the dictionary keys and values, since we modify the dictionary.
        for i, (old_name, new_name) in enumerate(list(node_names.items())):
            self.rename_node(old_name, "\00" + str(i))
            remaining_translation.append(new_name)
        for i, new_name in enumerate(remaining_translation):
            self.rename_node("\00" + str(i), remaining_translation[i])

    def shuffle_nodes(self):
        """Shuffle nodes.

        >>> from smallgraphlib import random_graph
        >>> g = random_graph(4, 7)
        >>> g2 = g.copy()
        >>> g2.shuffle_nodes()
        >>> g == g2  # doctest: +SKIP
        ...  # probably False
        >>> g.is_isomorphic_to(g2)
        True
        """
        # Sort nodes before shuffling, to make shuffling deterministic with a given random.seed.
        # nodes = sorted(self.nodes)
        nodes = list(self.nodes)
        random.shuffle(nodes)
        self.rename_nodes(dict((old_name, new_name) for old_name, new_name in zip(self.nodes, nodes)))

    # ----------------
    #   Edge methods
    # ================

    @cached_property
    def edges(self) -> tuple[Edge, ...]:
        edges_count: CounterType[Edge] = Multiset()
        for node in self.nodes:
            for successor in self.successors(node):
                edge: Edge = self._edge(node, successor)
                edges_count[edge] += self.count_edges(node, successor)
        if not self.is_directed:
            for key in edges_count:
                edges_count[key] //= 2
        return tuple(edges_count.elements())

    @cached_property
    def edges_set(self) -> FrozenSet[Edge]:
        return frozenset(self.edges)

    @staticmethod
    def _get_edge_extremities(edge: EdgeLike) -> tuple[Node, Node]:
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
        return start, end

    def _test_edges(self, *edges: EdgeLike) -> None:
        edges_nodes = set(chain(*edges))
        nodes = set(self.nodes)
        if not edges_nodes.issubset(nodes):
            raise ValueError(f"Nodes not found: {edges_nodes - nodes}")

    @clear_cache
    def add_edges(self, *new_edges: EdgeLike) -> None:
        self._test_edges(*new_edges)
        for edge in new_edges:
            start, end = self._get_edge_extremities(edge)
            self._successors[start][end] += 1
            self._predecessors[end][start] += 1
            if self.is_directed and isinstance(edge, (set, frozenset)):
                # This is a bidirectional edge.
                # Note that bidirectional loops are added twice too.
                self._successors[end][start] += 1
                self._predecessors[start][end] += 1

    @clear_cache
    def remove_edges(self, *edges: EdgeLike, ignore_missing=False) -> None:
        self._test_edges(*edges)

        for edge in edges:
            start, end = self._get_edge_extremities(edge)
            try:
                self._successors[start][end] -= 1
                self._predecessors[end][start] -= 1
            except ValueError:
                if not ignore_missing:
                    raise ValueError(f"Edge not found: ({start}, {end}).")
            # if self.is_directed:
            #     self._successors[end][start] -= 1

    @abstractmethod
    def simplify(self, remove_loops: bool = True) -> Self:
        ...

    # ----------
    # Comparison
    # ----------

    def __eq__(self, other: Any) -> bool:
        # Attention, there may be multiple edges, so don't use sets to compare edges !
        return (
            type(self) is type(other)
            and self.nodes_set == other.nodes_set
            and Counter(self.edges) == Counter(other.edges)
        )

    @abstractmethod
    def is_isomorphic_to(self, other) -> bool:
        if not isinstance(other, AbstractGraph):
            return False

        if self.order == 0:
            return other.order == 0

        def count_in_and_out_degrees(
            graph: AbstractGraph,
        ) -> CounterType[tuple[int, int]]:
            return Counter((graph.in_out_degree(node_) for node_ in graph.nodes))

        if count_in_and_out_degrees(self) != count_in_and_out_degrees(other):
            return False
        assert self.order == other.order and self.degree == other.degree

        degrees_to_nodes_for_other_graph: dict[tuple[int, int], list[Node]] = {}
        for node in other.nodes:
            degrees_to_nodes_for_other_graph.setdefault(other.in_out_degree(node), []).append(node)
        nodes_to_degrees_for_self = {node: self.in_out_degree(node) for node in self.nodes}

        remaining_self_nodes = set(self.nodes)
        remaining_other_nodes = set(other.nodes)
        used_nodes: list[Node] = [remaining_self_nodes.pop()]
        corresponding_nodes_possibilities: dict[Node, Optional[list[Node]]] = {}
        reversed_mapping: dict[Node, Node] = {}
        order = self.order

        adjacency_test_methods = [(self.successors, other.successors)]
        # If graph is undirected, predecessors are also successors, so no need to test twice.
        if self.is_directed:
            adjacency_test_methods.append((self.predecessors, other.predecessors))

        def test_possibility(candidate: Node) -> bool:
            """Test if this possibility matches with already detected corresponding nodes."""
            for self_method, other_method in adjacency_test_methods:
                for adjacent_node in self_method(node):
                    try:
                        _possibilities = corresponding_nodes_possibilities[adjacent_node]
                        assert _possibilities is not None and len(_possibilities) > 0
                        corresponding_node = _possibilities[0]
                        if corresponding_node not in other_method(candidate):
                            return False
                    except KeyError:
                        pass
                for adjacent_node in other_method(candidate):
                    try:
                        corresponding_node = reversed_mapping[adjacent_node]
                        if corresponding_node not in self_method(node):
                            return False
                    except KeyError:
                        pass
            return True

        while len(used_nodes) > 0:
            node = used_nodes[-1]
            possibilities = corresponding_nodes_possibilities.get(node)
            if possibilities is None:  # First time we test this node
                # Test for any possibilities to go further in this direction.
                possibilities = []
                for possibility in degrees_to_nodes_for_other_graph[nodes_to_degrees_for_self[node]]:
                    if possibility in remaining_other_nodes and test_possibility(possibility):
                        possibilities.append(possibility)
                corresponding_nodes_possibilities[node] = possibilities

            if len(possibilities) == 0:
                # This branch of possibilities failed, so go back to previous step.
                used_nodes.pop()
                corresponding_nodes_possibilities.pop(node)
                remaining_self_nodes.add(node)
                if used_nodes:
                    # Remove other graph node from possibilities, as this branch of possibilities failed.
                    previous_node = used_nodes[-1]
                    previous_possibilities = corresponding_nodes_possibilities[previous_node]
                    assert previous_possibilities is not None and len(previous_possibilities) > 0
                    registered_corresponding_node = previous_possibilities.pop(0)
                    remaining_other_nodes.add(registered_corresponding_node)
                    del reversed_mapping[registered_corresponding_node]
            else:
                # For now, assume that first possibility is correct.
                remaining_other_nodes.remove(possibilities[0])
                reversed_mapping[possibilities[0]] = node
                # Select another node to go ahead in matching process.
                # We'll choose a random node in currently unused nodes.
                # TODO: improve selection process ? It may be better to select an adjacent node, if any ?
                try:
                    used_nodes.append(remaining_self_nodes.pop())
                except KeyError:  # No remaining node
                    assert len(corresponding_nodes_possibilities) == order
                    assert len(reversed_mapping) == order
                    assert len(remaining_self_nodes) == 0
                    assert len(remaining_other_nodes) == 0
                    # Test for correctness
                    copy = self.copy()
                    mapping = {
                        node: possibilities[0]  # type: ignore
                        for node, possibilities in corresponding_nodes_possibilities.items()
                    }
                    copy.rename_nodes(mapping)
                    assert copy == other
                    return True
        assert len(corresponding_nodes_possibilities) == 0
        assert len(reversed_mapping) == 0
        assert len(remaining_self_nodes) == order
        assert len(remaining_other_nodes) == order
        return False

    # def _match(self, other, dict, other, match:dict=None):
    #     for

    # --------------
    # Classification
    # --------------

    @property
    def order(self) -> int:
        return len(self.nodes)

    @property
    def degree(self) -> int:
        return sum(self.node_degree(node) for node in self.nodes) // 2

    @property
    @abstractmethod
    def is_directed(self) -> bool:
        ...

    @cached_property
    def has_loop(self) -> bool:
        return any(node in self.successors(node) for node in self.nodes)

    @cached_property
    def has_multiple_edges(self) -> bool:
        return len(self.edges) != len(self.edges_set)

    @cached_property
    def is_simple(self) -> bool:
        return not (self.has_loop or self.has_multiple_edges)

    @cached_property
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
            border |= self.successors(node) - connected_nodes
        return len(connected_nodes) == self.order

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    def count_edges(self, node1: Node, node2: Node, count_undirected_loops_twice=True) -> int:
        """Count the number of edges from node1 to node2.

        Warning: by default, undirected loops are counted twice."""
        n = self._successors[node1][node2]
        if not count_undirected_loops_twice and node1 == node2 and not self.is_directed:
            n //= 2
        return n

    def weight(self, node1: Node, node2: Node) -> float:
        if node1 == node2:
            return 0
        elif node2 in self.successors(node1):
            return 1
        else:
            return math.inf

    @cached_property
    def weight_matrix(self):
        return tuple(tuple(self.weight(node1, node2) for node2 in self.nodes) for node1 in self.nodes)

    def labels(self, node1: Node, node2: Node) -> list[str]:
        n = self.count_edges(node1, node2, count_undirected_loops_twice=False)
        return n * [""]

    def node_degree(self, node: Node) -> int:
        if self.is_directed:
            return self.in_degree(node) + self.out_degree(node)
        return self.out_degree(node)

    @property
    def all_degrees(self) -> dict[Node, int]:
        return {node: self.node_degree(node) for node in self.nodes}

    @property
    def all_in_degrees(self) -> dict[Node, int]:
        return {node: self.in_degree(node) for node in self.nodes}

    @property
    def all_out_degrees(self) -> dict[Node, int]:
        return {node: self.out_degree(node) for node in self.nodes}

    def in_degree(self, node: Node) -> int:
        return sum(self._predecessors[node].values())

    def out_degree(self, node: Node) -> int:
        return sum(self._successors[node].values())

    def in_out_degree(self, node: Node) -> tuple[int, int]:
        return self.in_degree(node), self.out_degree(node)

    def successors(self, node: Node) -> Set[Node]:
        return set(self._successors[node])

    def predecessors(self, node: Node) -> Set[Node]:
        return set(self._predecessors[node])

    def copy(self):
        return self.__class__(self.nodes, *self.edges)

    @cached_property
    def adjacency_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Get the adjacency matrix of the graph.

        If operations on the matrix are needed, the matrix should be converted to a `numpy` or `sympy` matrix:

            >>> import numpy
            >>> from smallgraphlib import complete_graph
            >>> raw_matrix = complete_graph(3).adjacency_matrix
            >>> raw_matrix * raw_matrix
            Traceback (most recent call last):
            TypeError: can't multiply sequence by non-int of type 'tuple'
            >>> M = numpy.array(raw_matrix)
            >>> M @ M  # `@` is the operator corresponding to matrix multiplication
            array([[2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2]])

        Return:
            A matrix of integers, as a tuple of tuples.
        """
        return tuple(tuple(self.count_edges(start, end) for end in self.nodes) for start in self.nodes)

    @abstractmethod
    def are_adjacents(self, node1: Node, node2: Node) -> bool:
        ...

    @property
    @abstractmethod
    def is_eulerian(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_semi_eulerian(self) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def _edge(node1: Node, node2: Node = None) -> Edge:
        ...

    def _dijkstra(self, start: Node, end: Node = None) -> tuple[dict[Node, float], dict[Node, list[Node]]]:
        """Implementation of Dijkstra Algorithm."""
        if start not in self.nodes:
            raise ValueError(f"Unknown node {start!r}.")
        if end is not None and end not in self.nodes:
            raise ValueError(f"Unknown node {end!r}.")
        lengths: dict[Node, float] = {node: (0 if node == start else math.inf) for node in self.nodes}
        last_step: dict[Node, list[Node]] = {node: [] for node in self.nodes}
        never_selected_nodes = set(self.nodes)
        selected_node = start
        while selected_node != end and len(never_selected_nodes) > 1:
            never_selected_nodes.remove(selected_node)
            for successor in self.successors(selected_node) & never_selected_nodes:
                weight = self.weight(selected_node, successor)
                if weight < 0:
                    raise ValueError("Can't find shortest paths with negative weights.")
                new_length = lengths[selected_node] + weight
                if new_length < lengths[successor]:
                    lengths[successor] = new_length
                    last_step[successor] = [selected_node]
                elif new_length == lengths[successor]:
                    last_step[successor].append(selected_node)
            selected_node = min(never_selected_nodes, key=(lambda node_: lengths[node_]))
        return lengths, last_step

    # TODO: put in cache all distances from `start` node when running Dijkstra algorithm.
    #
    def distance(self, start: Node, end: Node) -> float:
        """Implementation of Dijkstra Algorithm."""
        lengths, _ = self._dijkstra(start, end)
        return lengths[end]

    # @cached_property
    # def distance_matrix(self):
    #     return tuple(tuple(self.distance(node1, node2) for node2 in self.nodes) for node1 in self.nodes)

    @cached_property
    def distance_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Compute the distance matrix, using Roy-Floyd-Warshall algorithm."""
        matrix = [list(row) for row in self.weight_matrix]
        n = self.order
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # Try to use node k to find a shorter path.
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
        # Making result immutable is much safer for user, since we use caching.
        return tuple(tuple(row) for row in matrix)

    @cached_property
    def diameter(self) -> float:
        return max(self.distance(node1, node2) for node1 in self.nodes for node2 in self.nodes)

    def shortest_paths(self, start: Node, end: Node) -> tuple[float, list[list[Node]]]:
        """Implementation of Dijkstra Algorithm."""
        lengths, last_step = self._dijkstra(start, end)

        def generate_paths(path: list[Node]) -> list[list[Node]]:
            if path[0] == start:
                return [path]
            paths = []
            for predecessor in last_step[path[0]]:
                paths.extend(generate_paths([predecessor] + path))
            return paths

        return lengths[end], generate_paths([end])

    def _iterative_depth_first_search(
        self,
        start: Node = None,
        *,
        error_if_already_visited=False,
    ) -> Iterator[Node]:
        """Return a node iterator, using a prefix DFS.

        If `error_if_already_visited` is set to True, any cycle will raise a `NodeAlreadyFoundError` when detected.
        """
        if start is None:
            start = self.nodes[0]
        stack: list[Node] = [start]
        previous_nodes: list[Optional[Node]] = [None]
        # If the graph isn't a rooted tree, there is no notion of parent.
        # If we want to use DFS to test if the graph is a tree, we must try to visit all adjacents nodes
        # except the one we come from, and see if there were already visited.
        # So we must keep track of the node we come from for each node of the stack.
        visited: Set[Node] = set()
        while stack:
            node = stack.pop()
            previous = previous_nodes.pop()
            if node not in visited:
                visited.add(node)
                yield node
                for successor in reversed(self._successors[node]):
                    # Don't go backward, especially if `error_if_already_visited` is True !
                    if successor != previous:
                        stack.append(successor)
                        previous_nodes.append(node)
            elif error_if_already_visited:
                raise NodeAlreadyFoundError

    def depth_first_search(
        self, start: Node = None, *, order: Traversal = Traversal.PREORDER
    ) -> Iterator[Node]:
        """Recursive implementation of DFS (Depth First Search)."""
        if start is None:
            start = self.nodes[0]
        visited: Set[Node] = set()
        if not isinstance(order, Traversal):
            raise NotImplementedError(
                f"Order must be Traversal.PREORDER, Traversal.POSTORDER or Traversal.INORDER, not {order!r}."
            )

        def preorder_dfs(node: Node) -> Iterator[Node]:
            if node not in visited:
                yield node
                visited.add(node)
                for successor in self._successors[node]:
                    yield from preorder_dfs(successor)

        def postorder_dfs(node: Node) -> Iterator[Node]:
            if node not in visited:
                visited.add(node)
                for successor in self._successors[node]:
                    yield from postorder_dfs(successor)
                yield node

        def inorder_dfs(node: Node) -> Iterator[Node]:
            visited.add(node)
            # Eliminate visited nodes from the list of successors *before* splitting it.
            successors: list[Node] = [
                successor for successor in self._successors[node] if successor not in visited
            ]
            for successor in successors[:1]:
                yield from inorder_dfs(successor)
            yield node
            for successor in successors[1:]:
                yield from inorder_dfs(successor)

        traversals = {
            Traversal.PREORDER: preorder_dfs,
            Traversal.POSTORDER: postorder_dfs,
            Traversal.INORDER: inorder_dfs,
        }
        return traversals[order](start)

    @cached_property
    def is_acyclic(self) -> bool:
        """Use DFS to test if the graph contains a cycle."""
        try:
            for _ in self._iterative_depth_first_search(error_if_already_visited=True):
                pass
        except NodeAlreadyFoundError:
            return False
        return True

    @cached_property
    def is_a_tree(self) -> bool:
        if self.order != self.degree + 1:
            return False
        try:
            # All nodes must be visited once.
            return self.nodes_set == set(self._iterative_depth_first_search(error_if_already_visited=True))
        except NodeAlreadyFoundError:
            return False

    def breadth_first_search(self, start: Node = None) -> Iterator[Node]:
        if start is None:
            start = self.nodes[0]
        queue = deque([start])
        visited: Set[Node] = set()
        while queue:
            node = queue.popleft()
            visited.add(node)
            yield node
            queue.extend(successor for successor in self._successors[node] if successor not in visited)

    def find_path(
        self,
        start: Node,
        end: Node,
        _filter_edges: Callable[[Self, Node, Node, dict[Node, Node]], bool] | None = None,
        _forbidden_nodes: Callable[[Self, Node], bool] | Iterable[Node] = (),
    ) -> list[Node]:
        """Return a path between nodes `start` and `end`, if any, or an empty list.

        A BFS algorithm is used, so the path is as short as possible.

        If defined, `_filter_edges` is a boolean function, which takes 4 arguments: the calling class,
        the current node, its successor and a dictionary {previous node: next node}
        describing the current state of the path.
        If `_filter_edges(self, node1, node2, previous_nodes)` returns `False`, the edge (node1, node2), if it exists,
        will be ignored.
        For convenience, a `_forbidden_nodes` argument is also provided, which may be either a list of forbidden nodes,
        or a function to generate such a list (if it returns `True`, the node will be removed).

        Note:
          - if `_forbidden_nodes` is a callable, it must return `True` to discard a node, contrary to `_filter_edges`
          which must return `False` to discard an edge.
          - if callable, `_forbidden_nodes` is only called once, while `_filter_edges` will be called everytime
            a new edge is tested.
        """
        previous: dict[Node, Node] = {}
        queue: deque[Node] = deque([start])
        if callable(_forbidden_nodes):
            _filter_nodes = set(node for node in self.nodes if _forbidden_nodes(self, node))
        else:
            _filter_nodes = set(_forbidden_nodes)
        while end not in previous and len(queue) > 0:
            # BFS
            node = queue.popleft()
            for successor in self.successors(node):
                if (
                    _filter_edges is None or _filter_edges(self, node, successor, previous)
                ) and node not in _filter_nodes:
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

    @cached_property
    def is_hamiltonian(self) -> bool:
        """Test whether the graph is hamiltonian."""
        if any(degree <= 1 for degree in self.all_degrees.values()):
            return False
        order = self.order
        if order >= 3 and all(2 * degree >= order for degree in self.all_degrees.values()):
            return True
        # Start from any node.
        start = next(iter(self.nodes))

        def find_n_steps(
            first: Node, last: Node, n: int, _already_used: frozenset[Node] = frozenset()
        ) -> bool:
            """Recursively test whether a path of n steps exists from node first to node last,
            ensuring that each node is used at most once."""
            assert n >= 1
            _already_used |= frozenset((first,))
            if n == 1:
                return last in self._successors[first]
            return any(
                find_n_steps(successor, last, n - 1, _already_used=_already_used)
                for successor in self._successors[first]
                if successor not in _already_used
            )

        return find_n_steps(start, start, order)

    @cached_property
    def is_semi_hamiltonian(self) -> bool:
        """Test whether the graph is semi-hamiltonian."""

        def find_n_steps(first: Node, n: int, _already_used: frozenset[Node] = frozenset()) -> bool:
            """Recursively test whether a path of n steps starting from node `first` exists,
            ensuring that each node is used at most once."""
            assert n >= 0
            _already_used |= frozenset((first,))
            if n == 0:
                return True
            return any(
                find_n_steps(successor, n - 1, _already_used=_already_used)
                for successor in self._successors[first]
                if successor not in _already_used
            )

        return not self.is_hamiltonian and any(find_n_steps(node, self.order - 1) for node in self.nodes)

    # ----------------------------
    #    Tikz and LaTeX export
    # ============================

    def as_tikz(self, *, shuffle_nodes=False, border: str = None, options="", preamble=False) -> str:
        r"""Generate tikz code corresponding to this graph.

        `Tikz` package must be loaded in the latex preamble, with `arrows.meta` library::

            \usepackage{tikz}
            \usetikzlibrary{arrows.meta}

        For labeled graphs, it is recommended to load `contour` package too::

            \usepackage[outline]{contour}
            \contourlength{0.5pt}

        If set, `border` have to be a combination of tikz path drawing styles,
        like "dotted", or "dashed,blue".
        """
        return self.printer(self, shuffle_nodes=shuffle_nodes).tikz_code(
            border=border, options=options, preamble=preamble
        )

    def latex_adjacency_matrix(self, env: str = "pmatrix") -> str:
        return latex_matrix(self.adjacency_matrix, env=env)

    def latex_distance_matrix(self, env: str = "pmatrix") -> str:
        return latex_matrix(self.distance_matrix, env=env)

    def latex_degrees(self) -> str:
        return latex_degrees_table(self)

    def latex_Dijkstra(self, start: Node | None = None, end: Node | None = None) -> str:
        if start is None:
            start = self.nodes[0]
        return latex_Dijkstra(self, start=start, end=end)


class InvalidGraphAttribute(AttributeError):
    """This error is raised when a method or property can't return a value for a particular graph."""
