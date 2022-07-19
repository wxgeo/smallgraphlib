import math
import random
from abc import ABC, abstractmethod
from collections import Counter, deque
from enum import Enum
from itertools import chain
from numbers import Integral
from typing import (
    TypeVar,
    Tuple,
    FrozenSet,
    Union,
    Set,
    Iterable,
    Any,
    Dict,
    List,
    Generic,
    Counter as CounterType,
    Optional,
    Iterator,
    Sequence,
)

from smallgraphlib.utilities import (
    ComparableAndHashable,
    cached_property,
    Multiset,
    clear_cache,
    segments_intersection,
)

_TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT = 3
_TIKZ_EXPORT_MAX_MULTIPLE_LOOPS_SUPPORT = 1

# Node = TypeVar("Node", bound=typing.Hashable)  # too subtile for Pycharm ? ;-(
Node = TypeVar("Node", bound=ComparableAndHashable)
DirectedEdge = Tuple[Node, Node]
UndirectedEdge = FrozenSet[Node]
Edge = Union[DirectedEdge, UndirectedEdge]
EdgeLike = Union[Edge, Set[Node], Iterable[Node]]
Label = Any
InternalGraphRepresentation = Dict[Node, Dict[Node, Union[int, List[Label]]]]


class Traversal(Enum):
    PREORDER = 0
    POSTORDER = 1
    INORDER = 2


class NodeAlreadyFoundError(RuntimeError):
    pass


class AbstractGraph(ABC, Generic[Node]):
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
        self._successors: Dict[Node, CounterType[Node]] = {}
        if self.is_directed:
            self._predecessors: Dict[Node, CounterType[Node]] = {}
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
    def from_string(cls, string: str):
        """DirectedGraph.from_string("A:B,C B:C C") will generate a graph of 3 nodes, A, B and C, with
        edges A->B, A->C and B->C."""
        nodes: List[str] = []
        edges: List[Tuple[str, str]] = []
        for substring in string.split():
            node, *remaining = substring.split(":", 1)
            nodes.append(node.strip())
            if remaining:
                edges.extend((node, successor.strip()) for successor in remaining[0].split(","))
        return cls(nodes, *edges)  # type: ignore

    @staticmethod
    def _matrix_as_tuple_of_tuples(matrix: Iterable[Iterable]) -> Tuple[Tuple, ...]:
        if hasattr(matrix, "tolist"):  # for numpy or sympy
            matrix = matrix.tolist()  # type: ignore
        M = tuple(tuple(iterable) for iterable in matrix)
        # Test if M is correct
        n = len(M)
        if any(len(line) != n for line in M):
            raise ValueError("All matrix lines must be the same length.")
        return M

    @staticmethod
    @abstractmethod
    def _get_edges_from_adjacency_matrix(
        matrix: Sequence[Sequence[int]],
    ) -> List[Tuple[int, int]]:
        ...

    @classmethod
    def from_matrix(
        cls, matrix: Iterable[Iterable[int]], nodes_names: Iterable[Node] = None
    ) -> "AbstractGraph":
        """Construct the graph corresponding to the given adjacency matrix.

        Matrix must be a matrix of positive integers
        (`int` or any integer type inheriting from `numbers.Integral`).
        """
        # Convert iterable to matrix.
        M = cls._matrix_as_tuple_of_tuples(matrix)
        # Test if M is correct
        n = len(M)
        for line in M:
            for val in line:
                if not (isinstance(val, Integral) and int(val) >= 0):
                    raise ValueError(f"All matrix values must be positive integers, but {val!r} is not.")

        edges = cls._get_edges_from_adjacency_matrix(M)

        g = cls(range(1, n + 1), *edges)  # type: ignore
        if nodes_names:
            g.rename_nodes(dict(enumerate(list(nodes_names)[: len(g.nodes)], start=1)))
        return g

    # ------------
    # Node methods
    # ------------

    # Nodes must be ordered, to generate the matrix, so do *not* return a set.
    @cached_property
    def nodes(self) -> Tuple[Node, ...]:
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
    def rename_nodes(self, node_names: Dict[Node, Node]) -> None:
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
        remaining_translation: List[Node] = []
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
        >>> g == g2
        ...  # probably False
        >>> g.is_isomorphic_to(g2)
        True
        """
        nodes = list(self.nodes)
        random.shuffle(nodes)
        self.rename_nodes(dict((old_name, new_name) for old_name, new_name in zip(self.nodes, nodes)))

    # ------------
    # Edge methods
    # ------------

    @cached_property
    def edges(self) -> Tuple[Edge, ...]:
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
    def _get_edge_extremities(edge: EdgeLike) -> Tuple[Node, Node]:
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
    def remove_edges(self, *edges: EdgeLike) -> None:
        self._test_edges(*edges)

        for edge in edges:
            start, end = self._get_edge_extremities(edge)
            self._successors[start][end] -= 1
            self._predecessors[end][start] -= 1
            # if self.is_directed:
            #     self._successors[end][start] -= 1

    # ----------
    # Comparison
    # ----------

    def __eq__(self, other: Any):
        # Attention, there may be multiple edges, so don't use sets to compare edges !
        return (
            type(self) == type(other)
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
        ) -> CounterType[Tuple[int, int]]:
            return Counter((graph.in_out_degree(node_) for node_ in graph.nodes))

        if count_in_and_out_degrees(self) != count_in_and_out_degrees(other):
            return False
        assert self.order == other.order and self.degree == other.degree

        degrees_to_nodes_for_other_graph: Dict[Tuple[int, int], List[Node]] = {}
        for node in other.nodes:
            degrees_to_nodes_for_other_graph.setdefault(other.in_out_degree(node), []).append(node)
        nodes_to_degrees_for_self = {node: self.in_out_degree(node) for node in self.nodes}

        remaining_self_nodes = set(self.nodes)
        remaining_other_nodes = set(other.nodes)
        used_nodes: List[Node] = [remaining_self_nodes.pop()]
        corresponding_nodes_possibilities: Dict[Node, Optional[List[Node]]] = {}
        reversed_mapping: Dict[Node, Node] = {}
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
        elif self.are_adjacents(node1, node2):
            return 1
        else:
            return math.inf

    @cached_property
    def weight_matrix(self):
        return tuple(tuple(self.weight(node1, node2) for node2 in self.nodes) for node1 in self.nodes)

    def labels(self, node1: Node, node2: Node) -> List[str]:
        n = self.count_edges(node1, node2, count_undirected_loops_twice=False)
        return n * [""]

    def node_degree(self, node: Node) -> int:
        if self.is_directed:
            return self.in_degree(node) + self.out_degree(node)
        return self.out_degree(node)

    @property
    def all_degrees(self) -> Dict[Node, int]:
        return {node: self.node_degree(node) for node in self.nodes}

    @property
    def all_in_degrees(self) -> Dict[Node, int]:
        return {node: self.in_degree(node) for node in self.nodes}

    @property
    def all_out_degrees(self) -> Dict[Node, int]:
        return {node: self.out_degree(node) for node in self.nodes}

    def in_degree(self, node: Node) -> int:
        return sum(self._predecessors[node].values())

    def out_degree(self, node: Node) -> int:
        return sum(self._successors[node].values())

    def in_out_degree(self, node: Node) -> Tuple[int, int]:
        return self.in_degree(node), self.out_degree(node)

    def successors(self, node: Node) -> Set[Node]:
        return set(self._successors[node])

    def predecessors(self, node: Node) -> Set[Node]:
        return set(self._predecessors[node])

    def copy(self):
        return self.__class__(self.nodes, *self.edges)

    @cached_property
    def adjacency_matrix(self) -> Tuple[Tuple[int, ...], ...]:
        """Get the adjacency matrix of the graph.

        If operations on the matrix are needed, the matrix should be converted to a `numpy` or `sympy` matrix:

            >>> import numpy  # doctest: +SKIP
            >>> from smallgraphlib import complete_graph
            >>> M = numpy.matrix(complete_graph(3).adjacency_matrix)
            >>> M**2
            matrix([[2, 1, 1],
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

    def _dijkstra(self, start: Node, end: Node = None) -> Tuple[Dict[Node, float], Dict[Node, List[Node]]]:
        """Implementation of Dijkstra Algorithm."""
        if start not in self.nodes:
            raise ValueError(f"Unknown node {start!r}.")
        if end is not None and end not in self.nodes:
            raise ValueError(f"Unknown node {end!r}.")
        lengths: Dict[Node, float] = {node: (0 if node == start else math.inf) for node in self.nodes}
        last_step: Dict[Node, List[Node]] = {node: [] for node in self.nodes}
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

    @cached_property
    def distance_matrix(self):
        return tuple(tuple(self.distance(node1, node2) for node2 in self.nodes) for node1 in self.nodes)

    @cached_property
    def diameter(self) -> float:
        return max(self.distance(node1, node2) for node1 in self.nodes for node2 in self.nodes)

    def shortest_paths(self, start: Node, end: Node) -> Tuple[float, List[List[Node]]]:
        """Implementation of Dijkstra Algorithm."""
        lengths, last_step = self._dijkstra(start, end)

        def generate_paths(path: List[Node]) -> List[List[Node]]:
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
        """Return a node iterator, using a prefix DFS."""
        if start is None:
            start = self.nodes[0]
        stack: List[Node] = [start]
        previous_nodes: List[Optional[Node]] = [None]
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

        def preorder_dfs(node):
            if node not in visited:
                yield node
                visited.add(node)
                for successor in self._successors[node]:
                    yield from preorder_dfs(successor)

        def postorder_dfs(node):
            if node not in visited:
                visited.add(node)
                for successor in self._successors[node]:
                    yield from postorder_dfs(successor)
                yield node

        def inorder_dfs(node):
            visited.add(node)
            # Eliminate visited nodes from the list of successors *before* splitting it.
            successors: List[Node] = [
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

    def as_tikz(self, *, shuffle_nodes=False, options="") -> str:
        r"""Generate tikz code corresponding to this graph.

        `Tikz` package must be loaded in the latex preamble, with `arrows.meta` library::

            \usepackage{tikz}
            \usetikzlibrary{arrows.meta}

        For labeled graphs, it is recommended to load `contour` package too::

            \usepackage[outline]{contour}
            \contourlength{0.5pt}

        """
        lines: List[str] = [
            r"\providecommand{\contour}[2]{#2}"  # avoid an error if package contour is not loaded.
            r"\begin{tikzpicture}["
            r"every node/.style = {font={\scriptsize}},"
            r"vertex/.style = {draw, circle,font={\scriptsize},inner sep=2},"
            "directed/.style = {-{Stealth[scale=1.1]}},"
            "reversed/.style = {{Stealth[scale=1.1]}-},"
            "undirected/.style = {},"
            f"{options}"
            "]"
        ]
        theta = 360 / self.order
        angles = {}
        nodes = list(self.nodes)
        if shuffle_nodes:
            random.shuffle(nodes)
        index: Dict[Node, int] = {node: i for i, node in enumerate(nodes)}
        # All nodes are placed around a circle, creating a regular polygon.
        for i, node in enumerate(nodes):
            angle = angles[node] = i * theta
            lines.append(rf"\node[vertex] ({node}) at ({angle}:1cm) {{${node}$}};")

        # Store nodes' cartesian coordinates.
        nodes_positions: Dict[Node, Tuple[float, float]] = {}
        for node in nodes:
            alpha = math.radians(angles[node])
            nodes_positions[node] = math.cos(alpha), math.sin(alpha)

        def label_position(node1_, node2_, k):
            """Return label cartesian coordinates for position `k`.

            Position `k` is a barycentric coefficient between 0 and 1."""
            x1, y1 = nodes_positions[node1_]
            x2, y2 = nodes_positions[node2_]
            return k * x2 + (1 - k) * x1, k * y2 + (1 - k) * y1

        # Store the places already occupied by labels, to avoid storing another label there.
        labels_positions: List[Tuple[float, float]] = []
        # Detect edges' intersections, to avoid positioning labels there.
        for node1 in nodes:
            for node2 in self.successors(node1) | self.predecessors(node1):
                if index[node1] < index[node2]:
                    for node3 in nodes:
                        for node4 in self.successors(node3) | self.predecessors(node3):
                            if index[node3] < index[node4] and len({node1, node2, node3, node4}) == 4:
                                A, B, C, D = (nodes_positions[node] for node in (node1, node2, node3, node4))
                                intersection = segments_intersection((A, B), (C, D))
                                if intersection is not None:
                                    labels_positions.append(intersection)

        for node1 in nodes:
            # If the graph is undirected, draw only i -> j edge and not j -> i edge,
            # since it is in fact the same edge.
            # An easy way to do that is to keep index[node2] always superior or equal to index[node1].
            for node2 in nodes if self.is_directed else nodes[index[node1] :]:
                if node1 == node2:
                    # This is a loop.
                    node = node1
                    style = "directed" if self.is_directed else "undirected"
                    n: int = self.count_edges(node, node, count_undirected_loops_twice=False)
                    if n > _TIKZ_EXPORT_MAX_MULTIPLE_LOOPS_SUPPORT:
                        raise NotImplementedError(n)
                    if n == 1:
                        (label,) = self.labels(node, node)
                        lines.append(
                            rf"\draw[{style}] ({node}) to "
                            f"[out={angles[node] - 45},in={angles[node] + 45},looseness=5] "
                            rf"node[midway] {{\contour{{white}}{{{label}}}}} "
                            f"({node});"
                        )
                else:
                    # This is a normal edge, joining two different nodes.
                    styles: List[str] = []
                    labels: List[str] = []
                    # Detect if node1 and node2 are neighbours on the circle.
                    node2_is_right_neighbour = (index[node1] - index[node2] - 1) % len(nodes) == 0
                    node2_is_left_neighbour = (index[node1] - index[node2] + 1) % len(nodes) == 0

                    if self.is_directed:
                        data = [("directed", node1, node2), ("reversed", node2, node1)]
                    else:
                        data = [("undirected", node1, node2)]
                    for direction, nodeA, nodeB in data:
                        labels.extend(self.labels(nodeA, nodeB))
                        styles += self.count_edges(nodeA, nodeB) * [direction]
                    n = len(styles)
                    assert len(labels) == n, f"len(styles)={n} != len(labels)={len(labels)}"
                    if n == 0:
                        bendings = []
                    elif n == 1:
                        bendings = [""]  # strait line by default
                        if len(nodes) >= 6:
                            if node2_is_left_neighbour:
                                bendings[0] = "bend right=30"
                            elif node2_is_right_neighbour:
                                bendings[0] = "bend left=30"
                    elif n == 2:
                        bendings = ["bend left=15", "bend right=15"]
                    elif n == 3:
                        bendings = ["bend left", "", "bend right"]
                    else:
                        assert n > _TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT
                        raise NotImplementedError(
                            f"Too much multiple edges : {n} > {_TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT} "
                            f"for graph {self}."
                        )
                    for style, bending, label in zip(styles, bendings, labels):
                        label_tikz_code = ""
                        if label:
                            pos = 0.5
                            if node2_is_right_neighbour or node2_is_left_neighbour:
                                pass
                            else:
                                # Try to minimize collisions between two labels.
                                # This dict will store the distance between the nearest labels
                                # for each position.
                                min_dists: Dict[float, float] = {}
                                # This one will store the coordinates of the label for each position.
                                coordinates: Dict[float, Tuple[float, float]] = {}
                                for pos in (
                                    0.1,
                                    0.2,
                                    0.3,
                                    0.4,
                                    0.5,
                                    0.6,
                                    0.7,
                                    0.8,
                                    0.9,
                                ):
                                    new_x, new_y = label_position(node1, node2, pos)
                                    min_dists[pos] = min(
                                        (new_x - x) ** 2 + (new_y - y) ** 2 for (x, y) in labels_positions
                                    )
                                    coordinates[pos] = new_x, new_y
                                pos = max(min_dists, key=min_dists.get)  # type: ignore
                                labels_positions.append(coordinates[pos])
                            label_tikz_code = rf"node[pos={pos}] {{\contour{{white}}{{{label}}}}}"
                        lines.append(rf"\draw[{style}] ({node1}) to[{bending}] {label_tikz_code} ({node2});")

        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)


class InvalidGraphAttribute(AttributeError):
    """This error is raised when a method or property can't return a value for a particular graph."""
