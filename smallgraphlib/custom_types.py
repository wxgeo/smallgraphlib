from typing import TypeVar, Tuple, FrozenSet, Union, Set, Iterable, Any, Dict, List, TYPE_CHECKING

from smallgraphlib.utilities import ComparableAndHashable

if TYPE_CHECKING:
    from smallgraphlib.core import AbstractGraph

_AbstractGraph = TypeVar("_AbstractGraph", bound="AbstractGraph")
Node = TypeVar("Node", bound=ComparableAndHashable)
# Node = TypeVar("Node", bound=typing.Hashable)  # too subtile for Pycharm ? ;-(
DirectedEdge = Tuple[Node, Node]
UndirectedEdge = FrozenSet[Node]
Edge = Union[DirectedEdge, UndirectedEdge]
EdgeLike = Union[Edge, Set[Node], Iterable[Node]]
Label = Any
InternalGraphRepresentation = Dict[Node, Dict[Node, Union[int, List[Label]]]]
Point = Tuple[float, float]
Segment = Tuple[Point, Point]
