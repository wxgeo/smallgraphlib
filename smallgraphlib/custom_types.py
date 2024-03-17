from typing import TypeVar, FrozenSet, Union, Set, Iterable, Any, TYPE_CHECKING

from smallgraphlib.utilities import ComparableAndHashable

if TYPE_CHECKING:
    from smallgraphlib.core import AbstractGraph

_AbstractGraph = TypeVar("_AbstractGraph", bound="AbstractGraph")
Node = TypeVar("Node", bound=ComparableAndHashable)
# Node = TypeVar("Node", bound=typing.Hashable)  # too subtile for Pycharm ? ;-(
DirectedEdge = tuple[Node, Node]
UndirectedEdge = FrozenSet[Node]
Edge = Union[DirectedEdge, UndirectedEdge]
EdgeLike = Union[Edge, Set[Node], Iterable[Node]]
Label_ = Any
InternalGraphRepresentation = dict[Node, dict[Node, Union[int, list[Label_]]]]
Point = tuple[float, float]
Segment = tuple[Point, Point]
Label = TypeVar("Label")
LabeledEdge = tuple[Node, Node, Label]
WeightedEdge = tuple[Node, Node, float]
