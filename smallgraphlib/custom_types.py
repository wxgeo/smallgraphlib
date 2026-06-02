from typing import TypeVar, Union, Any, TYPE_CHECKING
from collections.abc import Iterable

from smallgraphlib.utilities import ComparableAndHashable

if TYPE_CHECKING:
    from smallgraphlib.core import AbstractGraph

_AbstractGraph = TypeVar("_AbstractGraph", bound="AbstractGraph")
Node = TypeVar("Node", bound=ComparableAndHashable)
# Node = TypeVar("Node", bound=typing.Hashable)  # too subtile for Pycharm ? ;-(
DirectedEdge = tuple[Node, Node]
UndirectedEdge = frozenset[Node]
Edge = Union[DirectedEdge, UndirectedEdge]
EdgeLike = Union[Edge, set[Node], Iterable[Node]]
Label_ = Any
InternalGraphRepresentation = dict[Node, dict[Node, int | list[Label_]]]
Point = tuple[float, float]
Segment = tuple[Point, Point]
Label = TypeVar("Label")
LabeledEdge = tuple[Node, Node, Label]
WeightedEdge = tuple[Node, Node, float]
