import math
import random
from typing import Generic, TYPE_CHECKING

from smallgraphlib.custom_types import Node, Segment, Point

if TYPE_CHECKING:
    from smallgraphlib.core import AbstractGraph


_TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT = 3


def segments_intersection(segment1: Segment, segment2: Segment, eps: float = 10**-8):
    (xA, yA), (xB, yB) = segment1
    (xC, yC), (xD, yD) = segment2
    # solve: s*A + (1 - s)*B = t*C + (1 - t)*D  <==>  s*(A - B) + t*(D - C) = D - B
    #                                           <==>  s*xBA + t*xCD = xBD  and  s*yBA + t*yCD = yBD
    xCD, yCD = xD - xC, yD - yC
    xBA, yBA = xA - xB, yA - yB
    xBD, yBD = xD - xB, yD - yB
    t = s = None
    intersection = None
    if abs(yCD * xBA - xCD * yBA) > eps:
        if abs(xBA) > abs(xCD):
            t = (yBD * xBA - xBD * yBA) / (yCD * xBA - xCD * yBA)
            s = (xBD - t * xCD) / xBA
        else:
            s = (yBD * xCD - xBD * yCD) / (yBA * xCD - xBA * yCD)
            t = (xBD - s * xBA) / xCD
    if t is not None and s is not None and (0 <= t <= 1 and 0 <= s <= 1):
        # assert abs(s * xA + (1 - s) * xB - (t * xC + (1 - t) * xD)) < eps
        # assert abs(s * yA + (1 - s) * yB - (t * yC + (1 - t) * yD)) < eps
        intersection = s * xA + (1 - s) * xB, s * yA + (1 - s) * yB
    return intersection


class TikzPrinter(Generic[Node]):
    lines: list[str]
    # `nodes_positions` stores the cartesian coordinates of each of the graph's node.
    nodes_positions: dict[Node, tuple[float, float]]
    # The places already occupied by labels are stored in `labels_positions`, to avoid placing another
    # label there.
    labels_positions: list[tuple[float, float]]
    graph: "AbstractGraph[Node]"
    angles: dict[Node, float]
    # Nodes numeration (useful in particular when they are shuffled):
    index: dict[Node, int]
    nodes: list[Node]

    def _reset(self):
        self.lines = []
        self.nodes_positions = {}
        self.labels_positions = []
        self.angles = {}
        self.index = {}

    def tikz_code(self, graph, *, shuffle_nodes=False, options="") -> str:
        r"""Generate tikz code corresponding to this graph.

        `Tikz` package must be loaded in the latex preamble, with `arrows.meta` library::

            \usepackage{tikz}
            \usetikzlibrary{arrows.meta}

        For labeled graphs, it is recommended to load `contour` package too::

            \usepackage[outline]{contour}
            \contourlength{0.5pt}

        """
        self.graph = graph
        self._reset()
        self.lines = [
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
        theta = 360 / self.graph.order
        nodes = self.nodes = list(self.graph.nodes)
        if shuffle_nodes:
            random.shuffle(nodes)
        self.index = {node: i for i, node in enumerate(nodes)}
        # All nodes are placed around a circle, creating a regular polygon.
        for i, node in enumerate(nodes):
            angle = self.angles[node] = i * theta
            specific_style = self.graph._tikz_specific_node_style(node)
            self.lines.append(rf"\node[vertex,{specific_style}] ({node}) at ({angle}:1cm) {{${node}$}};")

        for node in nodes:
            alpha = math.radians(self.angles[node])
            self.nodes_positions[node] = math.cos(alpha), math.sin(alpha)

        # Detect edges' intersections, to avoid positioning labels there.
        for node1 in nodes:
            for node2 in self.graph.successors(node1) | self.graph.predecessors(node1):
                if self.index[node1] < self.index[node2]:
                    for node3 in nodes:
                        for node4 in self.graph.successors(node3) | self.graph.predecessors(node3):
                            if (
                                self.index[node3] < self.index[node4]
                                and len({node1, node2, node3, node4}) == 4
                            ):
                                A, B, C, D = (
                                    self.nodes_positions[node] for node in (node1, node2, node3, node4)
                                )
                                intersection = segments_intersection((A, B), (C, D))
                                if intersection is not None:
                                    self.labels_positions.append(intersection)

        # Let's draw now the edges and the labels.
        #
        # If the graph is undirected, one should only draw i -> j edge and not j -> i edge,
        # since it is in fact the same edge.
        # An easy way to do that is to keep index[node2] always superior or equal to index[node1].
        #
        # Generating an simple-to-read graph is not so easy.
        # One should avoid labels' collisions notably.
        # For the edges joining adjacent nodes, the label should always be placed midway for readability.
        # For the other edges, different positions are tried to minimize collisions' risk.

        # First, draw loops
        for node in nodes:
            self._generate_loop(node)

        # Then, draw the edges joining neighbours, since the label position is fixed.
        # (The label is always positioned midway because those edges are short).
        for i, node in enumerate(nodes):
            self._generate_edge(node, nodes[(i + 1) % len(nodes)])

        for node1 in nodes:
            for node2 in nodes[self.index[node1] + 2 :]:
                self._generate_edge(node1, node2)

        self.lines.append(r"\end{tikzpicture}")
        return "\n".join(self.lines)

    def _calculate_label_position(self, node1: Node, node2: Node, k: float) -> Point:
        """Return the label cartesian coordinates for position `k`.

        Position `k` is a barycentric coefficient between 0 and 1."""
        x1, y1 = self.nodes_positions[node1]
        x2, y2 = self.nodes_positions[node2]
        return k * x2 + (1 - k) * x1, k * y2 + (1 - k) * y1

    def _generate_loop(self, node: Node) -> None:
        style = "directed" if self.graph.is_directed else "undirected"
        # n: int = self._tikz_count_edges(node, node)
        for i, label in enumerate(self.graph._tikz_labels(node, node), start=1):
            self.lines.append(
                rf"\draw[{style}] ({node}) to "
                f"[out={self.angles[node] - 45},in={self.angles[node] + 45},looseness={1 + i * 4}] "
                rf"node[midway] {{\contour{{white}}{{{label}}}}} "
                f"({node});"
            )

    def _generate_edge(self, node1: Node, node2: Node) -> None:
        assert node1 != node2
        # This is a normal edge, joining two different nodes.
        styles: list[str] = []
        labels: list[str] = []
        # Detect if node1 and node2 are neighbours on the circle.
        node2_is_right_neighbour = (self.index[node1] - self.index[node2] - 1) % len(self.nodes) == 0
        node2_is_left_neighbour = (self.index[node1] - self.index[node2] + 1) % len(self.nodes) == 0

        if self.graph.is_directed:
            data = [("directed", node1, node2), ("reversed", node2, node1)]
        else:
            data = [("undirected", node1, node2)]
        for direction, nodeA, nodeB in data:
            _labels = self.graph._tikz_labels(nodeA, nodeB)
            labels.extend(_labels)
            styles += len(_labels) * [direction]
        n = len(styles)
        if n == 0:
            bendings = []
        elif n == 1:
            bendings = [""]  # strait line by default
            if len(self.nodes) >= 6:
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
                    min_dists: dict[float, float] = {}
                    # This one will store the coordinates of the label for each position.
                    coordinates: dict[float, tuple[float, float]] = {}
                    for pos in (
                        # 0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        # 0.9,
                    ):
                        new_x, new_y = self._calculate_label_position(node1, node2, pos)
                        min_dists[pos] = min(
                            [(new_x - x) ** 2 + (new_y - y) ** 2 for (x, y) in self.labels_positions],
                            default=math.inf,
                        )
                        coordinates[pos] = new_x, new_y
                    pos = max(min_dists, key=min_dists.get, default=0.5)  # type: ignore
                    self.labels_positions.append(coordinates[pos])
                label_tikz_code = rf"node[pos={pos}] {{\contour{{white}}{{{label}}}}}"
            self.lines.append(rf"\draw[{style}] ({node1}) to[{bending}] {label_tikz_code} ({node2});")


tikz_printer: TikzPrinter = TikzPrinter()
