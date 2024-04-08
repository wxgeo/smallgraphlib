import math
import random
from typing import Generic, TYPE_CHECKING, Final

from smallgraphlib.utilities import frange, cached_property, clear_cache

from smallgraphlib.custom_types import Node, Segment, Point, Label

if TYPE_CHECKING:
    from smallgraphlib.core import AbstractGraph
    from smallgraphlib.labeled_graphs import AbstractLabeledGraph
    from smallgraphlib.automatons import Automaton, Transducer, Char, Acceptor
    from smallgraphlib.flow_networks import FlowNetwork

_TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT = 3

GREEK_LETTERS: Final[tuple[str, ...]] = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)


def segments_intersection(segment1: Segment, segment2: Segment, eps: float = 10**-8) -> Point | None:
    (xA, yA), (xB, yB) = segment1
    (xC, yC), (xD, yD) = segment2
    # solve: s*A + (1 - s)*B = t*C + (1 - t)*D  <==>  s*(A - B) + t*(D - C) = D - B
    #                                           <==>  s*xBA + t*xCD = xBD  and  s*yBA + t*yCD = yBD
    xCD, yCD = xD - xC, yD - yC
    xBA, yBA = xA - xB, yA - yB
    xBD, yBD = xD - xB, yD - yB
    t = s = None
    intersection: Point | None = None
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


def barycenter(point1: Point, point2: Point, k: float) -> Point:
    """Return the label cartesian coordinates for position `k`.

    Position `k` is a barycentric coefficient between 0 and 1."""
    x1, y1 = point1
    x2, y2 = point2
    return k * x2 + (1 - k) * x1, k * y2 + (1 - k) * y1


def label_coordinates(point1: Point, point2: Point, k: float, bending: int) -> Point:
    """Return an approximation of the label cartesian coordinates on a bent tikz edge for position `k`.

    Position `k` is a barycentric coefficient between 0 and 1, as used in `pos` tikz parameter.
    The absolute `bending` value is that of the tikz `bending` parameter.
    It must be positive to bend left, and negative to bend right.
    """
    x1, y1 = point1
    x2, y2 = point2

    # We will calculate a fast approximation of the position of the label on the tikz bent edge.
    # Tikz uses Bézier curves internally, but I don't know the exact algorithm and anyway,
    # we only need a quick approximation.
    # Let's call `A` the first point, `B` the second one, and `I` the middle of the segment `(A B)`.
    # The maximal distance between the segment `(A, B)` and the edge is when the label is positioned
    # midway between `A` and `B`. If we call `M` the position of this midway label, then the maximal distance
    # is `IM`.
    #
    # Let's make a quick sketch:
    #
    #                   M
    #         N    ...  +  ...
    #        . +        |         ..
    #   P +    |        |             .
    #   .      |        |               .
    #  +--+----+--------+----------------+
    #  A  C    J        I                B
    #
    # Here, `N` is the middle of the arc `A --> M`, and `J` is the middle of the segment `(A I)`.
    #
    # The point `C` is the barycenter corresponding to the coefficient `k` on the segment (A B),
    # and is easy to compute.
    # However, we want the coordinates of the point `P`, which is the real position of the tikz label
    # on the curved edge for the same coefficient `k`.
    # So, we have to calculate an approximation of the vector `C->P`.
    #
    # (Note that the vectors `J->N` and `C->P` are not exactly orthogonal to the vector `A->B`,
    # but in practice the deviation is not so important since the bending is slight,
    # and because `k` is often around 0.5).
    #
    # +--------------------------+-----+-----+--------+-------+-----+
    # |     `k` value            |  0  |  k  |  0.25  |  0.5  |  1  |
    # +--------------------------+-----+-----+--------+-------+-----+
    # |  point on segment (A B)  |  A  |  C  |   J    |   I   |  B  |
    # +--------------------------+-----+-----+--------+-------+-----+
    # |  point on the bent edge  |  A  |  P  |   N    |   M   |  B  |
    # +--------------------------+-----+--------------+-------+-----+
    #
    # Experimentally, on a tikz drawing, the distance `IM` is approximately `0.006 * bending * AB`,
    # where `bending` is the value of the `bending` parameter of tikz.
    #
    # For the distance `JN`, still experimentally, the distance is approximately `0.0045 * bending * AB`
    # (so 0.0045 instead of 0.006/2 = 0.003, because of the curving of course).
    #
    # Since the edge is only slightly curved, we can now approximate `A --> N` and `N --> M`
    # portions of curves by a straight line.
    # Then, on each straight line, the distance between the edge and the `(A, B)` segment is
    # proportional to the position of the label on the edge.
    #
    # If 0 <= k <= 0.25, the distance from the label to the segment is approximately `4 * k * JN`.
    # (since for k = 0.25, the distance is of course `JN` itself).
    # If 0.75 <= k <= 1, the distance is `4 * (1 - k) * JN`.
    # So the approximation formula if abs(0.5 - k) > 0.25 is:
    #
    #                           `distance CP = 4 * min(k, 1 - k) * JN`
    #
    # So, `CP = 4 * min(k, 1 - k) * 0.0045 * bending * AB` and finally:
    #
    #                            `CP = 0.018 * bending * min(k, 1 - k) * AB`
    #
    # Now, if 0.25 <= k <= 0.5, the distance from the label to the segment is approximately
    # `JN + 4 * (k - 0.25) * (IM - JN)`, since for k = 0.25, the distance is of course `JN` itself,
    # and for k = 0.5, the distance is `IM`.
    # So the approximation formula for abs(0.5 - k) < 0.25 is:
    #
    #                       `distance CP = JN + 4 * min(k - 0.25, 0.75 - k) * (IM - JN)`
    #
    # This leads to:
    #
    #                       `CP = (0.006 * min(k - 0.25, 0.75 - k) + 0.0045) * bending * AB`
    #
    # The vector `I->M` is orthogonal to `A->B`, and its length is approximately `0.006 * bending * AB`.
    # So its coordinates are `0.006 * bending * (-yAB, xAB)` for a left bending.
    #
    # It follows that C->P coordinates are:
    #
    #              `C->P = | 0.018 * bending * min(k, 1 - k) * (-yAB, xAB)` if abs(0.5 - k) > 0.25
    #                      | (0.006 * min(k - 0.25, 0.75 - k) + 0.0045) * bending * (-yAB, xAB) else
    #
    x, y = barycenter(point1, point2, k)
    if bending == 0:
        # Small optimisation, since bending == 0 is the most frequent case.
        return x, y
    if abs(0.5 - k) > 0.25:
        # 0 <= k < 0.25
        coefficient = bending * 0.018 * min(k, 1 - k)
    else:
        # 0.25 <= k <= 0.5
        coefficient = bending * (0.006 * min(k - 0.25, 0.75 - k) + 0.0045)
    dx, dy = coefficient * (y1 - y2), coefficient * (x2 - x1)
    return x + dx, y + dy


def find_free_position(
    point1: Point, point2: Point, known_positions: list[Point], bending: int = 0
) -> tuple[float, Point]:
    """Find a free position on the edge between point1 and point2.

    Return a tuple:
        - the best position on the segment (point1, point2): a float between 0 and 1,
        - the coordinates of the corresponding point: a tuple of floats.

    The returned position maximize the shortest distance between the new position
    and all other existing positions.

    Parameter `bending` is the corresponding tikz bending.
    """
    # Try to minimize collisions between two labels.
    # This dict will store the distance between the nearest labels
    # for each position.
    min_dists: dict[float, float] = {}
    # This one will store the coordinates of the label for each position.
    coordinates: dict[float, tuple[float, float]] = {}
    for pos in frange(0.2, 0.81, 0.05):
        # The edge may be slightly curved, so the label position is not exactly a barycenter
        # of the two nodes positions.
        new_x, new_y = label_coordinates(point1, point2, pos, bending=bending)
        min_dists[pos] = min(
            [(new_x - x) ** 2 + (new_y - y) ** 2 for (x, y) in known_positions],
            default=math.inf,
        )
        coordinates[pos] = new_x, new_y
    pos = max(min_dists, key=min_dists.__getitem__, default=0.5)
    return pos, coordinates[pos]


class TikzPrinter(Generic[Node]):
    lines: list[str]
    # `nodes_positions` stores the cartesian coordinates of each of the graph's node.
    nodes_positions: dict[Node, Point]
    # The places already occupied by labels are stored in `labels_positions`, to avoid placing another
    # label there.
    labels_positions: list[Point]
    # Nodes numeration (useful in particular when they are shuffled):
    index: dict[Node, int]
    # For debugging:
    _cartography: dict[tuple[Node, ...], Point]

    def __init__(self, graph: "AbstractGraph[Node]", shuffle_nodes=False):
        self.graph: "AbstractGraph[Node]" = graph
        self._reset()
        self.shuffle_nodes: bool = shuffle_nodes

    @clear_cache
    def _reset(self):
        self.lines = []
        self.nodes_positions = {}
        self.labels_positions = []
        self._cartography = {}
        self.index = {}

    @staticmethod
    def latex_preamble_additions() -> list[str]:
        """Return the lines of LaTeX code to insert in the LaTeX preamble."""
        return [
            r"\usepackage{tikz}",
            r"\usetikzlibrary{arrows.meta}",
            r"\usepackage[outline]{contour}",
            r"\contourlength{0.15em}",
        ]

    @cached_property
    def angles(self) -> dict[Node, float]:
        """Overwrite this method to modify tikz automatic nodes' placement."""
        theta = 360 / self.graph.order
        return {node: i * theta for i, node in enumerate(self.nodes)}

    @cached_property
    def nodes(self) -> tuple[Node, ...]:
        nodes = list(self.graph.nodes)
        if self.shuffle_nodes:
            random.shuffle(nodes)
        return tuple(nodes)

    def specific_node_style(self, node: Node) -> str:
        """Overwrite this method to add a specific tikz style to some nodes."""
        return ""

    def labels(self, node1: Node, node2: Node) -> list[str]:
        """Overwrite this method to modify tikz value for some labels."""
        return self.graph.labels(node1, node2)

    def tikz_code(self, *, border: str = None, options="") -> str:
        r"""Generate tikz code corresponding to this graph.

        `Tikz` package must be loaded in the latex preamble, with `arrows.meta` library::

            \usepackage{tikz}
            \usetikzlibrary{arrows.meta}

        For labeled graphs, it is recommended to load `contour` package too::

            \usepackage[outline]{contour}
            \contourlength{0.15em}

        If set, `border` have to be a combination of tikz path drawing styles,
        like "dotted", or "dashed,blue".
        """
        if border:
            return "".join(
                [
                    rf"\begin{{tikzpicture}}\node[rounded corners,draw,inner sep=2pt,{border}]{{",
                    self.tikz_code(options=options),
                    r"};\end{tikzpicture}",
                ]
            )
        self._reset()
        self.lines = [
            r"\providecommand{\contour}[2]{#2}"  # avoid an error if package contour is not loaded.
            r"\begin{tikzpicture}[solid,black,"
            r"every node/.style = {font={\scriptsize}},"
            r"vertex/.style = {draw, circle,font={\scriptsize},inner sep=2},"
            "directed/.style = {-{Stealth[scale=1.1]}},"
            "reversed/.style = {{Stealth[scale=1.1]}-},"
            "undirected/.style = {},"
            f"{options}"
            "]"
        ]
        # theta = 360 / self.graph.order
        nodes = self.nodes  # = list(self.graph.nodes)
        self.index = {node: i for i, node in enumerate(nodes)}
        # All nodes are placed around a circle, creating a regular polygon.
        for i, node in enumerate(nodes):
            angle = self.angles[node]
            specific_style = self.specific_node_style(node)
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
                                    self._cartography[(node1, node2, node3, node4)] = intersection

        # Let's draw now the edges and the labels.
        #
        # If the graph is undirected, one should only draw i -> j edge and not j -> i edge,
        # since it is in fact the same edge.
        # An easy way to do that is to keep index[node2] always superior or equal to index[node1].
        #
        # Generating a simple-to-read graph is not so easy.
        # One should avoid labels' collisions notably.
        # For the edges joining adjacent nodes, the label should always be placed midway for readability.
        # For the other edges, different positions are tried to minimize collisions' risk.

        # First, draw loops
        for node in nodes:
            self._generate_loop(node)

        # Then, draw the edges joining neighbours, since the label position is fixed.
        # (The label is always positioned midway because those edges are short).
        if len(nodes) >= 2:
            for i, node in enumerate(nodes):
                self._generate_edge(node, nodes[(i + 1) % len(nodes)])

            for node1 in nodes:
                for node2 in nodes[self.index[node1] + 2 :]:
                    self._generate_edge(node1, node2)

        self.lines.append(r"\end{tikzpicture}")
        return "\n".join(self.lines)

    def _generate_loop(self, node: Node) -> None:
        style = "directed" if self.graph.is_directed else "undirected"
        for i, label in enumerate(self.labels(node, node), start=1):
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
            _labels = self.labels(nodeA, nodeB)
            labels.extend(_labels)
            styles += len(_labels) * [direction]
        n = len(styles)
        if n == 0:
            bendings = []
        elif n == 1:
            bendings = [0]  # strait line by default
            if len(self.nodes) >= 6:
                if node2_is_left_neighbour:
                    bendings[0] = -30
                elif node2_is_right_neighbour:
                    bendings[0] = 30
        elif n == 2:
            bendings = [15, -15]
        elif n == 3:
            bendings = [30, 0, -30]
        else:
            assert n > _TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT
            raise NotImplementedError(
                f"Too much multiple edges : {n} > {_TIKZ_EXPORT_MAX_MULTIPLE_EDGES_SUPPORT} "
                f"for graph {self}."
            )
        for style, bending, label in zip(styles, bendings, labels):
            label_tikz_code = ""
            if label:
                point1 = self.nodes_positions[node1]
                point2 = self.nodes_positions[node2]
                if node2_is_right_neighbour or node2_is_left_neighbour:
                    pos = 0.5
                    xy = label_coordinates(point1, point2, pos, bending=bending)
                else:
                    pos, xy = find_free_position(point1, point2, self.labels_positions, bending=bending)
                self.labels_positions.append(xy)
                self._cartography[(node1, node2)] = xy
                label_tikz_code = rf"node[pos={pos}] {{\contour{{white}}{{{label}}}}}"
            direction = "left" if bending > 0 else "right"
            tikz_bending = f"bend {direction}={abs(bending)}" if bending != 0 else ""
            self.lines.append(rf"\draw[{style}] ({node1}) to[{tikz_bending}] {label_tikz_code} ({node2});")


class TikzLabeledGraphPrinter(TikzPrinter, Generic[Node, Label]):
    def __init__(self, graph: "AbstractLabeledGraph[Node, Label]", shuffle_nodes=False):
        self.graph: "AbstractLabeledGraph[Node, Label]" = graph  # For Pycharm
        super().__init__(graph, shuffle_nodes=shuffle_nodes)

    def labels(self, node1: Node, node2: Node) -> list[str]:
        """Overwrite this method to modify tikz value for some labels."""
        labels = self.graph._labels.get(self.graph._edge(node1, node2), [])

        def format_(label):
            # Note: math.isinf() supports `sympy.oo` too.
            if label is None:
                return ""
            elif math.isinf(label) and label > 0:
                return r"$\infty$"
            elif math.isinf(label) and label < 0:
                return r"$-\infty$"
            else:
                return str(label)

        return [format_(label) for label in labels]


class TikzAutomatonPrinter(TikzPrinter, Generic[Node]):
    def __init__(self, graph: "Automaton[Node]", shuffle_nodes=False):
        self.graph: "Automaton[Node]" = graph  # For Pycharm
        super().__init__(graph, shuffle_nodes=shuffle_nodes)

    def specific_node_style(self, node: Node) -> str:
        styles = []
        if node in self.graph.initial_states:
            styles.append("rectangle")
        return ",".join(styles)

    def labels(self, node1, node2) -> list[str]:
        labels = sorted(self.graph.labels(node1, node2))
        if (
            self.graph.alphabet_name is not None
            and sorted(labels) == list(self.graph.alphabet)
            and len(self.graph.alphabet) > 1
        ):
            return [self._latex(self.graph.alphabet_name)]
        return [",".join(self._latex(label) for label in labels)] if labels else []

    @staticmethod
    def _latex(label: str) -> str:
        label = label.replace("#", r"\#")
        if label in GREEK_LETTERS:
            label = "\\" + label
        return f"${label}$" if label else r"$\varepsilon$"


class TikzAcceptorPrinter(TikzAutomatonPrinter, Generic[Node]):
    def __init__(self, graph: "Acceptor[Node]", shuffle_nodes=False):
        self.graph: "Acceptor[Node]" = graph  # For Pycharm
        super().__init__(graph, shuffle_nodes=shuffle_nodes)

    def specific_node_style(self, node: Node) -> str:
        styles = [super().specific_node_style(node)]
        if node in self.graph.final_states:
            styles.append("double,fill=lightgray")
        return ",".join(styles)


class TikzTransducerPrinter(TikzAutomatonPrinter, Generic[Node]):
    def __init__(self, graph: "Transducer[Node]", shuffle_nodes=False):
        self.graph: "Transducer[Node]" = graph  # For Pycharm
        super().__init__(graph, shuffle_nodes=shuffle_nodes)

    def labels(self, node1: Node, node2: Node) -> list[str]:
        # Associate to each output word all the input letters than can produce it.
        words: dict[str, set[Char]] = {}
        for letter in self.graph.alphabet:
            if self.graph.next_state(node1, letter) == node2:
                words.setdefault(self.graph.next_word(node1, letter), set()).add(letter)

        labels: list[str] = []
        for word, letters in words.items():
            sorted_letters = sorted(letters)
            if sorted_letters == list(self.graph.alphabet) and self.graph.alphabet_name is not None:
                letters_str = self.graph.alphabet_name
            else:
                letters_str = ",".join(sorted_letters)
            letters_str = self._latex(letters_str)
            if word:
                letters_str += rf"\fbox{{{self._latex(word)}}}"
            labels.append(letters_str)
        return labels


class TikzFlowNetworkPrinter(TikzPrinter, Generic[Node]):
    def __init__(self, graph: "FlowNetwork[Node]", shuffle_nodes=False):
        self.graph: "FlowNetwork[Node]" = graph  # For Pycharm
        super().__init__(graph, shuffle_nodes=shuffle_nodes)

    @cached_property
    def nodes(self) -> tuple[Node, ...]:
        """Return nodes in a nice order for tikz export."""
        source = self.graph.source
        sink = self.graph.sink
        # When exporting graph as a tikz picture, all nodes are displayed on a circle.
        # However, since it is a network flow, it is better to display source and sink at both extremities
        # of the drawing (source at left, sink at right).
        # We'll try to display nodes in a nice way too, detecting paths, to minimize edges' crossings,
        # # though it's not so obvious...
        angles: dict[Node, float] = {
            source: 180,  # Display the source at the extreme-left of the graph.
            sink: 0,  # Display the sink at the extreme-right of the graph.
        }
        path = self.graph.find_path(source, sink)
        assert path[0] == source and path[-1] == sink
        # If we find a path from source to sink, we display its nodes from left to right, on the top of the graph.
        # We must remove source and sink from path, since there are already handled.
        paths: tuple[list[Node], list[Node]] = (
            path[1:-1],
            self.graph.find_path(source, sink, _filter_nodes=path[1:-1])[1:-1],
        )
        assert len(paths) == 2
        remaining_nodes = self.graph.nodes_set - set(paths[0]) - set(paths[1]) - {source, sink}
        for i, node in enumerate(remaining_nodes):
            paths[i % 2].append(node)

        for j, path in enumerate(paths):
            theta = 180 / (len(path) + 1)
            for i, node in enumerate(reversed(path) if j == 0 else path, start=1):
                angles[node] = j * 180 + i * theta
        return tuple(node for angle, node in sorted((angle, node) for node, angle in angles.items()))
