import math
import random

from smallgraphlib import WeightedDirectedGraph, random_graph, Graph
from smallgraphlib.printers.tikz import (
    segments_intersection,
    barycenter,
    find_free_position,
    label_coordinates,
    TikzPrinter,
)


def test_tikz():
    oo = math.inf
    M = [
        [0, 16, oo, oo, 3, 15, 10],
        [oo, 0, 1, 4, oo, oo, oo],
        [oo, 1, 0, oo, oo, oo, oo],
        [oo, 4, 2, 0, oo, oo, oo],
        [oo, 13, 17, 7, 0, oo, oo],
        [15, 2, oo, oo, oo, 0, 3],
        [oo, oo, oo, oo, oo, 3, 0],
    ]
    WeightedDirectedGraph.from_matrix(M).as_tikz()
    WeightedDirectedGraph.from_matrix(M).as_tikz(border="dotted")


def test_tikz_support():
    for seed in range(100):
        random.seed(seed)
        g = random_graph(4, 6, directed=True)
        g.as_tikz()


def test_tikz_single_node():
    g = Graph(["I"])
    g.as_tikz()


def test_intersection():
    A = (-4.191374663072777, -0.4986522911051212)
    B = (-0.41778975741239854, 1.495956873315364)
    C = (-3.4905660377358494, 2.035040431266846)
    D = (0.8760107816711589, -0.12129380053908356)
    expected_intersection = (-1.374693295677149, 0.9901650030897102)
    M = segments_intersection((A, B), (C, D))
    assert math.hypot(M[0] - expected_intersection[0], M[1] - expected_intersection[1]) < 10**-8
    A = (-2.0572283216093616, 1.544030635724147)
    assert segments_intersection((A, B), (C, D)) is None


def test_barycenter():
    A = (0, 0)
    B = (2, 1)
    x, y = barycenter(A, B, 0.2)
    assert abs(x - 0.4) < 1e-10
    assert abs(y - 0.2) < 1e-10


def test_free_position():
    A = (0, 0)
    B = (20, 10)

    def assert_free_position(position):
        position /= 10
        pos, (x, y) = find_free_position(A, B, occupied_positions)
        xM, yM = barycenter(A, B, position)
        assert abs(x - xM) < 1e-6, f"{x=} {xM=}"
        assert abs(y - yM) < 1e-6, f"{y=} {yM=}"
        assert abs(pos - position) < 1e-6, f"{pos=} {position=}"

    occupied_positions = [barycenter(A, B, k / 10) for k in range(2, 6)]
    assert_free_position(8)

    occupied_positions = [barycenter(A, B, k / 10) for k in range(4, 9)]
    assert_free_position(2)

    occupied_positions = [barycenter(A, B, k / 10) for k in (2, 3, 4) + (6, 7, 8)]
    assert_free_position(5)

    occupied_positions = [barycenter(A, B, k / 10) for k in (2, 3) + (7, 8)]
    assert_free_position(5)

    occupied_positions = [barycenter(A, B, k / 10) for k in (2,) + (6, 7, 8)]
    assert_free_position(4)

    A = (0, 0)
    B = (10, 0)
    for y in (-2, -1, 0, 1, 2):
        for missing in range(2, 9):
            x_list = list(range(1, missing)) + list(range(missing + 1, 10))
            occupied_positions = [(x, y) for x in x_list]
            assert_free_position(missing)


def test_label_coordinates():
    A = (9.95, 5)
    B = (14.3, -4.25)
    x, y = label_coordinates(A, B, k=0.25, bending=-30)
    assert abs(x - 9.79) < 0.01, (x, y)
    assert abs(y - 2.1) < 0.01, (x, y)
    x, y = label_coordinates(A, B, k=0.4, bending=30)
    assert abs(x - 13.19) < 0.01, (x, y)
    assert abs(y - 2) < 0.01, (x, y)
    x, y = label_coordinates(A, B, k=0.4, bending=-30)
    assert abs(x - 10.19) < 0.01, (x, y)
    assert abs(y - 0.6) < 0.01, (x, y)


def test_latex_preamble_additions():
    # Test that the docstring of `TikzPrinter.tikz_code()` is coherent
    # with the LaTeX code generated by `TikzPrinter.latex_preamble_additions()`.
    preamble = set(line.strip() for line in TikzPrinter.latex_preamble_additions())
    doc = set(line.strip() for line in TikzPrinter.tikz_code.__doc__.split("\n"))
    assert doc & preamble == preamble


def test_tikz_result():
    g = Graph.from_subgraphs("P1,P7 P2,P3,P4,P8 P5,P6,P8 P1,P4,P5 P3,P7 P6,P3")
    result_template = r"""<PREAMBLE>
\begin{tikzpicture}[
solid,black,
every node/.style = {font={\scriptsize}},
vertex/.style = {draw, circle,font={\scriptsize},inner sep=2},
directed/.style = {-{Stealth[scale=1.1]}},
reversed/.style = {{Stealth[scale=1.1]}-},
undirected/.style = {},
<OPTIONS>
]
    \node[vertex,] (P1) at (0.0:1cm) {$P_{1}$};
    \node[vertex,] (P2) at (45.0:1cm) {$P_{2}$};
    \node[vertex,] (P3) at (90.0:1cm) {$P_{3}$};
    \node[vertex,] (P4) at (135.0:1cm) {$P_{4}$};
    \node[vertex,] (P5) at (180.0:1cm) {$P_{5}$};
    \node[vertex,] (P6) at (225.0:1cm) {$P_{6}$};
    \node[vertex,] (P7) at (270.0:1cm) {$P_{7}$};
    \node[vertex,] (P8) at (315.0:1cm) {$P_{8}$};
    \draw[undirected] (P2) to[bend right=30]  (P3);
    \draw[undirected] (P3) to[bend right=30]  (P4);
    \draw[undirected] (P4) to[bend right=30]  (P5);
    \draw[undirected] (P5) to[bend right=30]  (P6);
    \draw[undirected] (P1) to[]  (P4);
    \draw[undirected] (P1) to[]  (P5);
    \draw[undirected] (P1) to[]  (P7);
    \draw[undirected] (P2) to[]  (P4);
    \draw[undirected] (P2) to[]  (P8);
    \draw[undirected] (P3) to[]  (P6);
    \draw[undirected] (P3) to[]  (P7);
    \draw[undirected] (P3) to[]  (P8);
    \draw[undirected] (P4) to[]  (P8);
    \draw[undirected] (P5) to[]  (P8);
    \draw[undirected] (P6) to[]  (P8);
\end{tikzpicture}"""
    default_preamble = r"\providecommand{\contour}[2]{#2}"
    result = result_template.replace("<OPTIONS>\n", "").replace("<PREAMBLE>", default_preamble)
    assert g.as_tikz() == result
    result = result_template.replace("<OPTIONS>\n", "scale=2\n").replace("<PREAMBLE>", default_preamble)
    assert g.as_tikz(options="scale=2") == result
    result = (
        "\\begin{tikzpicture}\n"
        f"\\node[rounded corners,draw,inner sep=2pt,dotted]{{\n{result}\n}};\n"
        "\\end{tikzpicture}"
    )
    assert g.as_tikz(border="dotted", options="scale=2") == result
    result = result_template.replace("<OPTIONS>\n", "").replace(
        "<PREAMBLE>", "\n".join(TikzPrinter.latex_preamble_additions())
    )
    assert g.as_tikz(preamble=True) == result
