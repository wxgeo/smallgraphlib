import math

from smallgraphlib import WeightedDirectedGraph, Graph, LabeledDirectedGraph
from smallgraphlib.latex_export import latex_Dijkstra, latex_WelshPowell, latex_degrees_table


def test_latex_dijstra():
    oo = math.inf
    M = [
        [0, 16, oo, oo, 3, 15, 10],
        [oo, 0, 1, 4, oo, oo, oo],
        [oo, 1, 0, oo, oo, oo, oo],
        [oo, 4, 2, 0, oo, oo, oo],
        [oo, 13, 17, 6, 0, oo, oo],
        [15, 2, oo, oo, oo, 0, 3],
        [oo, oo, oo, oo, oo, 4, 0],
    ]
    g = WeightedDirectedGraph.from_matrix(M, nodes_names="ABCDEFG")

    assert (
        latex_Dijkstra(g, "A")
        == r"""
\begin{tabular}{|*9{c |}}\cline{2-9}
\multicolumn{1}{c|}{} & $A$ & $B$ & $C$ & $D$ & $E$ & $F$ & $G$ & Selected\\\hline
$\text{start}$ & \cellcolor{blue!20}\textbf{0} & $+\infty$ & $+\infty$ & $+\infty$ & $+\infty$ & $+\infty$ & $+\infty$ & A \cellcolor{blue!20}\textbf{0}\\\hline
$A$ & \cellcolor{lightgray} & 16 $(A)$ & $+\infty$ & $+\infty$ & \cellcolor{blue!20}\textbf{3 $(A)$} & 15 $(A)$ & 10 $(A)$ & E \cellcolor{blue!20}\textbf{3 $(A)$}\\\hline
$E$ & \cellcolor{lightgray} & 16 $(A,E)$ & 20 $(E)$ & \cellcolor{blue!20}\textbf{9 $(E)$} & \cellcolor{lightgray} & 15 $(A)$ & 10 $(A)$ & D \cellcolor{blue!20}\textbf{9 $(E)$}\\\hline
$D$ & \cellcolor{lightgray} & 13 $(D)$ & 11 $(D)$ & \cellcolor{lightgray} & \cellcolor{lightgray} & 15 $(A)$ & \cellcolor{blue!20}\textbf{10 $(A)$} & G \cellcolor{blue!20}\textbf{10 $(A)$}\\\hline
$G$ & \cellcolor{lightgray} & 13 $(D)$ & \cellcolor{blue!20}\textbf{11 $(D)$} & \cellcolor{lightgray} & \cellcolor{lightgray} & 14 $(G)$ & \cellcolor{lightgray} & C \cellcolor{blue!20}\textbf{11 $(D)$}\\\hline
$C$ & \cellcolor{lightgray} & \cellcolor{blue!20}\textbf{12 $(C)$} & \cellcolor{lightgray} & \cellcolor{lightgray} & \cellcolor{lightgray} & 14 $(G)$ & \cellcolor{lightgray} & B \cellcolor{blue!20}\textbf{12 $(C)$}\\\hline
$B$ & \cellcolor{lightgray} & \cellcolor{lightgray} & \cellcolor{lightgray} & \cellcolor{lightgray} & \cellcolor{lightgray} & \cellcolor{blue!20}\textbf{14 $(G)$} & \cellcolor{lightgray} & F \cellcolor{blue!20}\textbf{14 $(G)$}\\\hline
\end{tabular}

Shorter(s) path(s) from $A$ to $B$: $A-E-D-C-B$ (length: 12).

Shorter(s) path(s) from $A$ to $C$: $A-E-D-C$ (length: 11).

Shorter(s) path(s) from $A$ to $D$: $A-E-D$ (length: 9).

Shorter(s) path(s) from $A$ to $E$: $A-E$ (length: 3).

Shorter(s) path(s) from $A$ to $F$: $A-G-F$ (length: 14).

Shorter(s) path(s) from $A$ to $G$: $A-G$ (length: 10).
"""
    )


def test_latex_welsh_powell():
    g = Graph.from_string("A:B,D,E,F B:C,D,E,F,G C:E,G D:F,G E:G F:G G")
    assert latex_WelshPowell(g) == (
        "\\begin{tabular}{|l|*{7}{c|}}\n"
        "    \\hline\n"
        "    \\cellcolor{blue!10} nodes & $B$ & $G$ & $A$ & $D$ & $E$ & $F$ & $C$\\\\\n"
        "    \\hline\n"
        "    \\cellcolor{blue!10} degrees & 6 & 5 & 4 & 4 & 4 & 4 & 3\\\\\n"
        "    \\hline\n"
        "    \\cellcolor{blue!10} colors & red & blue & blue & green & green & orange & orange\\\\\n"
        "    \\hline\n"
        "\\end{tabular}\n"
    )


def test_latex_welsh_powell2():
    g = Graph.from_subgraphs("P1,P7 P2,P3,P4,P8 P5,P6,P8 P1,P4,P5 P3,P7 P6,P3")
    result = r"""\begin{tabular}{|l|*{8}{c|}}
    \hline
    \cellcolor{blue!10} nodes & $P_{3}$ & $P_{4}$ & $P_{8}$ & $P_{5}$ & $P_{1}$ & $P_{2}$ & $P_{6}$ & $P_{7}$\\
    \hline
    \cellcolor{blue!10} degrees & 5 & 5 & 5 & 4 & 3 & 3 & 3 & 2\\
    \hline
    \cellcolor{blue!10} colors & red & blue & green & red & green & orange & blue & blue\\
    \hline
\end{tabular}
"""
    assert latex_WelshPowell(g) == result


def test_latex_degrees_table():
    g = LabeledDirectedGraph.from_string("s5:s1=a1 s2:s1=a2 s4:s5=a3 s4:s2=a4 s3:s2=a5 s3:s3=a6 s1")
    result = r"""\begin{tabular}{|l|*{5}{c|}}
    \hline
    \cellcolor{blue!10} nodes & $s_{1}$ & $s_{2}$ & $s_{3}$ & $s_{4}$ & $s_{5}$\\
    \hline
    \cellcolor{blue!10} in degrees & 2 & 2 & 1 & 0 & 1\\
    \hline
    \cellcolor{blue!10} out degrees & 0 & 1 & 2 & 2 & 1\\
    \hline
\end{tabular}
"""
    assert latex_degrees_table(g) == result
