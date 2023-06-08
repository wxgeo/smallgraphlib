import math

from smallgraphlib import WeightedDirectedGraph


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
    from smallgraphlib.latex_export import latex_Dijkstra

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
