import math
from typing import Iterable

from smallgraphlib.basic_graphs import Graph

from smallgraphlib.core import AbstractGraph
from smallgraphlib.custom_types import Node

COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "magenta",
    "cyan",
    "violet",
    "brown",
    "pink",
    "yellow",
    "purple",
    "gray",
    "black",
    "silver",
    "sienna",
    "white",
    "maroon",
    "beige",
    "salmon",
    "olive",
]


def latex_Dijkstra(graph: AbstractGraph[Node], start: Node, end: Node = None) -> str:
    """Generate the LaTeX code of a table corresponding to Dijkstra algorithm's steps.

    If `end` is `None`, only stop when the shorter path to all other nodes have been found.
    """
    nodes = graph.nodes
    num_cols = len(nodes) + 2
    lines: list[str] = [
        rf"\begin{{tabular}}{{|*{num_cols}{{c |}}}}\cline{{2-{num_cols}}}",
        r"\multicolumn{1}{c|}{} & "
        + " & ".join(sorted(f"${node}$" for node in nodes))
        + r" & Selected\\\hline",
    ]
    # Nodes which have been already visited, but still not archived:
    # visited: dict[Node, tuple[float, list[Node]]] = {start: (0, [start])}
    # format: {node: [distance from start, [previous node, alternative previous node, ...]]}

    previous_nodes: dict[Node, set[Node]] = {node: set() for node in graph.nodes}
    distance_from_start: dict[Node, float] = {node: math.inf for node in graph.nodes}
    being_processed: set[Node] = {start}
    completed: set[Node] = set()
    distance_from_start[start] = 0

    # Nodes which will not change anymore (shorter path from start has been found):
    # archived: dict[Node, tuple[float, list[Node]]] = {}

    def cell_content(node: Node) -> str:
        """Used to print the node in the table."""
        dist = distance_from_start[node]
        previous = previous_nodes[node]
        if dist == math.inf:
            return r"$+\infty$"
        if node in completed:
            return r"\cellcolor{lightgray}"
        printing = str(dist)
        if node != start:
            printing += f" $({','.join(sorted(str(node) for node in previous))})$"
        if node == current:
            printing = rf"\cellcolor{{blue!20}}\textbf{{{printing}}}"
        return printing

    current: Node
    first_cell = r"\text{start}"
    while being_processed and end != (
        current := min(being_processed, key=(lambda n: distance_from_start[n]))
    ):
        lines.append(
            f"${first_cell}$ & "
            + " & ".join(cell_content(node) for node in nodes)
            + rf" & {current} {cell_content(current)}\\\hline"
        )
        first_cell = str(current)
        being_processed.remove(current)
        completed.add(current)
        # We update the distances
        for neighbor in graph.successors(current):
            if neighbor not in completed:
                being_processed.add(neighbor)
                # best distance found until now between neighbor and start
                current_distance = distance_from_start[neighbor]
                # new distance found using current node:
                new_distance = distance_from_start[current] + graph.weight(current, neighbor)
                # replace with new distance only if better
                if new_distance < current_distance:
                    distance_from_start[neighbor] = new_distance
                    previous_nodes[neighbor] = {current}
                elif new_distance == current_distance:
                    previous_nodes[neighbor].add(current)

    lines.append("\\end{tabular}")
    lines.append("")

    def shortest_paths() -> str:
        paths_in_construction = [[target]]
        completed_paths = []
        while paths_in_construction:
            partial_path = paths_in_construction.pop()
            assert len(partial_path) > 0
            for previous in previous_nodes[partial_path[0]]:
                if previous == start:
                    completed_paths.append([previous] + partial_path)
                else:
                    paths_in_construction.append([previous] + partial_path)

        def path_to_str(path: Iterable[Node]):
            return "-".join(str(node) for node in path)

        return ",".join(path_to_str(path) for path in completed_paths)

    targets = list(nodes) if end is None else [end]

    for target in targets:
        if target != start:
            distance = distance_from_start[target]
            lines.append(
                f"Shorter(s) path(s) from ${start}$ to ${target}$: "
                f"${shortest_paths()}$ (length: {distance})."
            )
            lines.append("")

    return "\n" + "\n".join(lines)


def latex_WelshPowell(graph: Graph[Node]) -> str:
    """Generate the LaTeX code of a table ordering nodes by degrees and attributing each a color."""
    nodes: list[str] = []
    degrees: list[str] = []
    colors: list[str] = []
    for node, color_num in graph.greedy_coloring.items():
        nodes.append(node)
        degrees.append(str(graph.all_degrees[node]))
        colors.append(COLORS[color_num] if color_num < len(COLORS) else str(color_num))
    return "\n".join(
        [
            r"\begin{tabular}{|l|*{" + str(graph.order) + "}{c|}}",
            r"\hline",
            r"\cellcolor{blue!10} nodes & " + " & ".join(nodes) + r"\\",
            r"\hline",
            r"\cellcolor{blue!10} degrees & " + " & ".join(degrees) + r"\\",
            r"\hline",
            r"\cellcolor{blue!10} colors & " + " & ".join(colors) + r"\\",
            r"\hline",
            r"\end{tabular}",
        ]
    )
