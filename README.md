# Small Graph Lib

## Installing

### Basic installation

Smallgraphlib is on PyPI, so you can download and install it with pip,
as any usual python package.
```bash
pip install smallgraphlib
```

### Development version

For development, please clone the last version from Github:
```bash
git clone https://github.com/wxgeo/smallgraphlib
```

You will then need `uv` to manage the development process.
If needed, install it on Linux or Mac with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

[More information on `uv` installation](https://docs.astral.sh/uv/getting-started/installation/)

It is also recommended to install `just` command runner, to execute
the recipes of the `justfile`.
```bash
uv tool install rust-just
```

## Usage

Main classes are `Graph`, `DirectedGraph`, `WeightedGraph` and `WeightedDirectedGraph`:
```python
>>> from smallgraphlib import DirectedGraph
>>> g = DirectedGraph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
>>> g.is_simple
True
>>> g.is_complete
False
>>> g.is_directed
True
>>> g.adjacency_matrix
[[0, 1, 0], [1, 0, 1], [0, 0, 0]]
>>> g.degree
3
>>> g.order
3
>>> g.is_eulerian
False
>>> g.is_semi_eulerian
True
```

Special graphs may be generated using factory functions:
```python
>>> from smallgraphlib import complete_graph, complete_bipartite_graph
>>> K5 = complete_graph(5)
>>> len(K5.greedy_coloring)
5
>>> K33 = complete_bipartite_graph(3, 3)
>>> K33.degree
6
>>> K33.diameter
2
```

If the graph is not too complex, Tikz code may be generated:
```python
>>> g.as_tikz()
...
```

## Development

1. [Get development version and `uv`](#development-version)

2. Install dependencies including development tools:
```bash
uv sync
```

3. Optionally, update dependencies and development tools:
```bash
uv sync --upgrade
```

4. Optionally, install library in editable mode:
```bash
uv pip install -e smallgraphlib
```

5. Make changes, add tests.

6. Launch tests:
```bash
just test
```

7. Everything OK? Commit. :)
