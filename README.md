# Small Graph Lib

## Installing

    $ git clone https://github.com/wxgeo/smallgraphlib

    $ pip install --user smallgraphlib

## Usage

Main classes are `Graph`, `DirectedGraph`, `WeightedGraph` and `WeightedDirectedGraph`:

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

Special graphs may be generated using factory functions:
    
    >>> from smallgraphlib import complete_graph, complete_bipartite_graph
    >>> K5 = complete_graph(5)
    >>> len(K5.greedy_coloring)
    5
    >>> K33 = complete_bipartite_graph(3, 3)
    >>> K33.degree
    6
    >>> K33.diameter
    2
    
If the graph is not to complex, Tikz code may be generated:

    >>> g.as_tikz()
    ...

## Development

1. Get last version:
   
       $ git clone https://github.com/wxgeo/smallgraphlib

2. Install Poetry.
    
   Poetry is a tool for dependency management and packaging in Python.

   Installation instructions are here:
   https://python-poetry.org/docs/#installation

3. Install developments tools:
    
       $ poetry install

4. Optionally, update development tools:
      
       $ poetry update

5. Optionally, install library in editable mode:

       $ pip install -e smallgraphlib

6. Make changes, add tests.
  
7. Launch tests:

        $ tox

8. Everything's OK ? Commit. :)