from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.utilities import cached_property
from smallgraphlib.weighted_graphs import AbstractNumericGraph


class MarkovChain(AbstractNumericGraph, DirectedGraph):
    """Markov Chain implementation."""

    # TODO:
    # - verify that there are no parallel edges
    # - verify that each edge value is a probability
    # - complete missing values if needed (only one missing value per node allowed,
    #   and missing value must be between 0 and 1).
    # - verify that the sum of probabilities for each node is 1

    def probability(self, node1, node2) -> float:
        labels = self._labels.get((node1, node2), [0])
        assert len(labels) == 1
        probability = labels[0]
        try:
            f = float(probability)
        except ValueError:
            raise TypeError(
                f"Incorrect type: {probability!r} between nodes {node1} and {node2} is of type {type(probability)!r},"
                " which can't be converted to float."
            )
        if not 0 <= f <= 1:
            raise ValueError(
                f"Incorrect value for a probability: {probability} between nodes {node1} and {node2}."
            )
        return probability

    @cached_property
    def transition_matrix(self):
        return tuple(tuple(self.probability(node1, node2) for node2 in self.nodes) for node1 in self.nodes)
