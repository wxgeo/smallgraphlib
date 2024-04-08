from smallgraphlib.basic_graphs import DirectedGraph
from smallgraphlib.utilities import cached_property
from smallgraphlib.weighted_graphs import AbstractNumericGraph


def gaussian_elimination(matrix: list[list[float]]) -> None:
    """Apply Gaussian elimination to `matrix`, where matrix is n×(n+1)."""
    n = len(matrix)
    # Make matrix triangular.
    for k in range(n):
        for i in range(k, n):
            if matrix[i][k] != 0:
                matrix[i], matrix[k] = matrix[k], matrix[i]
                break
        else:
            raise ValueError("No single solution.")
        for i in range(k + 1, n):
            coefficient = matrix[i][k] / matrix[k][k]
            for j in range(k, n + 1):
                matrix[i][j] -= coefficient * matrix[k][j]
    # Solve.
    for k in range(n - 1, -1, -1):
        matrix[k][n] /= matrix[k][k]
        matrix[k][k] = 1
        for i in range(k - 1, -1, -1):
            matrix[i][n] -= matrix[i][k] * matrix[k][n]
            matrix[i][k] = 0


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

    @property
    def stable_state(self) -> tuple[float, ...]:
        T = self.transition_matrix
        n = len(T)
        # T = (T11 T12 T13)
        #     (T21 T22 T23)
        #     (T31 T32 T33)
        # P = (x y (1-x-y))
        # PT = P <=> / T11 x + T21 y + T31 (1-x-y) = x
        #            \ T12 x + T22 y + T32 (1-x-y) = x
        #              (third equation isn't necessary)
        #        <=> / (T11-T31-1) x + (T21-T31) y + T31 = 0
        #            \ (T12-T32) x + (T22-T32-1) y + T32 = 0
        # M = ((T11-T31-1) (T21-T31)   T31)
        #     ((T12-T32)   (T22-T32-1) T32)
        M = [[T[i][j] - T[n - 1][j] - (i == j) for i in range(n - 1)] + [T[n - 1][j]] for j in range(n - 1)]
        gaussian_elimination(M)
        # Gaussian elimination:
        # M' = (1 0 M'31)
        #      (0 1 M'32)
        # x = -M'31
        # y = -M'32
        # z = 1 - (x + y)
        solutions = [-M[i][n - 1] for i in range(n - 1)]
        return tuple(solutions) + (1 - sum(solutions),)
