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
        # T = ⎡T₁₁ T₁₂ T₁₃⎤
        #     ⎜T₂₁ T₂₂ T₂₃⎟
        #     ⎣T₃₁ T₃₂ T₃₃⎦
        # P = (x y (1-x-y))
        # P × T = P <=> ⎧ T₁₁x + T₂₁y + T₃₁(1-x-y) = x
        #               ⎩ T₁₂x + T₂₂y + T₃₂(1-x-y) = y
        #               (third equation isn't necessary)
        #           <=> ⎧ (T₁₁ - T₃₁ - 1)x + (T₂₁ - T₃₁)y + T₃₁ = 0
        #               ⎩ (T₁₂ - T₃₂)x + (T₂₂ - T₃₂ - 1)y + T₃₂ = 0
        # M = ⎡(T₁₁ - T₃₁ - 1) (T₂₁ - T₃₁)   T₃₁⎤
        #     ⎣(T₁₂ - T₃₂)   (T₂₂ - T₃₂ - 1) T₃₂⎦
        M = [[T[i][j] - T[n - 1][j] - (i == j) for i in range(n - 1)] + [T[n - 1][j]] for j in range(n - 1)]
        gaussian_elimination(M)
        # Gaussian elimination:
        # M' = ⎡1 0 M'₃₁⎤
        #      ⎣0 1 M'₃₂⎦
        # x = -M'₃₁
        # y = -M'₃₂
        # z = 1 - (x + y)
        solutions = [-M[i][n - 1] for i in range(n - 1)]
        return tuple(solutions) + (1 - sum(solutions),)
