# -*- coding: utf-8 -*-
import numpy as np


def weighted_score(g: np.ndarray, w_vector: np.ndarray = None) -> float:
    """ Computes a weighted average based on the given weights to transform the multi-objective problem into a single-
    value optimization.

    Args:
        g: objective vector G = [solution_accuracy, solution_size, solution_latency]
        w_vector: vector containing all three weights in the same order than the objective vector.

    Returns:
        w_metric: weighted metric.
    """
    assert len(g) == 3, ValueError("Solution vector must have length 3")
    if w_vector is None:
        return np.sum(g)
    else:
        assert len(w_vector) == 3, ValueError("Weights vector must have length 3")
        return np.sum(np.multiply(g, w_vector))


def general_distance(x: np.ndarray, z: np.ndarray) -> float:
    """ Measures the distance from the solution to the Pareto-Front defined by Z = {z_1, z_2, ... , Z_X}.

    Args:
        x: input search space of the current solution.
        z: Pareto-Front vector, which contains the same number of dimensions than the input search space vector.

    Returns:
        dg: distance between X and Z.
    """
    assert len(x) == len(z), ValueError(
        "Both input search space and Pareto-Front vector must have the same length"
    )
    gd = (1 / len(x)) * np.power(np.sum(np.power(x - z, 2)), 0.5)
    return gd


def hypervolume(g: np.ndarray, optimal_values: np.ndarray = np.array([1, 3000, 200])) -> float:
    """ Measures the percentage between the current solution and the ideal situation where accuracy is 100%, model's
    size is 0, and latency is 0 as well.

    Args:
        g: objective vector G = [solution_accuracy, solution_size, solution_latency].
        optimal_values: optimal values for each metric inside the objective vector.

    Returns:
        hv: percentage of the hypervolume covered by the solution. Is it covers the 100%, means that the solution is
            perfect.
    """
    assert len(g) == 3, ValueError(
        "Both solution and weights vector must have length 3"
    )
    assert 0 <= g[0] <= 1, ValueError("Accuracy must be a number between 0 and 1")
    acc_volume = g[0] / optimal_values[0]
    size_volume = 1 - g[1] / optimal_values[1]
    latency_volume = 1 - g[2] / optimal_values[2]
    return acc_volume * size_volume * latency_volume
