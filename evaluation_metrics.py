# -*- coding: utf-8 -*-
import numpy as np


def weighted_score(g: np.ndarray, w_vector: np.ndarray) -> float:
    """ Computes a weighted average based on the given weights to transform the multi-objective problem into a single-
    value optimization.

    Args:
        g: objective vector G = [solution_accuracy, solution_size, solution_latency]
        w_vector: vector containing all three weights in the same order than the objective vector.

    Returns:
        w_metric: weighted metric.
    """
    assert len(g) == len(w_vector) == 3, ValueError(
        "Both solution and weights vector must have length 3"
    )
    return np.sum(np.multiply(g, w_vector))


def general_distance(x: np.ndarray, z: np.ndarray, plus: bool = False) -> float:
    """ Measures the distance from the solution to the Pareto-Front defined by Z = {z_1, z_2, ... , Z_X}.

    Args:
        x: input search space of the current solution.
        z: Pareto-Front vector, which contains the same number of dimensions than the input search space vector.
        plus: if True, computers the function plus where the distance between each coordinate is clipped to the maximum
            between 0 and the difference between both values (x_i and z_i).

    Returns:
        dg: distance between X and Z.
    """
    assert len(x) == len(z), ValueError(
        "Both input search space and Pareto-Front vector must have the same length"
    )
    if plus:
        gd = (1 / len(x)) * np.power(np.sum(np.power(np.max([x - z, 0]), 2)), 0.5)
    else:
        gd = (1 / len(x)) * np.power(np.sum(np.power(x - z, 2)), 0.5)
    return gd


def hypervolume(g: np.ndarray) -> float:
    """ Measures the percentage between the current solution and the ideal situation where accuracy is 100%, model's
    size is 0, and latency is 0 as well.

    Args:
        g: objective vector G = [solution_accuracy, solution_size, solution_latency]

    Returns:
        hv: percentage of the hypervolume covered by the solution. Is it covers the 100%, means that the solution is
            perfect.
    """
    assert len(g) == 3, ValueError(
        "Both solution and weights vector must have length 3"
    )
    assert 0 <= g[0] <= 1, ValueError("Accuracy must be a number between 0 and 1")
    acc_volume = g[0] / 1
    size_volume = 1 - g[1] / 3000
    latency_volume = 1 - g[2] / 200
    return acc_volume * size_volume * latency_volume
