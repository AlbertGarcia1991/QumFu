# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from evaluation_metrics import hypervolume, weighted_score


class TestEvaluationMetrics(TestCase):
    # weighted_score
    def test_weighted_score_N_values(self):
        input_array = np.array([1, 2, 4])
        input_weights = np.array([0.5, 0.1, 0.4])
        expected_output = 0
        for idx in range(len(input_array)):
            expected_output += input_array[idx] * input_weights[idx]
        self.assertEqual(
            first=weighted_score(g=input_array, w_vector=input_weights),
            second=expected_output,
        )

    # hypervolume
    def test_hypervolume(self):
        input_array = np.array([0.75, 1000, 100])
        expected_value = 0.75 * (1 - 1000 / 3000) * (1 - 100 / 200)
        self.assertEqual(first=hypervolume(g=input_array), second=expected_value)

    def test_hypervolume_acc_above_1(self):
        input_array = np.array([1.5, 1000, 100])
        try:
            hypervolume(g=input_array)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)

    def test_hypervolume_acc_below_0(self):
        input_array = np.array([-4, 1000, 100])
        try:
            hypervolume(g=input_array)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
