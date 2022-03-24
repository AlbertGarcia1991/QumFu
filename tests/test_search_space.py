# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from scipy.stats import chisquare
from search_space import (
    choice,
    float_normal,
    float_random,
    integer_normal,
    integer_random,
    transform_exp_base_2,
    transform_exp_base_10,
    transform_log_base_10,
)


class TestSearchSpace(TestCase):
    # aux functions
    def test_transform_exp_base_2_output(self):
        self.assertEqual(first=2, second=transform_exp_base_2(x=1))
        self.assertEqual(first=4, second=transform_exp_base_2(x=2))
        self.assertEqual(first=8, second=transform_exp_base_2(x=3))
        self.assertEqual(first=1024, second=transform_exp_base_2(x=10))

    def test_transform_exp_base_10_output(self):
        self.assertEqual(first=10, second=transform_exp_base_10(x=1))
        self.assertEqual(first=100, second=transform_exp_base_10(x=2))
        self.assertEqual(first=1000, second=transform_exp_base_10(x=3))
        self.assertEqual(first=10000000000, second=transform_exp_base_10(x=10))

    def test_transform_log_base_10_output(self):
        self.assertEqual(first=np.log10(1), second=transform_log_base_10(x=1))
        self.assertEqual(first=np.log10(2), second=transform_log_base_10(x=2))
        self.assertEqual(first=np.log10(3), second=transform_log_base_10(x=3))
        self.assertEqual(first=np.log10(10), second=transform_log_base_10(x=10))

    # integer_random
    # TODO

    # float_random
    # TODO

    # integer_normal
    def test_integer_normal_gaussian_distribution(self):
        drawn_samples = []
        for i in range(1000):
            drawn_samples.append(integer_normal(mu=0, sigma=1))
        _, pvalue = chisquare(drawn_samples)
        self.assertGreaterEqual(
            a=pvalue, b=0.05
        )  # Chi-Square test to reject Null hypothesis if p-value > 0.05
        self.assertLess(a=np.mean(drawn_samples), b=1e-1)
        self.assertLess(a=np.std(drawn_samples) - 1, b=1e-1)

    def test_integer_normal_returned_type(self):
        self.assertEqual(first=int, second=type(integer_normal(mu=0, sigma=1)))

    def test_integer_normal_gaussian_distribution_lambda(self):
        drawn_samples = []
        for i in range(10000):
            drawn_samples.append(
                integer_normal(mu=2, sigma=1, lambda_function=transform_exp_base_2)
            )
        self.assertLess(a=np.mean(drawn_samples) - transform_exp_base_2(2), b=1e-1)

    # float_normal
    def test_float_normal_gaussian_distribution(self):
        drawn_samples = []
        for i in range(1000):
            drawn_samples.append(float_normal(mu=0, sigma=1))
        _, pvalue = chisquare(drawn_samples)
        self.assertGreaterEqual(a=pvalue, b=0.05)
        self.assertLess(a=np.mean(drawn_samples), b=1e-1)
        self.assertLess(a=np.std(drawn_samples) - 1, b=1e-1)

    def test_float_normal_returned_type(self):
        self.assertEqual(first=float, second=type(float_normal(mu=0, sigma=1)))

    # TODO(AGP): Investigate why not passing while the same function for integer_normal does pass
    def test_float_normal_gaussian_distribution_lambda(self):
        drawn_samples = []
        for i in range(10000):
            drawn_samples.append(
                float_normal(mu=2, sigma=1, lambda_function=transform_exp_base_2)
            )
        self.assertLess(a=np.mean(drawn_samples) - transform_exp_base_2(2), b=1e-1)

    # choice
    def test_choice_in_options(self):
        self.assertIn("A", container=choice(options=["A"]))

    def test_choice_unitary_return(self):
        self.assertTrue(len("Test") == len(choice(options=["Test"])))

    def test_choice_uniform_distribution(self):
        drawn_samples = []
        for i in range(10000):
            drawn_samples.append(choice(options=[1, 2]))
        _, counts = np.unique(drawn_samples, return_counts=True)
        for c in counts:
            self.assertLess(a=abs(c / 10000 - 0.5), b=1e-1)
        drawn_samples = []
        for i in range(10000):
            drawn_samples.append(choice(options=[1, 2, 3]))
        _, counts = np.unique(drawn_samples, return_counts=True)
        for c in counts:
            self.assertLess(a=abs(c / 10000 - 0.33), b=1e-1)

    def test_choice_no_casting(self):
        self.assertEqual(first=np.str_, second=type(choice(options=["B"])))
        self.assertEqual(first=np.int32, second=type(choice(options=[1])))
        self.assertEqual(first=np.float64, second=type(choice(options=[1.0])))
        self.assertEqual(first=np.bool_, second=type(choice(options=[True])))
