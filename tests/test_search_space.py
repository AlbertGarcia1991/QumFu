# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
import pytest
from scipy.stats import chisquare
from search_space import (
    choice,
    float_normal,
    float_random,
    integer_normal,
    integer_random,
    transform_exp_2,
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

    def test_transform_transform_exp_2_output(self):
        self.assertEqual(first=np.power(1, 2), second=transform_exp_2(x=1))
        self.assertEqual(first=np.power(2, 2), second=transform_exp_2(x=2))
        self.assertEqual(first=np.power(3, 2), second=transform_exp_2(x=3))
        self.assertEqual(first=np.power(10, 2), second=transform_exp_2(x=10))

    # integer_random
    def test_integer_random_uniform_zero_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = integer_random(lower_bound=0, upper_bound=10)
            self.assertIn(member=drawn_value, container=range(0, 10 + 1))

    def test_integer_random_uniform_positive_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = integer_random(lower_bound=5, upper_bound=10)
            self.assertIn(member=drawn_value, container=range(5, 10 + 1))

    def test_integer_random_uniform_negative_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = integer_random(lower_bound=-5, upper_bound=10)
            self.assertIn(member=drawn_value, container=range(-5, 10 + 1))

    def test_integer_random_uniform_negative_to_negative_bounds(self):
        for i in range(1000):
            drawn_value = integer_random(lower_bound=-5, upper_bound=-1)
            self.assertIn(member=drawn_value, container=range(-5, -1 + 1))

    def test_integer_random_uniform_negative_to_zero_bounds(self):
        for i in range(1000):
            drawn_value = integer_random(lower_bound=-5, upper_bound=0)
            self.assertIn(member=drawn_value, container=range(-5, 0 + 1))

    def test_integer_random_uniform_bottom_bound_exclusive_upper_bound_exclusive(self):
        for i in range(1000):
            drawn_value = integer_random(
                lower_bound=1,
                upper_bound=3,
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
            self.assertEqual(first=drawn_value, second=2)

    def test_integer_random_uniform_bottom_bound_exclusive_upper_bound_inclusive(self):
        for i in range(1000):
            drawn_value = integer_random(
                lower_bound=1,
                upper_bound=3,
                lower_bound_inclusive=False,
                upper_bound_inclusive=True,
            )
            self.assertIn(member=drawn_value, container=[2, 3])

    def test_integer_random_uniform_bottom_bound_inclusive_upper_bound_exclusive(self):
        for i in range(1000):
            drawn_value = integer_random(
                lower_bound=1,
                upper_bound=3,
                lower_bound_inclusive=True,
                upper_bound_inclusive=False,
            )
            self.assertIn(member=drawn_value, container=[1, 2])

    def test_integer_random_uniform_bottom_bound_inclusive_upper_bound_inclusive(self):
        for i in range(1000):
            drawn_value = integer_random(
                lower_bound=1,
                upper_bound=3,
                lower_bound_inclusive=True,
                upper_bound_inclusive=True,
            )
            self.assertIn(member=drawn_value, container=[1, 2, 3])

    def test_integer_random_returned_type(self):
        self.assertEqual(
            first=int, second=type(integer_random(lower_bound=0, upper_bound=10))
        )

    def test_integer_random_types(self):
        try:
            integer_random(
                lower_bound="A",
                upper_bound=1,
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
        except TypeError as e:
            self.assertIsInstance(obj=e, cls=TypeError)
        try:
            integer_random(
                lower_bound=1,
                upper_bound="B",
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
        except TypeError as e:
            self.assertIsInstance(obj=e, cls=TypeError)
        try:
            integer_random(
                lower_bound=0,
                upper_bound=1,
                lower_bound_inclusive=5,
                upper_bound_inclusive=False,
            )
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            integer_random(
                lower_bound=0,
                upper_bound=1,
                lower_bound_inclusive=True,
                upper_bound_inclusive="C",
            )
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            integer_random(lower_bound=10, upper_bound=1)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)

    # float_random
    def test_float_random_uniform_zero_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = float_random(lower_bound=0, upper_bound=10)
            self.assertGreaterEqual(a=drawn_value, b=0)
            self.assertLessEqual(a=drawn_value, b=10)

    def test_float_random_uniform_positive_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = float_random(lower_bound=5, upper_bound=10)
            self.assertGreaterEqual(a=drawn_value, b=5)
            self.assertLessEqual(a=drawn_value, b=10)

    def test_float_random_uniform_negative_to_positive_bounds(self):
        for i in range(1000):
            drawn_value = float_random(lower_bound=-5, upper_bound=10)
            self.assertGreaterEqual(a=drawn_value, b=-5)
            self.assertLessEqual(a=drawn_value, b=10)

    def test_float_random_uniform_negative_to_negative_bounds(self):
        for i in range(1000):
            drawn_value = float_random(lower_bound=-5, upper_bound=-1)
            self.assertGreaterEqual(a=drawn_value, b=-5)
            self.assertLessEqual(a=drawn_value, b=-1)

    def test_float_random_uniform_negative_to_zero_bounds(self):
        for i in range(1000):
            drawn_value = float_random(lower_bound=-5, upper_bound=0)
            self.assertGreaterEqual(a=drawn_value, b=-5)
            self.assertLessEqual(a=drawn_value, b=0)

    def test_float_random_uniform_bottom_bound_exclusive_upper_bound_exclusive(self):
        for i in range(1000):
            drawn_value = float_random(
                lower_bound=1.00005,
                upper_bound=1.0001,
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
            self.assertGreater(a=drawn_value, b=1.00005)
            self.assertLess(a=drawn_value, b=1.0001)

    def test_float_random_uniform_bottom_bound_exclusive_upper_bound_inclusive(self):
        for i in range(1000):
            drawn_value = float_random(
                lower_bound=1.001,
                upper_bound=1.005,
                lower_bound_inclusive=False,
                upper_bound_inclusive=True,
            )
            self.assertGreater(a=drawn_value, b=1.001)
            self.assertLessEqual(a=drawn_value, b=1.005)

    def test_float_random_uniform_bottom_bound_inclusive_upper_bound_exclusive(self):
        for i in range(1000):
            drawn_value = float_random(
                lower_bound=1.00005,
                upper_bound=1.0001,
                lower_bound_inclusive=True,
                upper_bound_inclusive=False,
            )
            self.assertGreaterEqual(a=drawn_value, b=1.00005)
            self.assertLess(a=drawn_value, b=1.0001)

    def test_float_random_uniform_bottom_bound_inclusive_upper_bound_inclusive(self):
        for i in range(1000):
            drawn_value = float_random(
                lower_bound=1.00005,
                upper_bound=1.0001,
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
            self.assertGreaterEqual(a=drawn_value, b=1.00005)
            self.assertLessEqual(a=drawn_value, b=1.0001)

    def test_float_random_returned_type(self):
        self.assertEqual(
            first=float, second=type(float_random(lower_bound=0, upper_bound=10))
        )

    def test_float_random_types(self):
        try:
            float_random(
                lower_bound="A",
                upper_bound=1,
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
        except TypeError as e:
            self.assertIsInstance(obj=e, cls=TypeError)
        try:
            float_random(
                lower_bound=1,
                upper_bound="B",
                lower_bound_inclusive=False,
                upper_bound_inclusive=False,
            )
        except TypeError as e:
            self.assertIsInstance(obj=e, cls=TypeError)
        try:
            float_random(
                lower_bound=0,
                upper_bound=1,
                lower_bound_inclusive=5,
                upper_bound_inclusive=False,
            )
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            float_random(
                lower_bound=0,
                upper_bound=1,
                lower_bound_inclusive=True,
                upper_bound_inclusive="C",
            )
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            float_random(lower_bound=10, upper_bound=1)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)

    # integer_normal
    def test_integer_normal_gaussian_distribution(self):
        drawn_samples = []
        pvalue = 0.0
        while pvalue == 0:
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
        self.assertLess(a=abs(np.mean(drawn_samples) - transform_exp_base_2(2)), b=3)

    def test_integer_normal_types(self):
        try:
            integer_normal(mu="A", sigma=1)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            integer_normal(mu=1, sigma="A")
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            integer_normal(mu=1, sigma=1, lambda_function=3)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)

    # float_normal
    def test_float_normal_gaussian_distribution(self):
        drawn_samples = []
        pvalue = 0.0
        while pvalue == 0:
            for i in range(1000):
                drawn_samples.append(float_normal(mu=0, sigma=1))
            _, pvalue = chisquare(drawn_samples)
        self.assertGreaterEqual(a=pvalue, b=0.05)
        self.assertLess(a=np.mean(drawn_samples), b=1e-1)
        self.assertLess(a=np.std(drawn_samples) - 1, b=1e-1)

    def test_float_normal_returned_type(self):
        self.assertEqual(first=float, second=type(float_normal(mu=0, sigma=1)))

    def test_float_normal_gaussian_distribution_lambda(self):
        drawn_samples = []
        for i in range(10000):
            drawn_samples.append(
                float_normal(mu=2, sigma=1, lambda_function=transform_exp_base_2)
            )
        self.assertLess(a=abs(np.mean(drawn_samples) - transform_exp_base_2(2)), b=3)

    def test_float_normal_types(self):
        try:
            float_normal(mu="A", sigma=1)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            float_normal(mu=1, sigma="A")
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)
        try:
            float_normal(mu=1, sigma=1, lambda_function=3)
        except AssertionError as e:
            self.assertIsInstance(obj=e, cls=AssertionError)

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
