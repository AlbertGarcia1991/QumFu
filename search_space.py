# -*- coding: utf-8 -*-
import inspect
from typing import List, Union

import numpy as np
from utils import SAMPLING_PRECISION, SAMPLING_TOOL


def integer_random(
    lower_bound: int,
    upper_bound: int,
    lambda_function: callable = None,
    lower_bound_inclusive: bool = True,
    upper_bound_inclusive: bool = True,
) -> int:
    assert lower_bound < upper_bound, ValueError(
        "The given lower_bound must be smaller than upper_bound"
    )
    assert isinstance(lower_bound, int) and isinstance(upper_bound, int), TypeError(
        "Both given lower_bound and upper_bound must be integers"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must bre a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    assert isinstance(lower_bound_inclusive, bool) and isinstance(
        upper_bound_inclusive, bool
    ), TypeError(
        "Both given lower_bound_inclusive and upper_bound_inclusive must be Booleans"
    )
    if lower_bound_inclusive and upper_bound_inclusive:
        drawn_value = int(
            np.random.randint(low=lower_bound, high=upper_bound) + SAMPLING_TOOL
        )
    elif lower_bound_inclusive and not upper_bound_inclusive:
        drawn_value = np.random.randint(low=lower_bound, high=upper_bound, dtype=int)
    elif not lower_bound_inclusive and upper_bound_inclusive:
        drawn_value = -np.random.randint(low=-upper_bound, high=-lower_bound, dtype=int)
    else:
        while True:
            drawn_value = np.random.randint(
                low=lower_bound, high=upper_bound, dtype=int
            )
            if drawn_value != lower_bound:
                break
    if lambda_function is not None:
        drawn_value = lambda_function(drawn_value)
    return drawn_value


def float_random(
    lower_bound: Union[int, float],
    upper_bound: Union[int, float],
    lambda_function: callable = None,
    lower_bound_inclusive: bool = True,
    upper_bound_inclusive: bool = True,
) -> float:
    assert lower_bound < upper_bound, ValueError(
        "The given lower_bound must be smaller than upper_bound"
    )
    assert isinstance(lower_bound, (int, float)) and isinstance(
        upper_bound, (int, float)
    ), TypeError("Both given lower_bound and upper_bound must be integers or floats")
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must bre a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    assert isinstance(lower_bound_inclusive, bool) and isinstance(
        upper_bound_inclusive, bool
    ), TypeError(
        "Both given lower_bound_inclusive and upper_bound_inclusive must be Booleans"
    )
    while True:
        drawn_value = np.random.random() * (upper_bound - lower_bound) + lower_bound
        if lower_bound_inclusive and upper_bound_inclusive:
            if lower_bound <= drawn_value <= upper_bound:
                break
        if not lower_bound_inclusive and upper_bound_inclusive:
            if lower_bound < drawn_value <= upper_bound:
                break
        if lower_bound_inclusive and not upper_bound_inclusive:
            if lower_bound <= drawn_value < upper_bound:
                break
        if not lower_bound_inclusive and not upper_bound_inclusive:
            if lower_bound < drawn_value < upper_bound:
                break
    if lambda_function is not None:
        drawn_value = lambda_function(drawn_value)
    drawn_value = round(drawn_value, SAMPLING_PRECISION)
    return drawn_value


def integer_normal(
    mu: Union[int, float] = 0,
    sigma: Union[int, float] = 1,
    lambda_function: callable = None,
) -> int:
    assert isinstance(mu, (int, float)) and isinstance(sigma, (int, float)), TypeError(
        "Both given mu and sigma must be integers or floats"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must bre a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    drawn_value = int(np.random.normal(loc=mu, scale=sigma))
    if lambda_function is not None:
        drawn_value = lambda_function(drawn_value)
    return drawn_value


def float_normal(
    mu: Union[int, float] = 0,
    sigma: Union[int, float] = 1,
    lambda_function: callable = None,
) -> int:
    assert isinstance(mu, (int, float)) and isinstance(sigma, (int, float)), TypeError(
        "Both given mu and sigma must be integers or floats"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must bre a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    drawn_value = float(np.random.normal(loc=mu, scale=sigma))
    if lambda_function is not None:
        drawn_value = lambda_function(drawn_value)
    drawn_value = round(drawn_value, SAMPLING_PRECISION)
    return drawn_value


def choice(options: List[Union[int, float, bool, str]]) -> Union[int, float, bool, str]:
    drawn_value = np.random.choice(options)
    return drawn_value


class InputValueSpace:
    def __init__(self, se_type: callable, params_dict: dict):
        self.se_type = se_type
        self.distribution = self.se_type.__name__
        self.params_dict = params_dict
        self.initialize_defaults()

    def initialize_defaults(self):
        for arg in list(inspect.signature(self.se_type).parameters.values()):
            if arg.name not in self.params_dict.keys():
                self.params_dict[arg.name] = arg.default
                assert not isinstance(
                    self.params_dict[arg.name], inspect._empty
                ), ValueError(
                    f"For callable '{self.se_type.__name__}' is required the argument '{arg.name}'"
                )

    def __repr__(self):
        return f"Distribution: {self.distribution}. Parameters:{self.params_dict}"


class SearchSpace:
    def __init__(self, input_dict: dict):
        self.input_dict = input_dict
        self.drawn_history = None
        self.draw()

    def init_drawn_current_dict(self):
        self.drawn_current = dict()

    def draw(self):
        self.init_drawn_current_dict()
        for parameter, dist in self.input_dict.items():
            if dist.distribution in ["integer_random", "float_random"]:
                self.drawn_current[parameter] = dist.se_type(
                    lower_bound=dist.params_dict["lower_bound"],
                    upper_bound=dist.params_dict["upper_bound"],
                    lambda_function=dist.params_dict["lambda_function"],
                    lower_bound_inclusive=dist.params_dict["lower_bound_inclusive"],
                    upper_bound_inclusive=dist.params_dict["upper_bound_inclusive"],
                )
            elif dist.distribution in ["integer_normal", "float_normal"]:
                self.drawn_current[parameter] = dist.se_type(
                    mu=dist.params_dict["mu"],
                    sigma=dist.params_dict["sigma"],
                    lambda_function=dist.params_dict["lambda_function"],
                )
            elif dist.distribution == "choice":
                self.drawn_current[parameter] = dist.se_type(
                    options=dist.params_dict["options"]
                )
            else:
                raise AttributeError(
                    f"The given distribution '{dist.distribution}' does not exist"
                )


def transform_log_base_10(x: int):
    return np.log10(x)


def transform_exp_base_10(x: int):
    return 10 ** x


def transform_exp_base_2(x: int):
    return 2 ** x


def transform_exp_2(x: int):
    return x ** 2


"""
se_dict = {
    "integer_random_log": InputValueSpace(
        se_type=integer_random, params_dict={"lower_bound": 1, "upper_bound": 5, "lambda_function": transform_log}),
}

se = SearchSpace(input_dict=se_dict)


"Cases" for conditional scapes
"""
