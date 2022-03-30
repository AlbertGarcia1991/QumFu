# -*- coding: utf-8 -*-
import inspect
from typing import List, Union

import numpy as np
from errors import ConditionOperandNotValid, ConditionTermsTypeError
from utils import CONDITIONAL_OPERANDS, SAMPLING_PRECISION


def integer_random(
    lower_bound: int,
    upper_bound: int,
    lambda_function: callable = None,
    lower_bound_inclusive: bool = True,
    upper_bound_inclusive: bool = True,
) -> int:
    """ Returns a sampled value on the integer space from a random distribution.

    Args:
        lower_bound: lowest value to obtain when sampling.
        upper_bound: greatest value to obtain when sampling.
        lambda_function: function to apply to the sampled value before returning its value.
        lower_bound_inclusive: if True, the lower bound will be an option to be sampled. Otherwise, it will be omitted.
        upper_bound_inclusive: if True, the upper bound will be an option to be sampled. Otherwise, it will be omitted.

    Returns:
        sampled_value: value obtained from the distribution.
    """
    assert lower_bound < upper_bound, ValueError(
        "The given lower_bound must be smaller than upper_bound"
    )
    assert isinstance(lower_bound, int) and isinstance(upper_bound, int), TypeError(
        "Both given lower_bound and upper_bound must be integers"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must be a callable (function)"
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
        sampled_value = (
            int(np.random.randint(low=lower_bound - 1, high=upper_bound)) + 1
        )
    elif lower_bound_inclusive and not upper_bound_inclusive:
        sampled_value = np.random.randint(low=lower_bound, high=upper_bound, dtype=int)
    elif not lower_bound_inclusive and upper_bound_inclusive:
        sampled_value = -np.random.randint(
            low=-upper_bound, high=-lower_bound, dtype=int
        )
    else:
        while True:
            sampled_value = np.random.randint(
                low=lower_bound, high=upper_bound, dtype=int
            )
            if sampled_value != lower_bound:
                break
    if lambda_function is not None:
        sampled_value = lambda_function(sampled_value)
    return sampled_value


def float_random(
    lower_bound: Union[int, float],
    upper_bound: Union[int, float],
    lambda_function: callable = None,
    lower_bound_inclusive: bool = True,
    upper_bound_inclusive: bool = True,
) -> float:
    """ Returns a sampled value on the float space from a random distribution.

    Args:
        lower_bound: lowest value to obtain when sampling.
        upper_bound: greatest value to obtain when sampling.
        lambda_function: function to apply to the sampled value before returning its value.
        lower_bound_inclusive: if True, the lower bound will be an option to be sampled. Otherwise, it will be omitted.
        upper_bound_inclusive: if True, the upper bound will be an option to be sampled. Otherwise, it will be omitted.

    Returns:
        sampled_value: value obtained from the distribution.
    """
    assert lower_bound < upper_bound, ValueError(
        "The given lower_bound must be smaller than upper_bound"
    )
    assert isinstance(lower_bound, (int, float)) and isinstance(
        upper_bound, (int, float)
    ), TypeError("Both given lower_bound and upper_bound must be integers or floats")
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must be a callable (function)"
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
        sampled_value = np.random.random() * (upper_bound - lower_bound) + lower_bound
        if lower_bound_inclusive and upper_bound_inclusive:
            if lower_bound <= sampled_value <= upper_bound:
                break
        if not lower_bound_inclusive and upper_bound_inclusive:
            if lower_bound < sampled_value <= upper_bound:
                break
        if lower_bound_inclusive and not upper_bound_inclusive:
            if lower_bound <= sampled_value < upper_bound:
                break
        if not lower_bound_inclusive and not upper_bound_inclusive:
            if lower_bound < sampled_value < upper_bound:
                break
    if lambda_function is not None:
        sampled_value = lambda_function(sampled_value)
    sampled_value = round(sampled_value, SAMPLING_PRECISION)
    return sampled_value


def integer_normal(
    mu: Union[int, float] = 0,
    sigma: Union[int, float] = 1,
    lambda_function: callable = None,
) -> int:
    """ Returns a sampled value on the integer space from a Gaussian normal distribution.

    Args:
        mu: mean of the normal distribution.
        sigma: standard deviation of the normal distribution.
        lambda_function: function to apply to the sampled value before returning its value.

    Returns:
        sampled_value: value obtained from the distribution.
    """
    assert isinstance(mu, (int, float)) and isinstance(sigma, (int, float)), TypeError(
        "Both given mu and sigma must be integers or floats"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must be a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    sampled_value = int(np.random.normal(loc=mu, scale=sigma))
    if lambda_function is not None:
        sampled_value = lambda_function(sampled_value)
    return sampled_value


def float_normal(
    mu: Union[int, float] = 0,
    sigma: Union[int, float] = 1,
    lambda_function: callable = None,
) -> int:
    """ Returns a sampled value on the float space from a Gaussian normal distribution.

    Args:
        mu: mean of the normal distribution.
        sigma: standard deviation of the normal distribution.
        lambda_function: function to apply to the sampled value before returning its value.

    Returns:
        sampled_value: value obtained from the distribution.
    """
    assert isinstance(mu, (int, float)) and isinstance(sigma, (int, float)), TypeError(
        "Both given mu and sigma must be integers or floats"
    )
    if lambda_function is not None:
        assert callable(lambda_function), TypeError(
            "The given lambda function must be a callable (function)"
        )
        assert len(inspect.signature(lambda_function).parameters) == 1, ValueError(
            "The given lambda function must contain a single argument"
        )
    sampled_value = float(np.random.normal(loc=mu, scale=sigma))
    if lambda_function is not None:
        sampled_value = lambda_function(sampled_value)
    sampled_value = round(sampled_value, SAMPLING_PRECISION)
    return sampled_value


def choice(
    options: List[Union[int, float, bool, str]], weights: List[float] = None
) -> Union[int, float, bool, str]:
    """ Returns a sampled value randomly (or not if weights are passed) picked from the options given.

    Args:
        options: list of element from where to sample one.
        weights: weights for each option (probability of each one). If not indicated, all options have the same weight
            so selection will be purely random.

    Returns:
        sampled_value: value obtained from the distribution.
    """
    sampled_value = np.random.choice(options, p=weights)
    return sampled_value


def static(
    value: Union[int, float, bool, str, object]
) -> Union[int, float, bool, str, object]:
    """ Placeholder functions to have constants as parameters. It returns the same object passed as argument, without
    any type of modification.

    Args:
        value: value to be returned.

    Returns:
        value: same value than passed as argument.
    """
    return value


class InputValueSpace:
    """ Object that maps variables with its sampling distribution. It is instantiated as the value of a dictionary,
    where its key is the parameter name to store the distribution.

    Args:
        se_type: type of distribution from where values are sampled.
        params_dict: arguments requited to initialize each kind of distribution function (e.g. if distribution is
        'choice' we need to pass as params_dict a dictionary containing at least the key-value pair 'options'-options
        to choice. Optional arguments for each distribution have to be passed in the same key-value format.).

    Example:
        >> InputValueSpace(se_type=integer_normal, options={"mu"=0, "sigma":2, "lambda_function": transform_log_base_10}

        will instantiate an object to sample integer values from a Gaussian normal distribution centered around 0 and
        standard deviation of 2, applying after sampling a value the function transform_log_base_10.
    """

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
    """ Main object to store a seach space. It contains all parameters to be initialized (and from which distribution
    each parameter must be sampled), and keeps the history of different design points searched (everytime we sample a
    new current search space, it appends the previous one to the list attribute named history).

    Examples:
        The following code instantiates a search space for a MLP model with all hidden layers activation function
        as 'relu', a number of hidden hidden layers, a number of units per hidden layer as power of 2 (between 4 and
        1024), and a single choice for both kernel initialization and kernel_regularizer from the given options:

            se_dict = {
                "activation": InputValueSpace(
                    se_type=static, params_dict={"value": "relu"}
                ),
                "n_layers": InputValueSpace(
                    se_type=integer_random, params_dict={"lower_bound": 1, "upper_bound": 4}
                ),
                "units_layer_1": InputValueSpace(
                    se_type=integer_random, params_dict={
                        "lower_bound": 2, "upper_bound": 10, "lambda_function": transform_exp_base_2}
                ),
                "units_layer_2": InputValueSpace(
                    se_type=integer_random, params_dict={
                        "lower_bound": 2, "upper_bound": 10, "lambda_function": transform_exp_base_2}
                ),
                "units_layer_3": InputValueSpace(
                    se_type=integer_random, params_dict={
                        "lower_bound": 2, "upper_bound": 10, "lambda_function": transform_exp_base_2}
                ),
                "units_layer_4": InputValueSpace(
                    se_type=integer_random, params_dict={
                        "lower_bound": 2, "upper_bound": 10, "lambda_function": transform_exp_base_2}
                ),
                "kernel_initializer": InputValueSpace(
                    se_type=choice, params_dict={
                        "options": ["glorot_uniform", "glorot_normal", "identity"]
                    }
                ),
                "kernel_regularizer": InputValueSpace(
                    se_type=choice, params_dict={
                        "options": [None, "l1", "l2"]
                    }
                )
            }

            se = SearchSpace(input_dict=se_dict, conditions_dict={"units_layer_2": ["n_layers", "gte", 2],
                                                                  "units_layer_3": ["n_layers", "gte", 3],
                                                                  "units_layer_4": ["n_layers", "gte", 4]})
    """

    def __init__(self, input_dict: dict, conditions_dict: dict = None):
        self.input_dict = input_dict
        self.conditions_dict = conditions_dict

        self.current = dict()
        self.history = []
        self.sample()

    def delete_history(self):
        """ Deletes the history of sampled sets.
        """
        self.history = []

    def sample(self,
               check_repeated_space: bool = True,
               update_history: bool = True,
               overwrite_last_history: bool = False):
        """ Samples a new set of values for all parameters specified when instantiating the object, checking the given
        conditions (if any).

        Args:
            check_repeated_space: If True, it checks that the recent sampled search space has not been explored yet.
            update_history: If True, the new sampled set will be appended to the history attribute.
            overwrite_last_history: If True, the new sampled set will replace the latest sampled set inside history
                attribute.
        """
        if check_repeated_space:
            while not self.current or self.current in self.history:
                self.current = dict()
                self._sample_space()
        else:
            self.current = dict()
            self._sample_space()
        if update_history:
            self.update_history(overwrite_last_history=overwrite_last_history)

    def parse_condition(
        self,
        conditioning: str,
        condition_operand: str,
        conditional_space: Union[int, float, bool, str, list],
    ) -> bool:
        """ Auxiliary method to parse the conditions to enable conditional search space based on the values already
        sampled from other parameters. The current implementation draws a sample from a conditioned parameter
        (InputValueSpace) only if the operation:
            <conditional_parameter>: <conditioning> <condition_operand> <conditional_space>
        is True.

        Examples of these conditions are the following:
            "l1_ratio": "penalty" nin [None, "l1", "l2"]
            "l1_ratio": "penalty" eq "elasticnet"
            "units_layer3": "n_layers" qte 3

        Args:
            conditioning: parameter conditioned to the value of another parameter (conditional) -> RHT on the comparison
            condition_operand: comparison operand to set if the condition is True ot Not.
            conditional_space: value to compare the conditional parameter -> LHT on the comparison.

        """
        assert condition_operand in CONDITIONAL_OPERANDS, ConditionOperandNotValid(
            f"The given condition operand '{condition_operand}' is not valid"
        )
        output = False
        if condition_operand == "eq":
            output = self.current[conditioning] == conditional_space
        elif condition_operand == "neq":
            output = self.current[conditioning] != conditional_space
        elif condition_operand == "lt":
            assert type(self.current[conditioning]) in (int, float) and type(
                conditional_space
            ) in (int, float), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] < conditional_space
        elif condition_operand == "lte":
            assert type(self.current[conditioning]) in (int, float) and type(
                conditional_space
            ) in (int, float), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] <= conditional_space
        elif condition_operand == "gt":
            assert type(self.current[conditioning]) in (int, float) and type(
                conditional_space
            ) in (int, float), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] > conditional_space
        elif condition_operand == "gte":
            assert type(self.current[conditioning]) in (int, float) and type(
                conditional_space
            ) in (int, float), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] >= conditional_space
        elif condition_operand == "in":
            assert isinstance(conditional_space, list), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] in conditional_space
        elif condition_operand == "in":
            assert isinstance(conditional_space, list), ConditionTermsTypeError(
                "The conditioning type is not correct in order to be compared against the conditional space"
            )
            output = self.current[conditioning] not in conditional_space
        return output

    def _sample_space(self):
        """ This is the under-the-hood process for sampling a whole new search space set. It works as follows:
            1) Check which parameters are non-conditioned (always must appear)
            2) Sample each non-conditional parameter from its distribution.
            3) For each conditioned parameter, check the condition happened on the non-conditioned conditioning
                parameter and decide whether or not to sample that paramter.

        TODO: Implement second-order conditioning (the conditioning parameter could be also a conditioned parameter)
        """
        # Get non-conditioned parameters
        if not self.conditions_dict:
            self.non_conditioner_parameters = []
            for key in self.input_dict.keys():
                self.non_conditioner_parameters.append(key)
                self.sample_single_param(parameter=key)
        else:
            self.non_conditioner_parameters = []
            self.conditioner_parameters = []
            for key in self.input_dict.keys():
                if key not in self.conditions_dict.keys():
                    self.non_conditioner_parameters.append(key)
                    self.sample_single_param(parameter=key)
                else:
                    self.conditioner_parameters.append(key)

            # Get first order conditioned parameters
            for key in self.conditioner_parameters:
                conditioning = self.conditions_dict[key][0]
                condition_operand = self.conditions_dict[key][1]
                conditional_space = self.conditions_dict[key][2]
                assert conditioning in self.non_conditioner_parameters, NotImplementedError(
                    "Second order conditions not implemented yet"
                )
                if self.parse_condition(
                    conditioning=conditioning,
                    condition_operand=condition_operand,
                    conditional_space=conditional_space,
                ):
                    self.sample_single_param(parameter=key)

    def sample_single_param(self, parameter: str, update_history: bool = False):
        """ Samples a single parameter (InputValueSpace) and adds it to the current dictionary in the right key-value
        format.

        Args:
            parameter: parameter to sample.
            update_history: If True, the new sampled value will be changes from the latest sampled set stored inside the
                history attribute.
        """
        dist = self.input_dict[parameter]
        if dist.distribution in ["integer_random", "float_random"]:
            self.current[parameter] = dist.se_type(
                lower_bound=dist.params_dict["lower_bound"],
                upper_bound=dist.params_dict["upper_bound"],
                lambda_function=dist.params_dict["lambda_function"],
                lower_bound_inclusive=dist.params_dict["lower_bound_inclusive"],
                upper_bound_inclusive=dist.params_dict["upper_bound_inclusive"],
            )
        elif dist.distribution in ["integer_normal", "float_normal"]:
            self.current[parameter] = dist.se_type(
                mu=dist.params_dict["mu"],
                sigma=dist.params_dict["sigma"],
                lambda_function=dist.params_dict["lambda_function"],
            )
        elif dist.distribution == "choice":
            self.current[parameter] = dist.se_type(options=dist.params_dict["options"])
        elif dist.distribution == "static":
            self.current[parameter] = dist.se_type(value=dist.params_dict["value"])
        else:
            raise AttributeError(
                f"The given distribution '{dist.distribution}' does not exist"
            )
        if update_history:
            self.update_history()

    def update_history(self, overwrite_last_history: bool = False):
        """ Appends the current sampled search space to the history list attribute. It is called everytime we sample
        a new whole search space set by running sample() method.

        Args:
            overwrite_last_history: If True, the new sampled set will replace the latest sampled set inside history
                attribute.
        """
        if overwrite_last_history and len(self.history) > 0:
            self.history[-1] = self.current
        else:
            self.history.append(self.current)


def transform_log_base_10(x: int):
    return np.log10(x)


def transform_exp_base_10(x: int):
    return 10 ** x


def transform_exp_base_2(x: int):
    return 2 ** x


def transform_exp_2(x: int):
    return x ** 2
