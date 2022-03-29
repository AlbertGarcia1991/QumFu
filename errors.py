# -*- coding: utf-8 -*-

# search_space errors
class ConditionOperandNotValid(Exception):
    """Exception raised when the given condition operand is not correct"""

    pass


class ConditionTermsTypeError(Exception):
    """Exception raised when the given condition compares not valid types"""

    pass
