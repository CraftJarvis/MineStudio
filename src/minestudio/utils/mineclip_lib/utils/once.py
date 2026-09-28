from __future__ import annotations
import functools
import warnings
from typing import Literal


def meta_decorator(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = (
        lambda args, kwargs: len(args) == 1 and len(kwargs) == 0 and callable(args[0])
    )

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def call_once(func, on_second_call: Literal["noop", "raise", "warn"] = "noop"):
    """
    Decorator to ensure that a function is only called once.

    Args:
      on_second_call (str): what happens when the function is called a second time.
    """
    assert on_second_call in [
        "noop",
        "raise",
        "warn",
    ], "mode must be one of 'noop', 'raise', 'warn'"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if wrapper._called:
            if on_second_call == "raise":
                raise RuntimeError(
                    f"{func.__name__} has already been called. Can only call once."
                )
            elif on_second_call == "warn":
                warnings.warn(
                    f"{func.__name__} has already been called. Should only call once."
                )
        else:
            wrapper._called = True
            return func(*args, **kwargs)

    wrapper._called = False
    return wrapper
