"""
General-purpose decorators used across the mht package.
"""
from __future__ import annotations

import functools
import os
import sys
import time


def pandas_filter(func=None, *, arguments: dict = None):
    """
    Post-filter the return value of *func* on a DataFrame column == value.

    Parameters
    ----------
    arguments : dict
        Must contain keys ``'column'`` and ``'value'``.  When empty or
        ``None`` the return value is passed through unchanged.
    """
    if arguments is None:
        arguments = {}

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            value = f(*args, **kwargs)
            if arguments:
                value = value[value[arguments['column']] == arguments['value']]
            return value
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def pseudo_private(func=None, *, hide: bool = False):
    """
    Optionally suppress stdout during the decorated function's execution.

    Parameters
    ----------
    hide : bool
        When ``True``, redirect stdout to /dev/null for the duration of
        the call.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if hide:
                sys.stdout = open(os.devnull, 'w')
                try:
                    result = f(*args, **kwargs)
                finally:
                    sys.stdout = sys.__stdout__
            else:
                result = f(*args, **kwargs)
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def private_timer(func):
    """Print the execution time (in milliseconds) of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Terminated in {round(1000 * (time.time() - start))} milliseconds')
        return result
    return wrapper


def running_decorator(func):
    """Print the function name, then execution time (in seconds)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f'Running {func.__name__}')
        result = func(*args, **kwargs)
        print(f'Completed in {round(time.time() - start, 2)} seconds')
        return result
    return wrapper


def ignore_unhashable(func):
    """
    Allow a ``@functools.cache``-decorated function to fall back to its
    uncached version when called with unhashable arguments.

    Wrap in this order::

        @ignore_unhashable
        @cache
        def my_function(...): ...
    """
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')

    @functools.wraps(func, assigned=attributes)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            if 'unhashable type' in str(exc):
                return uncached(*args, **kwargs)
            raise

    wrapper.__uncached__ = uncached
    return wrapper
