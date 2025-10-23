"""Pure functional programming utilities.

This module contains functional programming helpers extracted from
scripts/deforum_helpers/rendering/util/fun_utils.py, utils.py, and generate.py,
following functional programming principles with no side effects.
"""

import collections.abc
from functools import reduce
from itertools import chain, tee
from typing import Callable, TypeVar, Any, Dict, List, Iterable
from PIL import Image

# ============================================================================
# FUNCTIONAL PROGRAMMING HELPERS
# ============================================================================

T = TypeVar('T')


def flat_map(func: Callable, iterable):
    """Apply function to each element in iterable and flatten results.

    Args:
        func: Function to apply to each element
        iterable: Iterable to process

    Returns:
        Flattened iterator of results
    """
    # Convert to list to avoid consuming iterator during check
    mapped_list = list(map(func, iterable))

    # Check if any results are iterable (but not strings)
    has_iterables = any(
        isinstance(item, collections.abc.Iterable) and not isinstance(item, (str, bytes))
        for item in mapped_list
    )

    if has_iterables:
        return chain.from_iterable(mapped_list)
    else:
        return iter(mapped_list)


def tube(*funcs: Callable[[T], T], is_do_process: Callable[[], bool] = lambda: True) -> Callable[[T], T]:
    """Compose functions into a pipeline with conditional execution.

    Creates a pipeline that threads a value through a sequence of functions,
    with an optional predicate to skip processing.

    Args:
        *funcs: Functions to compose (applied left to right)
        is_do_process: Predicate function to determine if processing should occur

    Returns:
        Composed function that applies all functions in sequence

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> pipeline = tube(add_one, double)
        >>> pipeline(5)  # (5 + 1) * 2 = 12
        12
    """
    return lambda value: reduce(lambda x, f: f(x) if is_do_process() else x, funcs, value)


# ============================================================================
# DICTIONARY UTILITIES
# ============================================================================


def put_all(dictionaries: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """Add key-value pair to all dictionaries in list.

    Args:
        dictionaries: List of dictionaries to update
        key: Key to add
        value: Value to associate with key

    Returns:
        New list of dictionaries with key-value added (original dicts unchanged)
    """
    return list(map(lambda d: {**d, key: value}, dictionaries))


def put_if_present(dictionary: Dict[str, Any], key: str, value: Any | None) -> None:
    """Conditionally add key-value pair to dictionary if value is not None.

    Args:
        dictionary: Dictionary to potentially update (modified in-place)
        key: Key to add
        value: Value to associate with key (only added if not None)
    """
    if value is not None:
        dictionary[key] = value


# ============================================================================
# CONDITIONAL EXECUTION UTILITIES
# ============================================================================


def _call_or_use(callable_or_value: Callable | Any) -> Any:
    """Call if callable, otherwise return the value.

    Args:
        callable_or_value: Either a callable to invoke or a value to return

    Returns:
        Result of calling callable or the value itself
    """
    return callable_or_value() if callable(callable_or_value) else callable_or_value


def call_or_use_on_cond(condition: bool, callable_or_value: Callable | Any) -> Any | None:
    """Conditionally call or use value based on condition.

    Args:
        condition: Whether to evaluate the callable_or_value
        callable_or_value: Either a callable to invoke or a value to return

    Returns:
        Result if condition is True, None otherwise
    """
    return _call_or_use(callable_or_value) if condition else None


# ============================================================================
# IMAGE AND RANDOM UTILITIES
# ============================================================================


def create_img(dimensions: tuple[int, int]) -> Image.Image:
    """Create binary PIL image with given dimensions.

    Args:
        dimensions: (width, height) tuple

    Returns:
        PIL Image in mode '1' (binary) filled with white (1)
    """
    return Image.new('1', dimensions, 1)


def generate_random_seed() -> int:
    """Generate random seed in valid range [0, 2^32-1].

    Returns:
        Random integer suitable for use as a seed
    """
    import random
    return random.randint(0, 2 ** 32 - 1)


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Return successive overlapping pairs from iterable.

    This is a backport of itertools.pairwise() from Python 3.10+.
    Uses itertools.tee to efficiently create pairs without consuming
    the entire iterable in memory.

    Args:
        iterable: Any iterable to pair up

    Returns:
        Iterator of successive pairs (a, b), (b, c), (c, d), ...

    Examples:
        >>> list(pairwise([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4)]
        >>> list(pairwise('ABCD'))
        [('A', 'B'), ('B', 'C'), ('C', 'D')]
        >>> list(pairwise([1]))
        []
        >>> list(pairwise([]))
        []

    Note:
        This function is equivalent to Python 3.10's itertools.pairwise().
        Once Python 3.10+ is required, this can be replaced with the
        standard library implementation.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
