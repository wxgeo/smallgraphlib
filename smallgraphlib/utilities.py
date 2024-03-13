from abc import abstractmethod
from collections import Counter
from functools import wraps
from typing import Protocol, Generator


def cached(f):
    """Decorator used to cache results in `self._cache` dictionary."""

    @wraps(f)
    def cached_f(self=None, *args, **kw):
        key = f.__name__
        try:
            if key in self._cache:
                return self._cache[key]
        except AttributeError:
            self._cache = {}
        result = f(self, *args, **kw)
        self._cache[key] = result
        return result

    return cached_f


def cached_property(f):
    return property(cached(f))


def clear_cache(f):
    """Decorator to indicate that a method must invalidate the class cache when called.

    If the class have a method `on_clear_cache()`, it will be called too.
    """

    @wraps(f)
    def cached_f(self=None, *args, **kw):
        result = f(self, *args, **kw)
        try:
            self._cache.clear()
        except AttributeError:
            self._cache = {}
        try:
            self.on_clear_cache()
        except AttributeError:
            pass
        return result

    return cached_f


# class HasInternalCache:
#     def put_in_cache(self, key, value):
#         try:
#             cache = self._cache
#         except AttributeError:
#             # noinspection PyAttributeOutsideInit
#             cache = self._cache = {}
#         cache[key] = value
#
#     def get_from_cache(self, key):
#         return self._cache[key]
#
#     def clear_cache(self):
#         self._cache.clear()


class Multiset(Counter):
    """
    A `collections.Counter` subclass that only accept positive values,
    and automatically removes empty keys (i.e. keys for which the count is zero).

        >>> s = Multiset(("a", "a", "b"))
        >>> s
        Multiset({'a': 2, 'b': 1})
        >>> s["b"] -= 1
        >>> s
        Multiset({'a': 2})
        >>> s["b"] -= 1
        Traceback (most recent call last):
        ValueError: Multiset value must be a positive integer, not -1.

    Trying to set negative values will raise a ValueError.

    Like `Counter`, `Multiset` is a `dict` subclass, so you may use dict methods on it.

    If you access it using the `dict` interface, be careful when iterating over keys.
    Modifying values when iterating over keys is unsafe, since if a value becomes zero,
    the corresponding key will be automatically removed.

    So, if you plan to modify values when iterating, you should use
    `list(_.keys())` or `list(_.items())` instead of `_.keys()` or `_.items()`.
    """

    def __init__(self, iterable=None, /, **kwds):
        super().__init__(iterable, **kwds)
        for key in list(self):
            self._clean_key(key)

    def _clean_key(self, key):
        if self[key] == 0:
            del self[key]
        elif self[key] < 0:
            raise ValueError(f"Multiset value can't be negative for key {key!r}.")

    def __setitem__(self, key, value):
        if not isinstance(value, int) or value < 0:
            if int(value) == value and int(value) >= 0:
                value = int(value)
            else:
                raise ValueError(f"Multiset value must be a positive integer, not {value!r}.")
        super().__setitem__(key, value)
        self._clean_key(key)

    def total(self) -> int:
        return sum(self.values())


class ComparableAndHashable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...


def frange(start: float, end: float, step: float = 1) -> Generator[float, None, None]:
    """Generate float values from `start` (included) to `end` (excluded) with an increment of `step`."""
    count = start
    while count < end:
        yield count
        count += step


def set_repr(obj: object) -> str:
    """Represent sets and (unnested) frozensets in a deterministic way, by sorting elements.

    >>> set_repr({4, 3, 2})
    '{2, 3, 4}'
    >>> set_repr(frozenset({3, 2, 4}))
    '{2, 3, 4}'

    For any other objects, it just calls python default repr().
    >>> set_repr(45.)
    '45.0'
    """
    if isinstance(obj, (set, frozenset)):
        return f"{{{', '.join(set_repr(elt) for elt in sorted(obj))}}}"
    return repr(obj)
