from abc import abstractmethod
from collections import Counter
from functools import wraps
from typing import Protocol


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
    """Decorator to indicate that a function must invalidate the cache."""

    @wraps(f)
    def cached_f(self=None, *args, **kw):
        result = f(self, *args, **kw)
        try:
            self._cache.clear()
        except AttributeError:
            self._cache = {}
        return result

    return cached_f


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
        ...
        ValueError: Multiset value can't be negative for key 'b'.

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
        if not isinstance(value, int):
            raise ValueError(f"Multiset value must be an integer, not {value!r}.")
        super().__setitem__(key, value)
        self._clean_key(key)

    def total(self) -> int:
        return sum(self.values())


class CycleFoundError(AttributeError):
    pass


class ComparableAndHashable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...
