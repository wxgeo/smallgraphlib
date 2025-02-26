import pytest

from smallgraphlib.utilities import Multiset, latexify


def test_Multiset():
    with pytest.raises(ValueError) as _:
        # Negative counts are not allowed.
        Multiset({"a": 2, "b": 4, "c": 4, "d": 0, "e": -1})
    s = Multiset({"a": 2, "b": 4, "c": 4, "d": 0})
    # Automatically remove key 'd', since its count is zero.
    assert set(s) == {"a", "b", "c"}
    s["a"] -= 1
    assert set(s) == {"a", "b", "c"}
    s["a"] -= 1
    assert set(s) == {"b", "c"}
    with pytest.raises(ValueError) as _:
        s["a"] -= 1


def test_latexify():
    assert latexify("") == ""
    assert latexify("", default=r"$\varnothing$") == r"$\varnothing$"
    assert latexify("lambda") == r"$\lambda$"
    assert latexify("lambdas") == r"$lambdas$"
    assert latexify("Gamma10") == r"$\Gamma_{10}$"
    assert latexify("-oo") == r"$-\infty$"
    assert latexify("oo") == r"$\infty$"
    assert latexify(float("-inf")) == r"$-\infty$"
    assert latexify(float("inf")) == r"$\infty$"
    # If it already starts with "\", don't insert another one.
    assert latexify(r"\Sigma") == r"$\Sigma$"
    assert latexify(r"\infty") == r"$\infty$"
