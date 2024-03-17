from smallgraphlib import __version__


def test_version():
    version = __version__.split(".")
    assert len(version) == 3
