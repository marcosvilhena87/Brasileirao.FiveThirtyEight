"""Teste de instalacao e importacao do pacote."""

from brasileirao538 import __version__


def test_version_is_defined():
    assert __version__ == "0.1.0"
