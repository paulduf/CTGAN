"""Synthesizers module."""

from ctgan.synthesizers.ctgan import CTGAN
from ctgan.synthesizers.tvae import TVAE, FTVAE

__all__ = ("CTGAN", "TVAE", "FTVAE")


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
