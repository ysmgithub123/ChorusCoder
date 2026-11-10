# Original Copyright 2021 Microsoft under MIT License.
# From https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/codeutils/keywords/keywordlist.py

import keyword
from functools import lru_cache
from typing import FrozenSet

__all__ = ['get_language_keywords']

@lru_cache()
def get_language_keywords(language: str) -> FrozenSet[str]:
    """Python evaluation keywords; boolean literals remain identifiers."""
    if language.lower() != 'python':
        raise ValueError('The paper benchmarks use Python')
    return frozenset(k for k in keyword.kwlist if k not in ('True', 'False'))
