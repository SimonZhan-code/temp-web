"""Small temporal-formula helpers used by LIBERO-Max LTL specs.

This module is intentionally limited to tokenization and formula inspection.
Runtime checking is handled by :mod:`libero.ltl_monitor.monitor`.
"""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[()!&|]")


def tokenize(formula: str) -> list[str]:
    """Tokenize the LTL subset used by LIBERO-Max task specs."""

    return _TOKEN_RE.findall(formula)


TEMPORAL_OPERATORS = {"F", "G", "X", "U"}
BOOLEAN_OPERATORS = {"!", "&", "|", "(", ")"}
CONSTANTS = {"true", "false"}


def is_atomic_token(token: str) -> bool:
    """Return whether ``token`` should name an atomic proposition."""

    return (
        token not in TEMPORAL_OPERATORS
        and token not in BOOLEAN_OPERATORS
        and token.lower() not in CONSTANTS
    )
