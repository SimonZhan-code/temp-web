"""Subprocess wrapper for the Rabinizer4 ``ltl2ldba`` binary.

Ported from Neuralsym-VLA/ltl_benchmark/automata/rabinizer.py. Rabinizer is
optional at import time; ``run_rabinizer`` is only called by the live fallback
in ``builder.build_ldba`` when no cached HOA is found. Set ``RABINIZER_PATH``
to the absolute path of the ``ltl2ldba`` executable.
"""

from __future__ import annotations

import os
import subprocess


RABINIZER_PATH = os.environ.get("RABINIZER_PATH", "rabinizer4/bin/ltl2ldba")


def run_rabinizer(formula: str) -> str:
    """Convert an LTL formula to an LDBA in HOA format.

    Flags: ``-p`` keep proposition names; ``-d`` deterministic where possible;
    ``-e`` allow epsilon transitions.
    """

    command = [RABINIZER_PATH, "-i", formula, "-p", "-d", "-e"]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode != 0 or run.stderr:
        raise RuntimeError(
            f'Rabinizer call `{" ".join(command)}` resulted in an error.\n'
            f"Error: {run.stderr}."
        )
    return run.stdout
