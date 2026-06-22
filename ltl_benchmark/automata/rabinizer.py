import os
import subprocess

RABINIZER_PATH = os.environ.get('RABINIZER_PATH', 'rabinizer4/bin/ltl2ldba')


def run_rabinizer(formula: str) -> str:
    """Convert an LTL formula to a LDBA in the HOA format."""
    command = [RABINIZER_PATH, '-i', formula, '-p', '-d', '-e']
    run = subprocess.run(command, capture_output=True, text=True)
    if run.stderr != '':
        raise RuntimeError(f'Rabinizer call `{" ".join(command)}` resulted in an error.\nError: {run.stderr}.')
    return run.stdout
