from .ldba import LDBA, LDBATransition, SCC
from .ldba_sequence import LDBASequence
from .rabinizer import run_rabinizer
from .hoa_parser import HOAParser


def build_ldba(formula: str, propositions: set[str], possible_assignments: list = None) -> LDBA:
    """Build an LDBA from an LTL formula: rabinizer -> HOA parse -> prune -> complete -> SCCs."""
    hoa = run_rabinizer(formula)
    ldba = HOAParser(formula, hoa, propositions).parse_hoa()
    if possible_assignments:
        ldba.prune(possible_assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


__all__ = ['LDBA', 'LDBATransition', 'SCC', 'LDBASequence', 'build_ldba', 'run_rabinizer', 'HOAParser']
