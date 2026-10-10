"""Functional checks on a Design, proved with ABC `cec` (blif.abc_cec).

Same file rules as the analysis package: public functions are queries called by
agent tools; single-use helpers are nested; shared helpers live in utils.py.
"""
from .equivalence import check_equivalence
from .signals import check_signal_equivalence, check_symmetry

__all__ = ["check_equivalence", "check_signal_equivalence", "check_symmetry"]
