"""Queries on a Design, grouped by topic (the agent's tools call these).

Where a new query goes: the file whose concept it shares; if none fits, a new
topic file (even for one function) -- never a misc/helpers file.  Results are
plain dicts.  Queries that list things show the first `inline_limit` items and
write the full list to a file; inline_limit=None returns the complete list and
writes no file (for code that uses the result, e.g. a transformation).

Rules for what a file contains:
  * public functions are queries -- each answers a complete question, called by
    an agent tool;
  * a helper used by one query is nested inside it, even if it is long;
  * a helper used by several files goes in eda_engine/utils.py, and only once it
    really has more than one user;
  * no `_private` module-level functions.
"""
from .cones import get_cone
from .depth import get_endpoint_stats, get_max_depth
from .expressions import derive_boolean_equation
from .fanout import get_fanout
from .inventory import count_gates, find_gates, get_ports
from .paths import find_all_paths, find_cut_points
from .patterns import find_duplicates, find_enable_hold, find_inverter_pairs

__all__ = ["count_gates", "find_gates", "get_ports", "get_max_depth", "get_endpoint_stats",
           "find_all_paths", "find_cut_points", "get_cone", "get_fanout",
           "find_inverter_pairs", "find_duplicates", "find_enable_hold",
           "derive_boolean_equation"]
