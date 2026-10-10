"""Transformations of a Design, grouped by topic (the agent's tools call these).

Each public function edits the design in place and returns a dict of counts
for the agent; it does not verify itself (the agent calls check_equivalence).

Rules (besides those in analysis/__init__.py, which apply here too):
  * transformations may use analysis, verification, utils and edit; analysis and
    verification never import transformation, so reading a design never depends
    on code that changes it;
  * OpenDB edits go through eda_engine/edit.py (unique names, port nets keep
    their names, constant nets kept up to date) -- except resynthesis, where
    OpenROAD's BLIF reader replaces gates itself (then Design.scan_constants);
  * a result from analysis or utils.netlist_view is a snapshot: after editing,
    query again instead of reusing it.
"""
from .buffering import insert_buffers_for_fanout, insert_dedicated_buffers_for_loads
from .cleanup import (collapse_inverter_pairs, merge_duplicate_gates, remove_dangling_gates,
                      simplify_gates_with_constant_inputs)
from .naming import rename_gate, rename_wire
from .resynthesis import reduce_depth, remap_gates, replace_gate_type

__all__ = ["rename_gate", "rename_wire", "remove_dangling_gates", "collapse_inverter_pairs",
           "merge_duplicate_gates", "simplify_gates_with_constant_inputs",
           "insert_buffers_for_fanout", "insert_dedicated_buffers_for_loads",
           "reduce_depth", "remap_gates", "replace_gate_type"]
