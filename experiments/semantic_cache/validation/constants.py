"""Constants of the regex extraction and template binding (experiments E4-E6)."""
from agent.cache.validation.constants import OPPOSITE_PAIRS  # noqa: F401  (re-exported)

# Mutating tools that alter design netlist state.
MUTATING_TOOLS = {
    "rename_gate",
    "rename_wire",
    "remove_dangling_gates",
    "collapse_inverter_pairs",
    "merge_duplicate_gates",
    "simplify_gates_with_constant_inputs",
    "insert_buffers_for_fanout",
    "insert_dedicated_buffers_for_loads",
    "reduce_depth",
    "remap_gates",
    "replace_gate_type",
}

# Known logic gate types in circuitAnt / OpenDB.
GATE_TYPES = {
    "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUF", "NOT",
    "AOI21", "AOI22", "AOI211", "AOI221", "AOI222", "AOI33",
    "OAI21", "OAI22", "OAI211", "OAI221", "OAI222", "OAI33",
    "MUX2", "FA", "HA", "DFF",
}

# Keywords for endpoint/path queries that should not be extracted as netlist-specific wire names.
CLASS_KEYWORDS = {
    "ANY_PI", "ANY_PO", "ANY_DFF_Q", "ANY_DFF_D", "ANY_START", "ANY_END",
    "ANY_SIGNAL", "ANY_GATE_OUTPUT",
}
