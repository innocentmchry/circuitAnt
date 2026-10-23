"""Constants and lists for semantic plan cache."""

# Opposite-word pairs: if request and template disagree on any pair, treat as a cache miss.
OPPOSITE_PAIRS = [
    ("max", "min"),
    ("maximum", "minimum"),
    ("increase", "decrease"),
    ("with", "without"),
    ("count", "list"),
    ("how many", "list"),
    ("fewest", "most"),
    ("fanin", "fanout"),
    ("input", "output"),
    ("before", "after"),
    ("greater", "less"),
    ("above", "below"),
    ("source", "sink"),
]

# Tools that mutate the design in OpenDB.
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

# Known gate types and logic families in circuitAnt.
GATE_TYPES = {
    "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUF", "NOT",
    "AOI21", "AOI22", "AOI211", "AOI221", "AOI222", "AOI33",
    "OAI21", "OAI22", "OAI211", "OAI221", "OAI222", "OAI33",
    "MUX2", "FA", "HA", "DFF",
}

# Class keywords used in depth/path tools that should not be masked as signals
CLASS_KEYWORDS = {
    "ANY_PI", "ANY_PO", "ANY_DFF_Q", "ANY_DFF_D", "ANY_START", "ANY_END",
    "ANY_SIGNAL", "ANY_GATE_OUTPUT",
}
