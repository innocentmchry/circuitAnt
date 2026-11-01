"""Constants and domain definitions for EDA semantic tool-call caching."""

# Opposite-word pairs: if request and candidate disagree on any pair, reject candidate.
OPPOSITE_PAIRS = [
    ("max", "min"),
    ("maximum", "minimum"),
    ("highest", "lowest"),
    ("deepest", "shallowest"),
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
    ("longest", "shortest"),
    ("read", "modify"),
    ("add", "remove"),
]
