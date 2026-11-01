import re
from typing import Optional, Tuple

from .constants import OPPOSITE_PAIRS


def check_negative_constraints(req_text: str, candidate_text: str) -> Tuple[bool, Optional[str]]:
    """Check opposite-word pairs between incoming request and candidate request."""
    req_lower = req_text.lower()
    cand_lower = candidate_text.lower()

    for w1, w2 in OPPOSITE_PAIRS:
        r_w1 = bool(re.search(rf"\b{re.escape(w1)}\b", req_lower))
        r_w2 = bool(re.search(rf"\b{re.escape(w2)}\b", req_lower))
        c_w1 = bool(re.search(rf"\b{re.escape(w1)}\b", cand_lower))
        c_w2 = bool(re.search(rf"\b{re.escape(w2)}\b", cand_lower))

        # Direct polarity clash (one has w1, the other has w2)
        if (r_w1 and c_w2) or (r_w2 and c_w1):
            return False, f"Opposite word mismatch: '{w1}' vs '{w2}'"

        # Disagreement when one text specifies a polarity/action and the other specifies its alternative
        if (r_w1 != c_w1) and (r_w2 or c_w2):
            return False, f"Opposite word disagreement on pair ('{w1}', '{w2}')"

    return True, None
