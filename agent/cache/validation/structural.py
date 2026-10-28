import copy
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..entry import ToolCall
from .constants import CLASS_KEYWORDS, GATE_TYPES, MUTATING_TOOLS, OPPOSITE_PAIRS
from .extractor import ExtractedEntities, extract_entities


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


class StructuralValidator:
    """GenCache-inspired structural template creation, parameter extraction, and schema validation."""

    def __init__(self):
        # Known valid tool argument types from circuitAnt tools.py
        self.tool_arg_types = {
            "set_testcase_name": {"name": str},
            "read_design": {"path": str},
            "write_design": {"path": str},
            "count_gates": {},
            "get_max_depth": {"source": str, "sink": str},
            "get_endpoint_stats": {"sink": str, "metric": str, "operator": str, "threshold": int},
            "get_cone": {"signal": str, "direction": str, "shared_with": str},
            "get_fanout": {"signal": str},
            "find_gates": {"gate_type": str, "connected_to": str, "pin": str, "name": str},
            "get_ports": {"direction": str},
            "find_cut_points": {"source": str, "sink": str, "signal": str},
            "find_inverter_pairs": {},
            "find_duplicates": {},
            "find_enable_hold": {},
            "derive_boolean_equation": {"signal": str},
            "check_equivalence": {},
            "check_signal_equivalence": {"a": str, "b": str},
            "check_symmetry": {"signal": str, "input_a": str, "input_b": str},
            "find_all_paths": {"source": str, "sink": str, "avoid": str, "max_depth": int},
            "rename_gate": {"old_name": str, "new_name": str},
            "rename_wire": {"old_name": str, "new_name": str},
            "remove_dangling_gates": {},
            "collapse_inverter_pairs": {},
            "merge_duplicate_gates": {},
            "simplify_gates_with_constant_inputs": {"gate_type": str},
            "insert_buffers_for_fanout": {"max_fanout": int, "net_name": str},
            "insert_dedicated_buffers_for_loads": {"net_name": str},
            "reduce_depth": {},
            "remap_gates": {"allowed": str, "signal": str},
            "replace_gate_type": {"gate_type": str, "allowed": str, "signal": str},
        }

    def validate_schema(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if tool_name not in self.tool_arg_types:
            return False, f"Unknown tool: {tool_name}"
        expected_params = self.tool_arg_types[tool_name]
        for k, v in args.items():
            if k not in expected_params:
                return False, f"Unexpected argument '{k}' for tool '{tool_name}'"
            expected_type = expected_params[k]
            if v is not None and not isinstance(v, expected_type):
                # Try soft type conversion (e.g. int vs str)
                try:
                    args[k] = expected_type(v)
                except (ValueError, TypeError):
                    return False, f"Argument '{k}' of tool '{tool_name}' must be of type {expected_type.__name__}, got {type(v).__name__}"
        return True, None

    def instantiate_template(
        self,
        candidate_calls: List[ToolCall],
        cand_request: str,
        new_request: str,
        testcase_name: str = "",
    ) -> Tuple[bool, Optional[List[ToolCall]], Optional[str], Dict[str, Any]]:
        """Instantiate tool-call template by binding parameters from the new request.
        
        Returns: (success, bound_calls, reason, extracted_params)
        """
        # 1. Negative constraint check
        pass_neg, neg_reason = check_negative_constraints(new_request, cand_request)
        if not pass_neg:
            return False, None, neg_reason, {}

        # 2. Extract entities from candidate and new request
        cand_entities = extract_entities(cand_request)
        new_entities = extract_entities(new_request)

        bound_calls: List[ToolCall] = []
        extracted_params: Dict[str, Any] = {}
        sig_idx = 0
        gate_idx = 0

        for orig_call in candidate_calls:
            tool_name = orig_call.tool
            new_args = copy.deepcopy(orig_call.args)

            # Special case: set_testcase_name
            if tool_name == "set_testcase_name":
                if testcase_name:
                    new_args["name"] = testcase_name
                    extracted_params["testcase_name"] = testcase_name
                elif new_entities.paths:
                    m = re.search(r"(test\d+)", new_request)
                    if m:
                        new_args["name"] = m.group(1)
                        extracted_params["testcase_name"] = m.group(1)

            # Special case: read_design
            elif tool_name == "read_design":
                if new_entities.paths:
                    p = os.path.normpath(new_entities.paths[0]).replace("\\", "/")
                    new_args["path"] = p
                    extracted_params["path"] = p
                elif testcase_name:
                    new_args["path"] = f"testcase/{testcase_name}/{testcase_name}.v"
                    extracted_params["path"] = new_args["path"]

            # Special case: write_design
            elif tool_name == "write_design":
                if new_entities.paths:
                    p = os.path.normpath(new_entities.paths[0]).replace("\\", "/")
                    new_args["path"] = p
                    extracted_params["path"] = p
                elif testcase_name:
                    new_args["path"] = f"{testcase_name}_out.v"
                    extracted_params["path"] = new_args["path"]

            # Special check for register-to-register path constraints in get_max_depth / find_all_paths
            if tool_name in ("get_max_depth", "find_all_paths"):
                is_cand_reg = bool(re.search(r"\b(register|reg-to-reg|dff|flip-flop)\b", cand_request, re.IGNORECASE))
                is_new_reg = bool(re.search(r"\b(register|reg-to-reg|dff|flip-flop)\b", new_request, re.IGNORECASE))
                if is_cand_reg != is_new_reg:
                    return False, None, "Mismatch on register-to-register path constraint", {}

            # General tools with gate_type (e.g. find_gates, simplify_gates_with_constant_inputs, replace_gate_type)
            if "gate_type" in new_args:
                if new_entities.gate_types:
                    new_args["gate_type"] = new_entities.gate_types[0]
                    extracted_params["gate_type"] = new_entities.gate_types[0]
                elif orig_call.args.get("gate_type") and not new_entities.gate_types:
                    return False, None, "Candidate requires gate_type, but new request specifies no gate type", {}

            # Tools with check_symmetry (takes 3 signals: signal, input_a, input_b)
            if tool_name == "check_symmetry":
                if len(new_entities.signals) >= 3:
                    new_args["signal"] = new_entities.signals[0]
                    new_args["input_a"] = new_entities.signals[1]
                    new_args["input_b"] = new_entities.signals[2]
                    extracted_params["signal"] = new_args["signal"]
                    extracted_params["input_a"] = new_args["input_a"]
                    extracted_params["input_b"] = new_args["input_b"]
                else:
                    return False, None, "check_symmetry requires 3 signals, insufficient signals in request", {}

            # General tools with signal / net_name (use sequential entity consumption)
            elif "signal" in new_args:
                orig_sig = orig_call.args.get("signal")
                if orig_sig not in CLASS_KEYWORDS:
                    if sig_idx < len(new_entities.signals):
                        new_args["signal"] = new_entities.signals[sig_idx]
                        extracted_params[f"signal_{sig_idx}"] = new_entities.signals[sig_idx]
                        sig_idx += 1
                    else:
                        return False, None, "Candidate requires specific signal, but request has insufficient signals", {}

            if "net_name" in new_args:
                if sig_idx < len(new_entities.signals):
                    new_args["net_name"] = new_entities.signals[sig_idx]
                    extracted_params[f"net_{sig_idx}"] = new_entities.signals[sig_idx]
                    sig_idx += 1
                elif new_entities.signals:
                    new_args["net_name"] = new_entities.signals[0]
                    extracted_params["net_name"] = new_entities.signals[0]
                else:
                    return False, None, "Candidate requires net_name, but new request has no signal", {}

            # Tools with direction (get_cone, get_ports)
            if "direction" in new_args:
                if new_entities.directions:
                    new_args["direction"] = new_entities.directions[0]
                    extracted_params["direction"] = new_entities.directions[0]

            # Tools with gate instance name (find_gates(name="..."))
            if "name" in new_args and tool_name == "find_gates":
                if gate_idx < len(new_entities.gates):
                    new_args["name"] = new_entities.gates[gate_idx]
                    extracted_params[f"gate_{gate_idx}"] = new_entities.gates[gate_idx]
                    gate_idx += 1
                else:
                    return False, None, "Candidate requires gate name, but new request specifies none", {}

            # Tools with source and sink (find_all_paths, get_max_depth, find_cut_points)
            if "source" in new_args and "sink" in new_args:
                orig_src = orig_call.args.get("source")
                orig_snk = orig_call.args.get("sink")
                if orig_src not in CLASS_KEYWORDS and orig_snk not in CLASS_KEYWORDS:
                    if len(new_entities.signals) >= 2:
                        m_from_to = re.search(r"from\s+([a-zA-Z0-9_\[\]]+)\s+to\s+([a-zA-Z0-9_\[\]]+)", new_request, re.IGNORECASE)
                        if m_from_to:
                            new_args["source"] = m_from_to.group(1)
                            new_args["sink"] = m_from_to.group(2)
                        else:
                            new_args["source"] = new_entities.signals[0]
                            new_args["sink"] = new_entities.signals[1]
                        extracted_params["source"] = new_args["source"]
                        extracted_params["sink"] = new_args["sink"]
                    else:
                        return False, None, "Candidate requires both source and sink signals, but request lacks them", {}
                elif orig_snk not in CLASS_KEYWORDS and orig_src in CLASS_KEYWORDS:
                    if new_entities.signals:
                        new_args["sink"] = new_entities.signals[0]
                        extracted_params["sink"] = new_entities.signals[0]

            # Tools with two signals a and b (check_signal_equivalence)
            if "a" in new_args and "b" in new_args:
                if len(new_entities.signals) >= 2:
                    new_args["a"] = new_entities.signals[0]
                    new_args["b"] = new_entities.signals[1]
                    extracted_params["a"] = new_args["a"]
                    extracted_params["b"] = new_args["b"]
                else:
                    return False, None, "Candidate requires two signals for equivalence check, but request lacks them", {}

            # Validate schema
            valid, schema_err = self.validate_schema(tool_name, new_args)
            if not valid:
                return False, None, schema_err, {}

            bound_calls.append(ToolCall(tool=tool_name, args=new_args))

        return True, bound_calls, None, extracted_params
