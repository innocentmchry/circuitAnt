"""Knowledge Graph and domain context extraction for dynamic tool synthesis.

Retrieves category hierarchy, sibling tool implementations, and low-level eda_engine
primitives from the Tool Knowledge Graph (or codebase ontology) to ground code generation
according to the Problem -> Knowledge -> Tool paradigm (KTCE, AAAI-25).
"""
import inspect
from pathlib import Path
from typing import Optional

from agent.tools import Tools

# Reference engine API summaries provided to the synthesizer
ENGINE_APIS = {
    "analysis": '''
# Design object methods (d = self._d()):
#   d.net(net_name) -> odb.dbNet
#   d.inst(inst_name) -> odb.dbInst
#   d.block -> OpenDB dbBlock
# eda_engine.utils available helpers:
#   drivers, loads = utils.netlist_view(d)
#   # drivers: {net_name: (inst_name, kind, [input_net_names], {pin_name: net_name})}
#   # loads: {net_name: [(inst_name, kind, pin_name, iterm)]}
#   # d.cells[kind].kind gives cell class, e.g. "inverter", "and", "or", "xor", "sequential", "buffer"
#   # utils.fanin_signals(d, net_name) -> list of input signal names driving the combinational gate
#   # utils.write_list(out_dir, base_name, items) -> file_path string if list is large
# eda_engine.analysis available helpers:
#   analysis.derive_boolean_equation(d, signal, out_dir=self.out_dir) -> dict
#   analysis.find_inverter_pairs(d, inline_limit=10, out_dir=self.out_dir) -> dict: {"count", "matches": [{"first", "second", "input", "output"}]}
#   analysis.count_gates(d) -> dict: {"total_gates", "by_type"}
#   analysis.find_gates(d, gate_type, ...) -> dict
#   analysis.get_fanout(d, signal) -> dict
''',
    "transformation": '''
# Design object methods (d = self._d()):
#   d.net(net_name) -> odb.dbNet
#   d.inst(inst_name) -> odb.dbInst
#   d.block -> OpenDB dbBlock
# eda_engine.transformation available implementations:
#   transformation.collapse_inverter_pairs(d) -> dict: {"pairs_collapsed", "gates_removed", "gates_before", "gates_after"}
#   transformation.insert_dedicated_buffers_for_loads(d, net_name) -> dict: {"net_name", "buffers_inserted"}
#   transformation.insert_buffers_for_fanout(d, threshold=10) -> dict
#   transformation.merge_duplicate_gates(d) -> dict
#   transformation.remove_dangling_gates(d) -> dict
#   transformation.reduce_depth(d) -> dict
#
# ==============================================================================
# THE "BIG 4" OPENDB MICRO-IDIOMS (STRICT RULES):
# 1. FIND LOADS OF A SIGNAL (NEVER net.getLoads()):
#    loads = [it for it in net.getITerms() if it.isInputSignal()]
# 2. FIND DRIVER OF A SIGNAL (NEVER net.getDriver()):
#    drivers = [it for it in net.getITerms() if it.isOutputSignal()]
#    driver_inst = drivers[0].getInst() if drivers else None
# 3. CREATE GATE / CELL INSTANCE:
#    # buf, a, y = d.buffer_cell()  # gets default ("BUF", "A", "Y")
#    inst = edit.add_gate(d, master_cell_name, {pin_name: dbNet_obj}, base_name)
# 4. REWIRE AN ITERM (NEVER iterm.setNet()):
#    iterm.disconnect()
#    iterm.connect(new_dbNet_obj)
# ==============================================================================
#
# eda_engine.utils available helpers:
#   drivers, flops = utils.netlist_view(d)
# eda_engine.edit available primitives:
#   net = edit.new_net(d, base_name)  # Returns odb.dbNet object directly (do NOT call d.net on it!)
#   inst = edit.add_gate(d, master_name, {pin_name: dbNet_obj}, base_name)  # master_name e.g. "BUF", "NOT"
#   edit.remove_gate(d, inst)
#   edit.merge_nets(d, keep_net_obj, drop_net_obj)  # Both must be odb.dbNet objects
#
# VERIFICATION (CEC):
#   # For transformation tools, verification can be checked via:
#   # verif = verification.check_equivalence(self._d())  # returns {"equivalent": bool, "status": "PASS"|"FAIL"}
''',
    "optimization": '''
# Design object methods (d = self._d()):
#   net = d.net(net_name) -> odb.dbNet for the signal
#   buf, a, y = d.buffer_cell() -> (buffer_master_cell_name, input_pin_name, output_pin_name)
# eda_engine.transformation available implementations:
#   transformation.insert_dedicated_buffers_for_loads(d, net_name) -> dict: {"net_name", "buffers_inserted"}
#   transformation.reduce_depth(d) -> dict: {"depth_before", "depth_after"}
#
# ==============================================================================
# THE "BIG 4" OPENDB MICRO-IDIOMS:
# 1. loads = [it for it in net.getITerms() if it.isInputSignal()]
# 2. drivers = [it for it in net.getITerms() if it.isOutputSignal()]
# 3. inst = edit.add_gate(d, buf, {a: net, y: out}, f"{net_name}_buf")
# 4. it.disconnect(); it.connect(new_net)
# ==============================================================================
'''
}


class KGContextExtractor:
    """Extracts ontology, sibling tool templates, and API primitives for tool synthesis."""

    def __init__(self, tools_instance: Optional[Tools] = None):
        self.tools = tools_instance or Tools()

    def get_sibling_examples(self, category: str) -> list[str]:
        """Returns source codes of 1-2 representative sibling methods in the same category."""
        category_lower = category.lower()
        candidates = []
        if category_lower in ("transformation", "cleanup"):
            candidates = ["remove_dangling_gates", "simplify_gates_with_constant_inputs"]
        elif category_lower in ("optimization", "buffering", "resynthesis"):
            candidates = ["insert_buffers_for_fanout", "reduce_depth"]
        elif category_lower in ("analysis", "patterns", "cones"):
            candidates = ["find_duplicates", "find_gates"]
        else:
            candidates = ["count_gates", "get_fanout"]

        examples = []
        for name in candidates:
            fn = getattr(Tools, name, None)
            if fn:
                try:
                    src = inspect.getsource(fn)
                    examples.append(src.strip())
                except Exception:
                    pass
        return examples

    def build_context_bundle(self, category: str, intent: str, suggested_name: str) -> str:
        """Constructs the full context bundle for the synthesizer LLM."""
        cat_key = category.lower()
        if "opt" in cat_key or "buf" in cat_key:
            cat_norm = "optimization"
        elif "trans" in cat_key or "clean" in cat_key:
            cat_norm = "transformation"
        else:
            cat_norm = "analysis"

        siblings = self.get_sibling_examples(cat_norm)
        siblings_text = "\n\n# --- Sibling Tool Example ---\n".join(siblings)
        engine_api = ENGINE_APIS.get(cat_norm, ENGINE_APIS["transformation"])

        bundle = f"""### Domain Category: {cat_norm.upper()}
Target Tool: `{suggested_name}`
Target Intent: {intent}

### Architecture & Typing Guidelines:
1. Method must be a method of `Tools` taking `self` and appropriate typed parameters.
2. State is accessed via `self._d()` which returns the loaded `Design` object (`d`).
3. Return value MUST be a JSON-serializable Python dictionary (e.g. counts, status, lists).
4. Annotate parameters with `Annotated[str, Param(...)]` or use type aliases:
   - `Signal = Annotated[str, Param("a signal name", kind="Signal")]`
   - `Gate = Annotated[str, Param("a gate instance name", kind="Gate")]`
5. Decorate with `@tool(examples=(...))` providing 1-2 realistic natural-language phrases.
6. Write a descriptive docstring whose first sentence summarizes what the tool does.
7. CRITICAL (KTCE Reusability Contract): The function must be fully general. NEVER hardcode
   specific netlist names (e.g. 'n12', 'g100').

### Available Engine APIs:
{engine_api}

### Sibling Reference Implementations:
{siblings_text}
"""
        return bundle
