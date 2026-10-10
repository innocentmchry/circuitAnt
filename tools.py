"""The tools offered to the LLM: each public method of `Tools` is one function tool.

Its JSON schema is generated from the method (schema(): name, docstring, parameter
types), so adding a tool means adding one method here.  The methods forward to the
engine (analysis, transformation, verification) on the loaded design.
"""
import inspect
import typing
from pathlib import Path

from eda_engine import Design, analysis, read_design, transformation, verification, write_design


class Tools:
    """Engine operations exposed to the model.  State: the loaded design."""

    def __init__(self, design_config: dict | None = None, output_dir: str | Path = "."):
        self.design_config = design_config or {}  # format / lef / liberty / top, from the config
        self.out_dir = Path(output_dir)  # testcase logs, written designs and result lists go here
        self.design = None
        self.case = None
        self.log = None  # set by set_testcase_name

    def _d(self) -> Design:
        if self.design is None:
            raise ValueError("no design loaded; call read_design first")
        return self.design

    def set_testcase_name(self, name: str) -> str:
        """Set the current testcase name; responses are also logged to <name>.log."""
        self.case = name
        if self.log:
            self.log.close()
        self.log = open(self.out_dir / f"{name}.log", "w")
        return f"Testcase name set to {name!r}; log file: {str(self.out_dir / (name + '.log'))!r}"

    def read_design(self, path: str) -> str:
        """Load a gate-level Verilog netlist (replaces any loaded design)."""
        self.design = read_design(path, **self.design_config)
        return f"Design loaded from {str(Path(path).resolve())!r}"

    def write_design(self, path: str) -> str:
        """Write the current design to a Verilog file."""
        path = self.out_dir / path  # a relative path goes into the output directory
        write_design(self._d(), path)
        return f"Design written to {str(path)!r}"

    def count_gates(self) -> dict:
        """Count all gates, total and broken down by type (AND, OR, NOT, NAND, NOR, XOR, XNOR, BUF, DFF)."""
        return analysis.count_gates(self._d())

    def get_max_depth(self, source: str = "ANY_START", sink: str = "ANY_END") -> dict:
        """Maximum combinational logic depth (number of gates) from source to sink, with one deepest path.
        For the maximum logic depth of the design ('in the design', 'of the whole design', the critical path)
        omit both: the defaults ANY_START and ANY_END make paths from and to flip-flops count too.  Each of source/sink is a signal name (e.g. n12, n25[0], an internal net) or a class keyword:
        source ANY_PI (all primary inputs), ANY_DFF_Q (all flip-flop outputs), ANY_START (both, plus constants);
        sink ANY_PO (all primary outputs), ANY_DFF_D (all flip-flop D inputs), ANY_END (both).
        Use signal names whenever the request names signals.  Examples: register-to-register depth =
        (ANY_DFF_Q, ANY_DFF_D); depth of the fanin cone of signal X = (ANY_START, X).  Flip-flops cut paths."""
        return analysis.get_max_depth(self._d(), source, sink)

    def get_endpoint_stats(self, sink: str = "ANY_PO", metric: str = "depth",
                           operator: str | None = None, threshold: int | None = None) -> dict:
        """A value for every endpoint in `sink` (default ANY_PO: each primary-output bit; also ANY_DFF_D,
        ANY_END): metric "depth" = logic depth from any start point, metric "cone_size" = number of gates in
        its fanin cone.  Returns the maximum, which endpoints reach it, and -- with operator (>, >=, <, <=,
        ==, !=) and threshold -- how many endpoints satisfy value <operator> threshold.  Use for 'how many
        outputs have depth greater than 4', 'which output has the deepest fanin cone' (depth), 'which
        primary output has the largest fanin logic cone' (cone_size)."""
        return analysis.get_endpoint_stats(self._d(), sink, metric, operator, threshold)

    def get_cone(self, signal: str, direction: str = "fanin", shared_with: str | None = None) -> dict:
        """Gates in the transitive fanin cone (direction "fanin": the gates the signal depends on) or fanout
        cone (direction "fanout": the gates reachable from / driven by the signal) of a signal:
        the gate count, the count of each gate type, the first gates (sorted), and a file listing all of them
        if there are more.  Fanin cones stop at primary inputs, flip-flop outputs and constants; for a signal
        driven by a flip-flop the cone of that flop's D input is reported (through_dff).  Fanout cones
        include the flip-flops they reach.  shared_with: a second signal, to report only the gates common to
        both cones.  Use for every question about the gates in the cone of a signal: 'transitive fanin/fanout
        cone of X', 'number of each gate type in the cone of X', 'how many NAND (or any type) gates are in the
        cone of X' (read the type's count), 'how many gates are in the fanin cone of X', 'gates reachable from
        X' (fanout), 'gates shared between the
        fanin cones of A and B', and to compare the cones of two named outputs (call once per output)."""
        return analysis.get_cone(self._d(), signal, direction, shared_with, out_dir=self.out_dir)

    def get_fanout(self, signal: str) -> dict:
        """Fanout = number of input pins a signal drives (gate inputs and flip-flop D/CK/RN/SN pins; output
        ports are not loads).  signal: a signal name -> its fanout and the gates it drives directly (with the
        pins used); a gate name (e.g. g27319) -> the same for that gate's output signal; or a keyword ->
        the maximum fanout over a class and which signals have it: ANY_PI (primary inputs), ANY_DFF_Q
        (flip-flop outputs), ANY_GATE_OUTPUT (gate outputs), ANY_SIGNAL (all of these).  Use for 'maximum
        fanout of the design' (ANY_SIGNAL), 'which primary input has the highest fanout' (ANY_PI), 'fanout
        of X / list the gates X drives', 'gates connected to the output of gate g'."""
        return analysis.get_fanout(self._d(), signal, out_dir=self.out_dir)

    def find_gates(self, gate_type: str | None = None, connected_to: str | None = None,
                   pin: str | None = None, name: str | None = None) -> dict:
        """Gates matching every filter given, each with its type and all pin connections.  gate_type is either
        a logic type, matched by what the cell computes (any library, any number of inputs): AND, NAND, OR, NOR,
        XOR, XNOR, BUF, NOT, AOI21, AOI22, AOI211, AOI221, AOI222, AOI33, OAI21, OAI22, OAI211, OAI221, OAI222, OAI33, MUX2
        (2:1 multiplexer), FA (full adder), HA (half adder); or a cell name or pattern, matched by name (DFF,
        NAND2_X1, NAND2_*, DFF*).  connected_to = a signal on any of the gate's pins, or only on `pin` if given
        (e.g. pin CK for flip-flops on a clock) -- gates with a pin directly on that signal, not the signal's
        logic cone (for the gates in a cone use get_cone); name = one gate instance.  Returns the count, a count per cell
        type (by_cell), the first gates (sorted by name) and a file listing all of them if there are more.  Use
        for 'how many NAND gates / full adders / AOI21 cells are in the design', 'list all XNOR gates', 'list 5 NAND gates with
        their inputs and outputs', 'flip-flops driven by clock n0' (gate_type DFF, connected_to n0, pin CK),
        'gates connected to signal X', 'NAND gates with an input tied to constant 1' (gate_type NAND,
        connected_to 1'b1), 'what type of gate is g12 and its pin connections' (name g12)."""
        return analysis.find_gates(self._d(), gate_type, connected_to, pin, name, out_dir=self.out_dir)

    def get_ports(self, direction: str | None = None) -> dict:
        """Primary inputs and/or outputs (direction "input", "output", or omitted for both): the number of
        declared ports, the total number of bits, and each port with its width and range.  Use for 'how many
        primary inputs and outputs does this design have' and 'list all primary inputs with their bit widths'."""
        return analysis.get_ports(self._d(), direction)

    def find_cut_points(self, source: str = "ANY_PI", sink: str = "ANY_PO", signal: str | None = None) -> dict:
        """Points that every combinational path must pass through.  Without `signal` (source and sink are
        signal names): the gates lying on every source->sink path, i.e. articulation points -- removing any
        one disconnects source from sink.  With `signal`: whether some source and some sink are connected
        only through that signal (source/sink may also be class keywords as in get_max_depth; default
        ANY_PI to ANY_PO).  Use for 'articulation points between A and B' (source A, sink B) and 'is wire
        X a cut between any primary input and any primary output' (signal X)."""
        return analysis.find_cut_points(self._d(), source, sink, signal)

    def find_inverter_pairs(self) -> dict:
        """Back-to-back inverters: NOT gates fed directly by another NOT gate.  Returns the count, the first
        pairs (first gate, second gate, input, output) and a file listing all of them if there are more."""
        return analysis.find_inverter_pairs(self._d(), out_dir=self.out_dir)

    def find_duplicates(self) -> dict:
        """Structural duplicates: groups of gates of the same type on the same input signals.  Returns the
        number of groups, how many gates could be removed (all but one per group), and the first groups."""
        return analysis.find_duplicates(self._d(), out_dir=self.out_dir)

    def find_enable_hold(self) -> dict:
        """Flip-flops with enable or hold structures in their D input logic (D fed back from the flip-flop's
        own Q directly or through a gating/mux structure).  Returns the count, a breakdown by kind, and the
        first flip-flops.  Use for 'how many flip-flops have enable or hold structures in their D input'."""
        return analysis.find_enable_hold(self._d(), out_dir=self.out_dir)

    def derive_boolean_equation(self, signal: str) -> dict:
        """Boolean equation of a signal in terms of the inputs of its logic cone (* AND, + OR, ^ XOR, ~ NOT),
        plus each gate's own equation.  Says which primary inputs it uses and whether it also depends on
        flip-flop outputs (then no primary-input-only equation exists; report that first).  A signal driven
        directly by a flip-flop is stored state.  Large cones keep shared signals as named terms.  Use for
        'derive the Boolean equation for output X in terms of its primary inputs'."""
        return analysis.derive_boolean_equation(self._d(), signal, out_dir=self.out_dir)

    def check_equivalence(self) -> dict:
        """Formally check whether the current design is functionally equivalent to the netlist as it was
        loaded from disk (at flip-flop boundaries: same ports and flip-flops, flip-flop outputs treated as free
        inputs).  Returns equivalent true/false and status PASS / FAIL / TIMEOUT / ERROR.  Use for 'verify /
        check / prove / confirm that the current design is equivalent to the original netlist'."""
        return verification.check_equivalence(self._d())

    def check_signal_equivalence(self, a: str, b: str) -> dict:
        """Formally check whether two signals compute the same function for all input combinations
        (flip-flop outputs treated as free inputs).  Returns equivalent true/false."""
        return verification.check_signal_equivalence(self._d(), a, b)

    def check_symmetry(self, signal: str, input_a: str, input_b: str) -> dict:
        """Formally check whether the function at `signal` is symmetric in two of its inputs, i.e. unchanged
        when input_a and input_b are swapped.  The inputs must be primary inputs (or flip-flop outputs).
        Returns symmetric true/false."""
        return verification.check_symmetry(self._d(), signal, input_a, input_b)

    def find_all_paths(self, source: str, sink: str, avoid: str | None = None,
                       max_depth: int | None = None) -> dict:
        """All combinational paths from source to sink: the exact count, whether any exists, the first few
        paths, and a file listing all of them if there are more.  source/sink take a signal name or the same
        class keywords as get_max_depth (ANY_PI, ANY_DFF_Q, ANY_START / ANY_PO, ANY_DFF_D, ANY_END).
        avoid: a signal no path may pass through.  max_depth: only paths with at most this many gates.
        Examples: does a path exist from n4[3] to n5[0] = (n4[3], n5[0]), read `exists`; a path avoiding
        n5206 = avoid n5206; paths of length 0 from a PI straight to a PO = (ANY_PI, ANY_PO, max_depth 0);
        all register-to-register paths = (ANY_DFF_Q, ANY_DFF_D)."""
        return analysis.find_all_paths(self._d(), source, sink, avoid, max_depth, out_dir=self.out_dir)


    def rename_gate(self, old_name: str, new_name: str) -> dict:
        """Rename a gate instance (e.g. g462 -> renamed_gate).  Connections are unchanged."""
        return transformation.rename_gate(self._d(), old_name, new_name)

    def rename_wire(self, old_name: str, new_name: str) -> dict:
        """Rename a wire/signal; every reference follows.  Primary input/output ports cannot be renamed."""
        return transformation.rename_wire(self._d(), old_name, new_name)

    def remove_dangling_gates(self) -> dict:
        """Remove every gate that does not contribute to any primary output or flip-flop input (dangling /
        unused gates and the wires only they use).  Returns gates_removed and the gate count before/after.
        Use for 'remove dangling gates', 'prune unused gates', 'trim unused wires and gates'."""
        return transformation.remove_dangling_gates(self._d())

    def collapse_inverter_pairs(self) -> dict:
        """Collapse every back-to-back inverter pair (NOT feeding NOT) into a direct wire, repeatedly, so
        longer NOT chains collapse too.  Returns pairs_collapsed and gates_removed."""
        return transformation.collapse_inverter_pairs(self._d())

    def merge_duplicate_gates(self) -> dict:
        """Merge structural duplicates -- gates of the same type on the same inputs -- keeping one per group,
        repeatedly until none are left.  Also use it to share a value that is recomputed (e.g. a * b built
        twice): the copies are structurally identical, so merging computes it once.  Returns gates_merged."""
        return transformation.merge_duplicate_gates(self._d())

    def simplify_gates_with_constant_inputs(self, gate_type: str | None = None) -> dict:
        """Simplify 2-input gates with one input tied to a constant 1'b0 / 1'b1 into a single gate on the
        other input: NAND(x,1) -> NOT, NOR(x,0) -> NOT, AND(x,1) -> BUF, XOR(x,1) -> NOT, etc.  gate_type: only
        that type (e.g. NAND).  Gates that reduce to a constant are reported as skipped, not changed."""
        return transformation.simplify_gates_with_constant_inputs(self._d(), gate_type)

    def insert_buffers_for_fanout(self, max_fanout: int, net_name: str | None = None) -> dict:
        """Insert BUF trees so no signal drives more than max_fanout loads (gate/flip-flop input pins),
        with the fewest buffers.  net_name: only that signal; default every signal in the design.  Some
        loads may stay on the original signal.  Do not use when every load needs its own buffer (use
        insert_dedicated_buffers_for_loads).  Returns buffers_inserted."""
        return transformation.insert_buffers_for_fanout(self._d(), max_fanout, net_name)

    def insert_dedicated_buffers_for_loads(self, net_name: str) -> dict:
        """Give every load of a signal its own dedicated BUF: afterwards the signal drives only the new
        buffers, one per former load.  Use for 'each load of n2 is driven through a dedicated buffer'.
        Returns buffers_inserted."""
        return transformation.insert_dedicated_buffers_for_loads(self._d(), net_name)


    def reduce_depth(self) -> dict:
        """Restructure the combinational logic to reduce the maximum logic depth (critical path) of the whole
        design, with ABC; flip-flops are kept.  Returns depth and gate count before and after.  Use for
        'reduce the critical path depth', 'minimize maximum path depth', 'restructure to shrink the depth'."""
        return transformation.reduce_depth(self._d())

    def remap_gates(self, allowed: str, signal: str | None = None) -> dict:
        """Rebuild the combinational logic using only the allowed gate types, given comma-separated (e.g.
        "NAND,NOT", "AND,NOT", "NOR,NOT", or a single "NAND" / "NOR").  signal: only the fanin (logic) cone of
        this signal; default the entire design.  Use for 'remap the design to use only NAND and NOT gates',
        'reconstruct the netlist with only AND and NOT', 'convert the logic cone of n4042 to only NOR and NOT',
        'restructure the cone of output n64[0] to use only NAND and NOT'."""
        return transformation.remap_gates(self._d(), [a.strip() for a in allowed.split(",")], signal)

    def replace_gate_type(self, gate_type: str, allowed: str, signal: str | None = None) -> dict:
        """Replace every gate of one type (gate_type, e.g. XOR, OR) by equivalent logic using only the allowed
        types, comma-separated (e.g. "NAND", "NOR", "NAND,NOT", "AND,OR,NOT"); only NAND or only NOR uses the
        standard circuits (XOR -> 4 NAND, XNOR -> 4 NOR).  signal: only gates in the fanin cone of this signal;
        default the whole design.  Returns how many gates were replaced and the gates removed and added by type.
        Use for 'replace every XOR with an equivalent 4-NAND circuit' (XOR, NAND), 'replace all XNOR with
        NOR-only logic' (XNOR, NOR), 'replace OR gates in the cone of n45313 with NAND and NOT gates'
        (OR, "NAND,NOT", n45313), 'decompose XOR in the cone of n5[27] into AND, OR and NOT'."""
        return transformation.replace_gate_type(self._d(), gate_type, [a.strip() for a in allowed.split(",")],
                                                signal)


def schema(fn) -> dict:
    """OpenAI tool schema from a Tools method: name, docstring, parameter types.
    Parameters without a default are required; a hint of str, int or bool (optionally
    `| None`) gives the JSON type."""
    def json_type(annotation) -> str:
        types = {str: "string", int: "integer", bool: "boolean"}
        hint = [a for a in typing.get_args(annotation) if a is not type(None)] or [annotation]
        return types[hint[0]]

    params = inspect.signature(fn).parameters
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": inspect.getdoc(fn),
        "parameters": {
            "type": "object",
            "properties": {n: {"type": json_type(p.annotation)} for n, p in params.items()},
            "required": [n for n, p in params.items() if p.default is p.empty],
            "additionalProperties": False,  # else the model may invent arguments
        }}}
