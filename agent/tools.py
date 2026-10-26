"""The tools offered to the LLM: each public method of `Tools` is one function tool.

Its JSON schema is generated from the method (schema()), so adding a tool means adding
one method here.  What the model is told about a tool is kept structured, so it can be
given as a tool schema (tool access "all") or as a knowledge graph:
  * the docstring: what the tool does and returns; its first line is a one-line summary;
  * each parameter's type hint, annotated with a Param: what the argument is, the kind of
    value (Signal, StartPoint, ... -- one shared definition, used by several tools) and the
    allowed values when there is a fixed set;
  * @tool: example requests the tool answers (with their arguments) and the neighbouring
    tools to use instead in given cases.
The methods forward to the engine (analysis, transformation, verification) on the loaded design.
"""
import inspect
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from eda_engine import Design, analysis, read_design, transformation, verification, write_design


@dataclass(frozen=True)
class Param:
    """What an argument is: a description, the allowed values when there is a fixed set, the kind
    of value (a name shared by every argument of that kind, e.g. Signal), and the keywords it
    accepts besides names ((keyword, meaning) pairs, e.g. ("ANY_PI", "all primary inputs"))."""
    description: str = ""
    values: tuple = ()
    kind: str = ""
    keywords: tuple = ()

    def text(self) -> str:
        """The description, followed by the keywords and their meanings."""
        if not self.keywords:
            return self.description
        return f"{self.description}: " + ", ".join(f"{k} ({meaning})" for k, meaning in self.keywords)


# Kinds of argument values shared by several tools.
Signal = Annotated[str, Param("a signal (net) name as in the netlist, e.g. n12 or n25[0]", kind="Signal")]
Gate = Annotated[str, Param("a gate instance name, e.g. g27319", kind="Gate")]
StartPoint = Annotated[str, Param(
    "a signal name, or a class of start points", kind="StartPoint",
    keywords=(("ANY_PI", "all primary inputs"), ("ANY_DFF_Q", "all flip-flop outputs"),
              ("ANY_START", "ANY_PI and ANY_DFF_Q, plus constants")))]
EndPoint = Annotated[str, Param(
    "a signal name, or a class of endpoints", kind="EndPoint",
    keywords=(("ANY_PO", "all primary outputs"), ("ANY_DFF_D", "all flip-flop D inputs"),
              ("ANY_END", "ANY_PO and ANY_DFF_D")))]
GateType = Annotated[str, Param(
    "a logic type, matched by what the cell computes (any library, any number of inputs): AND, NAND, OR, "
    "NOR, XOR, XNOR, BUF, NOT, AOI21, AOI22, AOI211, AOI221, AOI222, AOI33, OAI21, OAI22, OAI211, OAI221, "
    "OAI222, OAI33, MUX2 (2:1 multiplexer), FA (full adder), HA (half adder); or a cell name or pattern, "
    "matched by name (DFF, NAND2_X1, NAND2_*, DFF*)", kind="GateType")]
GateTypes = Annotated[str, Param("logic types, comma-separated (e.g. NAND,NOT)", kind="GateTypeList")]
FilePath = Annotated[str, Param("a file path, exactly as the request gives it", kind="FilePath")]


def tool(examples: tuple = (), instead: dict | None = None, grounded: tuple = ()):
    """Knowledge about a tool beyond its docstring.  examples: requests it answers, each a phrase or
    (phrase, {argument: value}); instead: {other tool: when to use that one instead}; grounded:
    arguments whose value (a file: its file name) the request itself must name -- otherwise the call is
    refused (for tools that reset the session, which models otherwise call unasked)."""
    def mark(fn):
        fn.examples = [(e, {}) if isinstance(e, str) else e for e in examples]
        fn.instead = dict(instead or {})
        fn.grounded = tuple(grounded)
        return fn
    return mark


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

    @tool(grounded=("name",))
    def set_testcase_name(self, name: Annotated[str, Param("the testcase name, e.g. test001")]) -> str:
        """Start a testcase: set its name; responses are also logged to <name>.log.  Only for a request
        that starts a new testcase (it names the case); the name of the current one is in the context."""
        if name == self.case and self.log:  # already this testcase: keep its log (reopening empties it)
            return f"Testcase name is already {name!r}; log file: {str(self.out_dir / (name + '.log'))!r}"
        self.case = name
        if self.log:
            self.log.close()
        self.log = open(self.out_dir / f"{name}.log", "w")
        return f"Testcase name set to {name!r}; log file: {str(self.out_dir / (name + '.log'))!r}"

    @tool(grounded=("path",))
    def read_design(self, path: FilePath) -> str:
        """Load a gate-level Verilog netlist, replacing the loaded design and undoing every change made to
        it.  Only for a request that asks to load a design; the loaded netlist is in the context."""
        self.design = read_design(path, **self.design_config)
        return f"Design loaded from {str(Path(path).resolve())!r}"

    @tool(grounded=("path",))
    def write_design(self, path: FilePath) -> str:
        """Write the current design to a Verilog file.  Only for a request that asks to write the design."""
        path = self.out_dir / path  # a relative path goes into the output directory
        write_design(self._d(), path)
        return f"Design written to {str(path)!r}"

    def count_gates(self) -> dict:
        """Count all gates, total and broken down by type (AND, OR, NOT, NAND, NOR, XOR, XNOR, BUF, DFF)."""
        return analysis.count_gates(self._d())

    @tool(examples=(("maximum logic depth of the design / the critical path", {}),
                    ("register-to-register depth", {"source": "ANY_DFF_Q", "sink": "ANY_DFF_D"}),
                    ("depth of the fanin cone of signal X", {"source": "ANY_START", "sink": "X"})),
          instead={"find_all_paths": "to count or list paths rather than measure the deepest one"})
    def get_max_depth(self, source: StartPoint = "ANY_START", sink: EndPoint = "ANY_END") -> dict:
        """Maximum combinational logic depth (number of gates) from source to sink, with one deepest path.
        For the depth of the whole design omit both: the defaults ANY_START and ANY_END make paths from and
        to flip-flops count too.  Use signal names whenever the request names signals.  Flip-flops cut
        paths."""
        return analysis.get_max_depth(self._d(), source, sink)

    @tool(examples=(("how many outputs have depth greater than 4", {"operator": ">", "threshold": 4}),
                    ("which output has the deepest fanin cone", {"metric": "depth"}),
                    ("which primary output has the largest fanin logic cone", {"metric": "cone_size"})))
    def get_endpoint_stats(
            self,
            sink: Annotated[EndPoint, Param("the endpoints, each scored separately; ANY_PO = each primary-"
                                            "output bit")] = "ANY_PO",
            metric: Annotated[str, Param("depth: logic depth from any start point; cone_size: number of gates "
                                         "in its fanin cone", values=("depth", "cone_size"))] = "depth",
            operator: Annotated[str | None, Param("with threshold: count the endpoints whose value <operator> "
                                                  "threshold", values=(">", ">=", "<", "<=", "==", "!="))] = None,
            threshold: Annotated[int | None, Param("the value compared against")] = None) -> dict:
        """A value (logic depth or fanin-cone size) for every endpoint: the maximum, which endpoints reach
        it, and -- with operator and threshold -- how many endpoints satisfy value <operator> threshold."""
        return analysis.get_endpoint_stats(self._d(), sink, metric, operator, threshold)

    @tool(examples=("transitive fanin/fanout cone of X",
                    "number of each gate type in the cone of X",
                    "how many NAND (or any type) gates are in the cone of X (read the type's count)",
                    "how many gates are in the fanin cone of X",
                    ("gates reachable from X", {"direction": "fanout"}),
                    ("gates shared between the fanin cones of A and B", {"signal": "A", "shared_with": "B"})))
    def get_cone(self, signal: Signal,
                 direction: Annotated[str, Param("fanin: the gates the signal depends on; fanout: the gates "
                                                 "reachable from / driven by the signal",
                                                 values=("fanin", "fanout"))] = "fanin",
                 shared_with: Annotated[Signal | None, Param("a second signal: report only the gates common "
                                                             "to both cones")] = None) -> dict:
        """Gates in the transitive fanin or fanout cone of a signal: the gate count, the count of each gate
        type, the first gates (sorted), and a file listing all of them if there are more.  Fanin cones stop
        at primary inputs, flip-flop outputs and constants; for a signal driven by a flip-flop the cone of
        that flop's D input is reported (through_dff).  Fanout cones include the flip-flops they reach.
        Use for every question about the gates in the cone of a signal; to compare the cones of two named
        outputs, call once per output."""
        return analysis.get_cone(self._d(), signal, direction, shared_with, out_dir=self.out_dir)

    @tool(examples=(("maximum fanout of the design", {"signal": "ANY_SIGNAL"}),
                    ("which primary input has the highest fanout", {"signal": "ANY_PI"}),
                    ("fanout of X / list the gates X drives", {"signal": "X"}),
                    ("gates connected to the output of gate g", {"signal": "g"})))
    def get_fanout(self, signal: Annotated[str, Param(
            "a signal name -> its fanout and the gates it drives directly (with the pins used); a gate name "
            "(e.g. g27319) -> the same for that gate's output signal; or a class keyword -> the maximum fanout "
            "over the class and which signals have it", kind="FanoutTarget",
            keywords=(("ANY_PI", "all primary inputs"), ("ANY_DFF_Q", "all flip-flop outputs"),
                      ("ANY_GATE_OUTPUT", "all gate outputs"),
                      ("ANY_SIGNAL", "ANY_PI, ANY_DFF_Q and ANY_GATE_OUTPUT")))]) -> dict:
        """Fanout (number of input pins driven) of a signal or a gate's output, or the maximum fanout over a
        class of signals.  Loads are gate inputs and flip-flop D/CK/RN/SN pins; output ports are not loads."""
        return analysis.get_fanout(self._d(), signal, out_dir=self.out_dir)

    @tool(examples=(("how many NAND gates / full adders / AOI21 cells are in the design", {"gate_type": "NAND"}),
                    ("list all XNOR gates", {"gate_type": "XNOR"}),
                    ("list 5 NAND gates with their inputs and outputs", {"gate_type": "NAND"}),
                    ("flip-flops driven by clock X", {"gate_type": "DFF", "connected_to": "X", "pin": "CK"}),
                    ("gates connected to signal X", {"connected_to": "X"}),
                    ("NAND gates with an input tied to constant 1", {"gate_type": "NAND", "connected_to": "1'b1"}),
                    ("what type of gate is g and its pin connections", {"name": "g"})),
          instead={"get_cone": "for the gates in the logic cone of a signal"})
    def find_gates(self, gate_type: GateType | None = None,
                   connected_to: Annotated[Signal | None, Param(
                       "a signal on any of the gate's pins, or only on `pin` if given: gates with a pin directly "
                       "on that signal, not the signal's logic cone")] = None,
                   pin: Annotated[str | None, Param("the pin connected_to must be on, e.g. CK for the clock "
                                                    "pin of flip-flops")] = None,
                   name: Annotated[Gate | None, Param("one gate instance")] = None) -> dict:
        """Gates matching every filter given, each with its type and all pin connections.  Returns the count,
        a count per cell type (by_cell), the first gates (sorted by name) and a file listing all of them if
        there are more."""
        return analysis.find_gates(self._d(), gate_type, connected_to, pin, name, out_dir=self.out_dir)

    @tool(examples=("how many primary inputs and outputs does this design have",
                    ("list all primary inputs with their bit widths", {"direction": "input"})))
    def get_ports(self, direction: Annotated[str | None, Param("omit for both",
                                                               values=("input", "output"))] = None) -> dict:
        """Primary inputs and/or outputs: the number of declared ports, the total number of bits, and each
        port with its width and range."""
        return analysis.get_ports(self._d(), direction)

    @tool(examples=(("articulation points between A and B", {"source": "A", "sink": "B"}),
                    ("is wire X a cut between any primary input and any primary output", {"signal": "X"})))
    def find_cut_points(self, source: StartPoint = "ANY_PI", sink: EndPoint = "ANY_PO",
                        signal: Annotated[Signal | None, Param("check only whether some source and some sink "
                                                               "are connected only through this signal")] = None
                        ) -> dict:
        """Points that every combinational path must pass through.  Without signal (source and sink are
        signal names): the gates lying on every source->sink path, i.e. articulation points -- removing any
        one disconnects source from sink.  With signal: whether some source and some sink are connected only
        through that signal."""
        return analysis.find_cut_points(self._d(), source, sink, signal)

    @tool(examples=("how many back-to-back inverter pairs are in the design",
                    "list the double inversions (NOT gates feeding NOT gates)"),
          instead={"collapse_inverter_pairs": "to remove the pairs (each becomes a wire), not just report them"})
    def find_inverter_pairs(self) -> dict:
        """Back-to-back inverters: NOT gates fed directly by another NOT gate.  Returns the count, the first
        pairs (first gate, second gate, input, output) and a file listing all of them if there are more."""
        return analysis.find_inverter_pairs(self._d(), out_dir=self.out_dir)

    @tool(examples=("how many gates are structural duplicates",
                    "which gates compute the same function on the same inputs"),
          instead={"merge_duplicate_gates": "to remove the duplicates (keep one per group), not just report them"})
    def find_duplicates(self) -> dict:
        """Structural duplicates: groups of gates of the same type on the same input signals.  Returns the
        number of groups, how many gates could be removed (all but one per group), and the first groups."""
        return analysis.find_duplicates(self._d(), out_dir=self.out_dir)

    @tool(examples=("how many flip-flops have enable or hold structures in their D input",))
    def find_enable_hold(self) -> dict:
        """Flip-flops with enable or hold structures in their D input logic (D fed back from the flip-flop's
        own Q directly or through a gating/mux structure).  Returns the count, a breakdown by kind, and the
        first flip-flops."""
        return analysis.find_enable_hold(self._d(), out_dir=self.out_dir)

    @tool(examples=("derive the Boolean equation for output X in terms of its primary inputs",))
    def derive_boolean_equation(self, signal: Signal) -> dict:
        """Boolean equation of a signal in terms of the inputs of its logic cone (* AND, + OR, ^ XOR, ~ NOT),
        plus each gate's own equation.  Says which primary inputs it uses and whether it also depends on
        flip-flop outputs (then no primary-input-only equation exists; report that first).  A signal driven
        directly by a flip-flop is stored state.  Large cones keep shared signals as named terms."""
        return analysis.derive_boolean_equation(self._d(), signal, out_dir=self.out_dir)

    @tool(examples=("verify / check / prove / confirm that the current design is equivalent to the original "
                    "netlist",),
          instead={"check_signal_equivalence": "to compare two signals within the current design"})
    def check_equivalence(self) -> dict:
        """Formally check whether the current design is functionally equivalent to the netlist as it was
        loaded from disk (at flip-flop boundaries: same ports and flip-flops, flip-flop outputs treated as free
        inputs).  Returns equivalent true/false and status PASS / FAIL / TIMEOUT / ERROR."""
        return verification.check_equivalence(self._d())

    @tool(examples=(("are signals A and B functionally equivalent", {"a": "A", "b": "B"}),
                    "do two internal signals compute the same function for every input combination (yes / no)"),
          instead={"check_equivalence": "to compare the whole current design with the netlist as loaded"})
    def check_signal_equivalence(self, a: Signal, b: Signal) -> dict:
        """Formally check whether two signals compute the same function for all input combinations
        (flip-flop outputs treated as free inputs).  Returns equivalent true/false."""
        return verification.check_signal_equivalence(self._d(), a, b)

    @tool(examples=(("is the function at X symmetric with respect to inputs A and B",
                     {"signal": "X", "input_a": "A", "input_b": "B"}),
                    "can two inputs of a signal's function be swapped without changing it"))
    def check_symmetry(self, signal: Signal,
                       input_a: Annotated[Signal, Param("an input of the function: a primary input or "
                                                        "flip-flop output")],
                       input_b: Annotated[Signal, Param("another input of the function")]) -> dict:
        """Formally check whether the function at `signal` is symmetric in two of its inputs, i.e. unchanged
        when input_a and input_b are swapped.  Returns symmetric true/false."""
        return verification.check_symmetry(self._d(), signal, input_a, input_b)

    @tool(examples=(("does a path exist from A to B (read exists)", {"source": "A", "sink": "B"}),
                    ("does output X depend on input Y (a path from Y to X)", {"source": "Y", "sink": "X"}),
                    ("paths from A to B avoiding X", {"source": "A", "sink": "B", "avoid": "X"}),
                    ("paths of length 0 from a primary input straight to a primary output",
                     {"source": "ANY_PI", "sink": "ANY_PO", "max_depth": 0}),
                    ("all register-to-register paths", {"source": "ANY_DFF_Q", "sink": "ANY_DFF_D"})),
          instead={"get_max_depth": "for the length of the deepest path"})
    def find_all_paths(self, source: StartPoint, sink: EndPoint,
                       avoid: Annotated[Signal | None, Param("a signal no path may pass through")] = None,
                       max_depth: Annotated[int | None, Param("only paths with at most this many gates")] = None
                       ) -> dict:
        """All combinational paths from source to sink: the exact count, whether any exists, the first few
        paths, and a file listing all of them if there are more."""
        return analysis.find_all_paths(self._d(), source, sink, avoid, max_depth, out_dir=self.out_dir)

    @tool(examples=(("rename gate g to NEW", {"old_name": "g", "new_name": "NEW"}),
                    "change the identifier of a gate instance and update all references"),
          instead={"rename_wire": "to rename a wire / signal"})
    def rename_gate(self, old_name: Gate, new_name: Annotated[str, Param("the new instance name")]) -> dict:
        """Rename a gate instance (e.g. g462 -> renamed_gate).  Connections are unchanged."""
        return transformation.rename_gate(self._d(), old_name, new_name)

    @tool(examples=(("rename wire X to NEW", {"old_name": "X", "new_name": "NEW"}),
                    "change the identifier of a signal and update all references"),
          instead={"rename_gate": "to rename a gate instance"})
    def rename_wire(self, old_name: Signal, new_name: Annotated[str, Param("the new signal name")]) -> dict:
        """Rename a wire/signal; every reference follows.  Primary input/output ports cannot be renamed."""
        return transformation.rename_wire(self._d(), old_name, new_name)

    @tool(examples=("remove dangling gates", "prune unused gates", "trim unused wires and gates"),
          instead={"merge_duplicate_gates": "for redundant gates that repeat another gate on the same inputs "
                                            "(a request for all redundant gates may need both)"})
    def remove_dangling_gates(self) -> dict:
        """Remove every gate that does not contribute to any primary output or flip-flop input (dangling /
        unused gates and the wires only they use).  Returns gates_removed and the gate count before/after."""
        return transformation.remove_dangling_gates(self._d())

    @tool(examples=("find / locate the back-to-back inverter pairs and collapse them into wires",
                    "remove double inversions"),
          instead={"find_inverter_pairs": "to only count or list the pairs without changing the design"})
    def collapse_inverter_pairs(self) -> dict:
        """Collapse every back-to-back inverter pair (NOT feeding NOT) into a direct wire, repeatedly, so
        longer NOT chains collapse too.  Returns pairs_collapsed and gates_removed."""
        return transformation.collapse_inverter_pairs(self._d())

    @tool(examples=("merge the gates that are structural duplicates",
                    "remove redundant copies of a gate (same type on the same inputs)"),
          instead={"find_duplicates": "to only count or list the duplicates without changing the design",
                   "remove_dangling_gates": "for redundant gates that drive nothing used (a request for all "
                                            "redundant gates may need both)"})
    def merge_duplicate_gates(self) -> dict:
        """Merge structural duplicates -- gates of the same type on the same inputs -- keeping one per group,
        repeatedly until none are left.  Also use it to share a value that is recomputed (e.g. a * b built
        twice): the copies are structurally identical, so merging computes it once.  Returns gates_merged."""
        return transformation.merge_duplicate_gates(self._d())

    @tool(examples=(("replace every T gate that has an input tied to a constant by the simpler gate it reduces to",
                     {"gate_type": "T"}),
                    "simplify gates with constant inputs"),
          instead={"find_gates": "to only list gates with an input tied to a constant (connected_to 1'b0 or 1'b1)"})
    def simplify_gates_with_constant_inputs(
            self, gate_type: Annotated[GateType | None, Param("only gates of this type (e.g. NAND); default "
                                                              "every type")] = None) -> dict:
        """Simplify 2-input gates with one input tied to a constant 1'b0 / 1'b1 into a single gate on the
        other input: NAND(x,1) -> NOT, NOR(x,0) -> NOT, AND(x,1) -> BUF, XOR(x,1) -> NOT, etc.  Gates that
        reduce to a constant are reported as skipped, not changed."""
        return transformation.simplify_gates_with_constant_inputs(self._d(), gate_type)

    @tool(instead={"insert_dedicated_buffers_for_loads": "when every load needs its own buffer"})
    def insert_buffers_for_fanout(
            self, max_fanout: Annotated[int, Param("the most loads (gate/flip-flop input pins) a signal may "
                                                   "drive")],
            net_name: Annotated[Signal | None, Param("only this signal; default every signal in the design")]
            = None) -> dict:
        """Insert BUF trees so no signal drives more than max_fanout loads, with the fewest buffers.  Some
        loads may stay on the original signal.  Returns buffers_inserted."""
        return transformation.insert_buffers_for_fanout(self._d(), max_fanout, net_name)

    @tool(examples=("each load of X is driven through a dedicated buffer",),
          instead={"insert_buffers_for_fanout": "to limit the fanout of signals with as few buffers as possible"})
    def insert_dedicated_buffers_for_loads(self, net_name: Signal) -> dict:
        """Give every load of a signal its own dedicated BUF: afterwards the signal drives only the new
        buffers, one per former load.  Returns buffers_inserted."""
        return transformation.insert_dedicated_buffers_for_loads(self._d(), net_name)

    @tool(examples=("reduce the critical path depth", "minimize maximum path depth",
                    "restructure to shrink the depth"))
    def reduce_depth(self) -> dict:
        """Restructure the combinational logic to reduce the maximum logic depth (critical path) of the whole
        design, with ABC; flip-flops are kept.  Returns depth and gate count before and after."""
        return transformation.reduce_depth(self._d())

    @tool(examples=(("remap the design to use only NAND and NOT gates", {"allowed": "NAND,NOT"}),
                    ("reconstruct the netlist with only AND and NOT", {"allowed": "AND,NOT"}),
                    ("convert the logic cone of X to only NOR and NOT", {"allowed": "NOR,NOT", "signal": "X"}),
                    ("restructure the cone of output X to use only NAND and NOT",
                     {"allowed": "NAND,NOT", "signal": "X"})),
          instead={"replace_gate_type": "to replace only the gates of one type"})
    def remap_gates(self, allowed: Annotated[GateTypes, Param(
                        "the only gate types the result may use, e.g. \"NAND,NOT\", \"AND,NOT\", \"NOR,NOT\", or "
                        "a single \"NAND\" / \"NOR\"")],
                    signal: Annotated[Signal | None, Param("only the fanin (logic) cone of this signal; default "
                                                           "the entire design")] = None) -> dict:
        """Rebuild the combinational logic using only the allowed gate types."""
        return transformation.remap_gates(self._d(), [a.strip() for a in allowed.split(",")], signal)

    @tool(examples=(("replace every XOR with an equivalent 4-NAND circuit", {"gate_type": "XOR", "allowed": "NAND"}),
                    ("replace all XNOR with NOR-only logic", {"gate_type": "XNOR", "allowed": "NOR"}),
                    ("replace OR gates in the cone of X with NAND and NOT gates",
                     {"gate_type": "OR", "allowed": "NAND,NOT", "signal": "X"}),
                    ("decompose XOR in the cone of X into AND, OR and NOT",
                     {"gate_type": "XOR", "allowed": "AND,OR,NOT", "signal": "X"})),
          instead={"remap_gates": "to rebuild all the logic (of the design or a cone) with a gate set"})
    def replace_gate_type(self, gate_type: Annotated[GateType, Param("the type to replace, e.g. XOR, OR")],
                          allowed: Annotated[GateTypes, Param(
                              "the types the replacement may use, e.g. \"NAND\", \"NOR\", \"NAND,NOT\", "
                              "\"AND,OR,NOT\"; only NAND or only NOR uses the standard circuits (XOR -> 4 NAND, "
                              "XNOR -> 4 NOR)")],
                          signal: Annotated[Signal | None, Param("only gates in the fanin cone of this signal; "
                                                                 "default the whole design")] = None) -> dict:
        """Replace every gate of one type by equivalent logic using only the allowed types.  Returns how many
        gates were replaced and the gates removed and added by type."""
        return transformation.replace_gate_type(self._d(), gate_type, [a.strip() for a in allowed.split(",")],
                                                signal)


def changes_design(fn) -> bool:
    """Whether a Tools method edits the loaded design: a transformation, or read_design (replaces it)."""
    return "transformation" in fn.__code__.co_names or fn.__name__ == "read_design"


def parameter(hint) -> tuple[type, list[Param]]:
    """(base type, Params) of a parameter's type hint: str, int or bool, optionally `| None`, possibly
    Annotated (at any level) with Params.  The parameter's own Params come before its kind's."""
    notes = []
    while True:
        if typing.get_origin(hint) is Annotated:
            notes += [m for m in hint.__metadata__ if isinstance(m, Param)]
            hint = typing.get_args(hint)[0]
        elif type(None) in typing.get_args(hint):
            hint = next(a for a in typing.get_args(hint) if a is not type(None))
        else:
            return hint, [n for n in notes if not n.kind] + [n for n in notes if n.kind]


def schema(fn) -> dict:
    """OpenAI tool schema from a Tools method: its name; the docstring plus the @tool examples and
    alternatives as the description; per parameter the JSON type, its description (its own, or else its
    kind's) and the allowed values (enum).  Parameters without a default are required."""
    types = {str: "string", int: "integer", bool: "boolean"}
    hints = typing.get_type_hints(fn, include_extras=True)
    params = inspect.signature(fn).parameters
    properties = {}
    for name in params:
        base, notes = parameter(hints[name])
        prop = {"type": types[base]}
        own = [n.text() for n in notes if n.description and not n.kind]
        kind = [n.text() for n in notes if n.description and n.kind]
        if own or kind:
            prop["description"] = "; ".join(own or kind)
        values = next((n.values for n in notes if n.values), ())
        if values:
            prop["enum"] = list(values)
        properties[name] = prop

    description = inspect.getdoc(fn)
    examples = getattr(fn, "examples", [])
    if examples:
        args = lambda a: f" ({', '.join(f'{k}={v}' for k, v in a.items())})" if a else ""
        description += "\nUse for: " + "; ".join(f"'{phrase}'{args(a)}" for phrase, a in examples) + "."
    for other, when in getattr(fn, "instead", {}).items():
        description += f"\nUse {other} instead {when}."
    if getattr(fn, "grounded", ()):
        description += f"\nRefused unless the request itself names the {' and '.join(fn.grounded)}."
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": [n for n, p in params.items() if p.default is p.empty],
            "additionalProperties": False,  # else the model may invent arguments
        }}}
