"""The loaded netlist: an OpenDB block plus what OpenDB does not keep.

Created by the readers (design_io.read_design); the analysis, transformation and
verification packages work on `Design.block`.  Cells are classified from their
libraries (Design.cells: sequential or not, Liberty functions), so any
standard-cell library works.  OpenROAD's Tcl interface is reached through
Design.tcl only.
"""
import re
from dataclasses import dataclass, field
from itertools import permutations

CONSTANT = {"POWER": "1'b1", "GROUND": "1'b0"}  # how OpenDB types literal 1'b1 / 1'b0 nets
TIE_VALUE = {"constant1": "1'b1", "constant0": "1'b0"}  # Cell.kind of a tie cell -> its value

# Named functions a cell can be recognised as (Cell.logic), besides AND / NAND / OR /
# NOR / XOR / XNOR of all inputs, BUF and NOT.  A cell matches when its truth tables
# equal the reference's under some assignment of its pins (names and order do not
# matter).  Liberty syntax: ! NOT, * AND, + OR, ^ XOR.
REFERENCE = {
    "AOI21": "!(A+(B*C))", "AOI22": "!((A*B)+(C*D))", "AOI211": "!((A*B)+C+D)",
    "AOI221": "!((A*B)+(C*D)+E)", "AOI222": "!((A*B)+(C*D)+(E*F))", "AOI33": "!((A*B*C)+(D*E*F))",
    "OAI21": "!(A*(B+C))", "OAI22": "!((A+B)*(C+D))", "OAI211": "!((A+B)*C*D)",
    "OAI221": "!((A+B)*(C+D)*E)", "OAI222": "!((A+B)*(C+D)*(E+F))", "OAI33": "!((A+B+C)*(D+E+F))",
    "MUX2": "(S*B)+(!S*A)",
    "FA": {"CO": "(A*B)+(C*(A+B))", "S": "A^B^C"},  # full adder
    "HA": {"CO": "A*B", "S": "A^B"},                # half adder
}
FAMILIES = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUF", "NOT"]  # of any number of inputs
LOGIC_TYPES = FAMILIES + list(REFERENCE)  # every value Cell.logic can take


@dataclass
class Cell:
    """A cell type of the loaded libraries (not an instance in the netlist).

    kind: sequential | constant0 | constant1 | buffer | inverter | combinational,
          or other (no Liberty function: filler, tap, ...).
    Combinational cells also get their truth tables, from which `logic` (one of
    LOGIC_TYPES: AND ... NOT of all inputs, or a REFERENCE function such as AOI21 or
    FA, with `logic_pins` saying which cell pin plays which reference pin) and
    `swappable` (inputs that can be exchanged without changing any output) follow --
    so nothing depends on cell names.
    """
    kind: str
    inputs: list[str]          # signal pins; power pins left out
    outputs: list[str]
    functions: dict[str, str]  # output pin -> Liberty function, e.g. {"ZN": "!((A1*A2)+(B1*B2))"}
    area: float                # LEF width x height
    tables: dict[str, tuple[int, ...]] = field(default_factory=dict)  # output -> value in row r,
                                                                      # input i = bit i of r
    logic: str | None = None
    logic_pins: dict[str, str] = field(default_factory=dict)  # reference pin -> cell pin
    swappable: list[list[str]] = field(default_factory=list)  # every input in exactly one group
    data_inputs: list[str] = field(default_factory=list)  # sequential: pins with a setup check
                                                          # (D; scan flops also SI, SE)
    inverted_outputs: list[str] = field(default_factory=list)  # sequential: outputs carrying the
                                                               # inverted state (QN)

    @classmethod
    def build(cls, sequential: bool, inputs: list[str], outputs: list[str],
              functions: dict[str, str], area: float) -> "Cell":
        """A Cell from its library facts: kind, truth tables, logic and swappable inputs."""
        cell = cls("other", inputs, outputs, functions, area)
        if sequential:
            cell.kind = "sequential"
            return cell
        if not functions:
            return cell
        cell.kind = "combinational"
        cell.tables = cls.truth_tables(inputs, functions)
        if len(cell.tables) != len(functions):  # a function we cannot evaluate
            return cell
        rows = range(2 ** len(inputs))
        if len(functions) == 1:
            (table,) = cell.tables.values()
            families = {"AND": tuple(int(r == len(rows) - 1) for r in rows),
                        "OR": tuple(int(r != 0) for r in rows),
                        "XOR": tuple(bin(r).count("1") % 2 for r in rows)}
            if len(inputs) == 1:
                families = {"BUF": (0, 1)}
            for name, ones in families.items() if inputs else ():
                if table == ones:
                    cell.logic = name
                elif table == tuple(1 - v for v in ones):
                    cell.logic = {"BUF": "NOT", "XOR": "XNOR"}.get(name, "N" + name)
            if not inputs:
                cell.kind = f"constant{table[0]}"
        if cell.logic is None and inputs:
            cell.logic, cell.logic_pins = cls.match_reference(inputs, cell.tables)
        cell.kind = {"BUF": "buffer", "NOT": "inverter"}.get(cell.logic, cell.kind)
        groups = [[p] for p in inputs]  # merge groups whose inputs can be swapped
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                swap = lambda r: r ^ (((r >> i ^ r >> j) & 1) * ((1 << i) | (1 << j)))
                if all(t[swap(r)] == t[r] for t in cell.tables.values() for r in rows):
                    gi = next(g for g in groups if inputs[i] in g)
                    gj = next(g for g in groups if inputs[j] in g)
                    if gi is not gj:
                        gi.extend(gj)
                        groups.remove(gj)
        cell.swappable = groups
        return cell

    @staticmethod
    def reference_tables(functions: str | dict[str, str]) -> tuple[list[str], dict[str, tuple[int, ...]]]:
        """(inputs, truth tables) of a REFERENCE entry; its inputs are the pin names it
        uses, sorted; a single function is output Y."""
        functions = functions if isinstance(functions, dict) else {"Y": functions}
        inputs = sorted(set(re.findall(r"[A-Za-z_]\w*", " ".join(functions.values()))))
        return inputs, Cell.truth_tables(inputs, functions)

    @staticmethod
    def match_reference(inputs: list[str], tables: dict[str, tuple[int, ...]]) -> tuple[str | None, dict]:
        """(name, {reference pin: cell pin}) of the REFERENCE function these truth tables
        compute under some assignment of inputs and outputs, or (None, {})."""
        n = len(inputs)
        for name, (ref_inputs, ref_tables) in REFERENCE_TABLES.items():
            if len(ref_inputs) != n or len(ref_tables) != len(tables):
                continue
            for order in permutations(range(n)):  # reference input i is cell input order[i]
                rows = [sum((r >> i & 1) << order[i] for i in range(n)) for r in range(2 ** n)]
                for outs in permutations(tables):  # reference output j is cell output outs[j]
                    if all(tuple(tables[o][row] for row in rows) == ref_tables[ref_out]
                           for ref_out, o in zip(ref_tables, outs)):
                        return name, {**{p: inputs[order[i]] for i, p in enumerate(ref_inputs)},
                                      **dict(zip(ref_tables, outs))}
        return None, {}

    @staticmethod
    def truth_tables(inputs: list[str], functions: dict[str, str]) -> dict[str, tuple[int, ...]]:
        """Each output's value in every row r (input i = bit i of r).  Liberty's operators
        (as OpenSTA prints them, fully parenthesised) map onto Python's bitwise ones:
        ! -> ~, * -> &, + -> |, ^ -> ^; the low bit is the value.  Outputs whose
        function cannot be evaluated are left out."""
        tables = {}
        for out, function in functions.items():
            try:
                code = compile(function.replace("!", "~").replace("*", "&").replace("+", "|"), out, "eval")
                tables[out] = tuple(eval(code, {"__builtins__": {}}, {p: r >> i & 1 for i, p in enumerate(inputs)}) & 1
                                    for r in range(2 ** len(inputs)))
            except (SyntaxError, NameError, TypeError):
                pass
        return tables

    def input_key(self, pins: dict[str, str]) -> tuple:
        """The signals on this cell's inputs, with swappable inputs sorted: two instances
        with the same key compute the same outputs.  pins: {pin: signal}."""
        return tuple(tuple(sorted(pins.get(p, "") for p in group)) for group in self.swappable)


REFERENCE_TABLES = {name: Cell.reference_tables(f) for name, f in REFERENCE.items()}


class Design:
    """One linked netlist.

    block  : the OpenDB dbBlock -- the netlist itself.
    top    : top module name.
    liberty: the Liberty files the cells come from (cell functions for ABC).
    cells  : {cell type name: Cell} for the loaded libraries.
    read_args: how it was read (design_io.read_design arguments), to read the
             original again.
    source_header: the source module header, kept by readers whose writer must
             reproduce it (ICCAD), else None.  Read ports with port_list().
    constant_nets: {net name: 1'b0 | 1'b1} for nets driven by tie cells; kept up
             to date by edit.py (and scan_constants after other edits).
    """

    def __init__(self, path: str, top: str, tech, or_design, liberty: list[str],
                 read_args: dict, source_header: list[tuple] | None = None) -> None:
        self.path = str(path)
        self.top = top
        self.liberty = liberty
        self.read_args = read_args
        self.source_header = source_header
        # ord::Design only keeps a raw pointer to the Tech, which owns the
        # database: the Tech must live as long as the Design.
        self._tech = tech
        self._design = or_design
        self.block = or_design.getBlock()
        self.cells = self.library_cells()
        self.scan_constants()

    def scan_constants(self) -> None:
        """Rebuild constant_nets from the tie cells in the netlist.  edit.py keeps it up
        to date; call this after edits made outside it (e.g. OpenROAD's BLIF reader)."""
        self.constant_nets = {}
        for inst in self.block.getInsts():
            value = TIE_VALUE.get(self.cell(inst).kind)
            for it in inst.getITerms() if value else []:
                if it.isOutputSignal() and it.getNet():
                    self.constant_nets[it.getNet().getName()] = value

    def tcl(self, command: str) -> str:
        """Run a Tcl command in OpenROAD on this design (OpenSTA queries, write_verilog, ...)."""
        return self._design.evalTclString(command)

    def library_cells(self) -> dict[str, Cell]:
        """Every cell type in the loaded libraries: pins from LEF, functions from
        Liberty (one OpenSTA query), the rest from Cell.build."""
        listing = self.tcl(
            "set out {}\n"
            "foreach c [get_lib_cells -quiet */*] {\n"
            "  set cn [get_name $c]\n"
            "  foreach p [get_lib_pins -quiet */$cn/*] { append out \"$cn|[get_name $p]|[$p function]\\n\" }\n"
            "}\n"
            "return $out")
        functions = {}  # cell -> {pin: function}; the first library defining a cell wins
        listed = {}     # cell -> its Liberty pins in order, including a flop's state variables
        for line in listing.splitlines():
            cell, pin, function = line.split("|", 2)
            listed.setdefault(cell, []).append(pin)
            if function:
                functions.setdefault(cell, {}).setdefault(pin, function.replace(" ", ""))
        cells = {}
        for lib in self._tech.getDB().getLibs():
            for m in lib.getMasters():
                pins = [mt for mt in m.getMTerms() if mt.getSigType() not in ("POWER", "GROUND")]
                ins = [mt.getName() for mt in pins if mt.getIoType() == "INPUT"]
                outs = [mt.getName() for mt in pins if mt.getIoType() == "OUTPUT"]
                fn = {p: f for p, f in functions.get(m.getName(), {}).items() if p in outs}
                cell = cells.setdefault(m.getName(), Cell.build(m.isSequential(), ins, outs, fn,
                                                                m.getWidth() * m.getHeight()))
                # A flop's state variables are Liberty pins that are not LEF pins, declared
                # as ff (IQ, IQN): state, then inverted state.  Outputs naming the second
                # (or the negated first) carry the inverted state.
                state = [p for p in listed.get(m.getName(), []) if m.findMTerm(p) is None][:2]
                if cell.kind == "sequential" and len(state) == 2:
                    cell.inverted_outputs = [p for p, f in fn.items() if f in (state[1], f"!{state[0]}")]
        # Data inputs of sequential cells: the pins the library gives a setup check
        # (timing arc "CK -> D", role setup) -- not clock, set or reset pins.
        sequential = " ".join(name for name, c in cells.items() if c.kind == "sequential")
        arcs = self.tcl(
            "set out {}\n"
            f"foreach cn {{{sequential}}} {{\n"
            "  foreach a [[lindex [get_lib_cells */$cn] 0] timing_arc_sets] {\n"
            "    if {[$a role] eq \"setup\"} { append out \"$cn|[lindex [$a full_name] end]\\n\" }\n"
            "  }\n"
            "}\n"
            "return $out")
        for line in arcs.splitlines():
            cell, pin = line.split("|")
            if pin not in cells[cell].data_inputs:
                cells[cell].data_inputs.append(pin)
        return cells

    def port_list(self) -> list[tuple[str, str, int | None, int | None]]:
        """[(name, direction, msb, lsb)] of the top module, msb/lsb None for a scalar port:
        the source header when the reader kept it, else as OpenSTA knows the ports
        (bus ranges as declared)."""
        if self.source_header is not None:
            return self.source_header
        listing = self.tcl(
            "set it [[[sta::top_instance] cell] port_iterator]; set out {}\n"
            "while {[$it has_next]} {\n"
            "  set p [$it next]; set bits {}\n"
            "  if {[$p is_bus]} { set m [$p member_iterator]; while {[$m has_next]} { lappend bits [get_name [$m next]] } }\n"
            "  append out \"[get_name $p]|[sta::port_direction $p]|[lindex $bits 0]|[lindex $bits end]\\n\"\n"
            "}\n"
            "return $out")
        index = lambda bit: int(re.search(r"\[(\d+)\]$", bit)[1]) if bit else None
        return [(name, direction, index(first), index(last))
                for name, direction, first, last in (line.split("|") for line in listing.splitlines())]

    def cell(self, inst) -> Cell:
        """The cell type of an OpenDB instance."""
        return self.cells[inst.getMaster().getName()]

    def is_gate(self, inst) -> bool:
        """A combinational logic gate: not a flip-flop, tie cell or physical-only cell."""
        return self.cells[inst.getMaster().getName()].kind in ("combinational", "buffer", "inverter")

    def net(self, name: str):  # -> odb.dbNet
        """A net by name as written (r3[0]) or as OpenDB stores escaped names."""
        net = self.block.findNet(name) or self.block.findNet(self.escaped(name))
        if net is None:
            raise ValueError(f"signal {name!r} not found")
        return net

    def inst(self, name: str):  # -> odb.dbInst
        """An instance by name as written (r3_reg[0]) or as OpenDB stores escaped
        names (r3_reg\\[0\\])."""
        inst = self.block.findInst(name) or self.block.findInst(self.escaped(name))
        if inst is None:
            raise ValueError(f"instance {name!r} not found")
        return inst

    def master(self, name: str):  # -> odb.dbMaster
        """A cell type by name, as OpenDB's dbMaster (to create instances)."""
        master = self._tech.getDB().findMaster(name)
        if master is None:
            raise ValueError(f"cell {name!r} not found in the libraries")
        return master

    def smallest_cell(self, kind: str) -> tuple[str, str, str]:
        """(cell, input pin, output pin) of the smallest cell of `kind` ("buffer" or
        "inverter") in the libraries, by area then name -- found by function, not name."""
        found = sorted((c.area, name) for name, c in self.cells.items() if c.kind == kind)
        if not found:
            raise ValueError(f"no {kind} cell found in the design's Liberty libraries")
        name = found[0][1]
        return name, self.cells[name].inputs[0], self.cells[name].outputs[0]

    def buffer_cell(self) -> tuple[str, str, str]:
        """(cell, input pin, output pin) of the smallest buffer."""
        return self.smallest_cell("buffer")

    def signal(self, net) -> str:
        """Verilog expression for a net: its name, or 1'b0 / 1'b1 for a constant
        (a tie-cell output or a literal constant net)."""
        name = net.getName()
        return self.constant_nets.get(name) or CONSTANT.get(net.getSigType(), name)

    @staticmethod
    def escaped(name: str) -> str:
        """OpenDB's form of an escaped Verilog name: `\\r3_reg[0] ` is stored as r3_reg\\[0\\]."""
        return re.sub(r"(?<!\\)([\[\]])", r"\\\1", name)
