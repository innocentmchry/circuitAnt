"""Test automation for the engine (no LLM): each engine function checked against known answers.

    circuitAnt/openroad_python.sh circuitAnt/tests_automation/test_engine.py

Expected test10 values were cross-checked against the cada1067 engine.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eda_engine import analysis, read_design, write_design  # noqa: E402

TEST10 = HERE / "test10.v"  # ICCAD test10 (4836 gates); expected values cross-checked with cada1067
EXAMPLE = HERE.parent / "examples" / "dsp_circuit_1_16_bit_Nan45.v"
NANGATE = Path.home() / "OpenROAD" / "test" / "Nangate45"
NANGATE_CONFIG = {"format": "generic", "lef": [str(NANGATE / "Nangate45_tech.lef"), str(NANGATE / "Nangate45_stdcell.lef")],
                  "liberty": [str(NANGATE / "Nangate45_typ.lib")]}  # top: the netlist's one module


def connectivity(d):
    return {i.getName(): (i.getMaster().getName(),
                          [(it.getMTerm().getName(), d.signal(it.getNet()) if it.getNet() else None)
                           for it in i.getITerms() if it.getMTerm().getSigType() not in ("POWER", "GROUND")])
            for i in d.block.getInsts()}


class Test10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = read_design(TEST10)

    def test_count_gates(self):
        self.assertEqual(analysis.count_gates(self.d), {
            "total_gates": 4836,
            "breakdown": {"AND": 903, "OR": 1076, "NOT": 1293, "NAND": 0, "NOR": 339,
                          "XOR": 82, "XNOR": 807, "BUF": 336, "DFF": 0}})

    def test_g825_pins(self):
        self.assertEqual(connectivity(self.d)["g825"],
                         ("AND", [("Y", "n1096"), ("A", "n1003"), ("B", "n1065")]))

    def test_max_depth(self):
        r = analysis.get_max_depth(self.d, "n12", "n25[0]")
        self.assertEqual(r["depth"], 10)
        self.assertEqual(len(r["path"]), 11)
        self.assertEqual((r["path"][0], r["path"][-1]), ("n12", "n25[0]"))

    def test_all_paths(self):
        r = analysis.find_all_paths(self.d, "n14", "n25[0]")
        self.assertEqual((r["count"], r["exists"]), (1, True))
        self.assertEqual(r["paths"], [{"start": "n14", "end": "n25[0]", "path": [
            "n14", "n4789", "n4776", "n4777", "n4767", "n4768",
            "n3971", "n273", "n449", "n872", "n25[0]"]}])

    def test_many_paths_go_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = analysis.find_all_paths(self.d, "n12", "n25[16]", out_dir=tmp)
            self.assertEqual(r["count"], 4475)
            self.assertEqual(len(r["paths"]), 5)
            self.assertEqual(len(Path(r["file_path"]).read_text().splitlines()), 4475)

    def test_unknown_signal(self):
        with self.assertRaises(ValueError):
            analysis.get_max_depth(self.d, "no_such_net", "n25[0]")

    def test_write_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.v"
            write_design(self.d, out)
            d2 = read_design(out)
        self.assertEqual(d2.source_header, self.d.source_header)
        self.assertEqual(connectivity(d2), connectivity(self.d))


class Test10Classes(unittest.TestCase):
    """Depth with class keywords; expected values from the cada1067 engine."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(TEST10)

    def test_whole_design(self):
        r = analysis.get_max_depth(self.d, "ANY_START", "ANY_END")
        self.assertEqual((r["depth"], r["start"], r["end"], r["tie_count"]),
                         (53, "n4[0]", "n25[16]", 1))
        self.assertEqual(analysis.get_max_depth(self.d, "ANY_PI", "ANY_PO")["depth"], 53)

    def test_fanin_cone(self):
        self.assertEqual(analysis.get_max_depth(self.d, "ANY_START", "n25[0]")["depth"], 14)

    def test_reverse_direction(self):
        r = analysis.get_max_depth(self.d, "n25[0]", "n12")  # "between n12 and n25[0]"
        self.assertEqual((r["depth"], r["path"][0], r["path"][-1]), (10, "n12", "n25[0]"))
        self.assertIn("note", r)

    def test_endpoint_stats(self):
        r = analysis.get_endpoint_stats(self.d, "ANY_PO", "depth", ">", 4)
        self.assertEqual((r["endpoints"], r["max_value"], r["at_max"], r["count"]),
                         (17, 53, ["n25[16]"], 17))

    def test_bad_keyword_role(self):
        with self.assertRaises(ValueError):
            analysis.get_max_depth(self.d, "ANY_PO", "n25[0]")


class Classes(unittest.TestCase):
    """tests_automation/regs.v: two flip-flops (see the comment in that file)."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")

    def depth(self, source, sink):
        return analysis.get_max_depth(self.d, source, sink)

    def test_register_to_register(self):
        r = self.depth("ANY_DFF_Q", "ANY_DFF_D")
        self.assertEqual((r["depth"], r["path"], r["end"]), (1, ["q2", "n3"], "F1/D"))

    def test_other_classes(self):
        self.assertEqual(self.depth("ANY_START", "ANY_END")["depth"], 2)
        self.assertEqual(self.depth("ANY_PI", "ANY_DFF_D")["depth"], 2)   # a -> n1 -> n3
        self.assertEqual(self.depth("ANY_DFF_Q", "ANY_PO")["depth"], -1)  # flops don't reach y
        self.assertEqual(self.depth("a", "q1")["depth"], -1)              # F1 cuts the path


class Paths(unittest.TestCase):
    """find_all_paths options on tests_automation/regs.v."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")

    def paths(self, *args, **kw):
        r = analysis.find_all_paths(self.d, *args, **kw)
        return r["count"], [(p["start"], p["end"], p["path"]) for p in r["paths"]]

    def test_register_to_register(self):  # F1/Q wired straight to F2/D is a 0-gate path
        self.assertEqual(self.paths("ANY_DFF_Q", "ANY_DFF_D"),
                         (2, [("F1/Q", "F2/D", ["q1"]), ("F2/Q", "F1/D", ["q2", "n3"])]))

    def test_avoid(self):
        self.assertEqual(self.paths("a", "n3")[0], 1)
        self.assertEqual(self.paths("a", "n3", avoid="n1")[0], 0)
        self.assertEqual(self.paths("ANY_START", "n3", avoid="n1"),
                         (1, [("F2/Q", "n3", ["q2", "n3"])]))

    def test_max_depth(self):
        self.assertEqual(self.paths("ANY_START", "ANY_END")[0], 6)
        self.assertEqual(self.paths("ANY_START", "ANY_END", max_depth=1)[0], 2)  # q1; q2->n3
        self.assertEqual(self.paths("ANY_PI", "ANY_PO", max_depth=0)[0], 0)


class Cones(unittest.TestCase):
    """get_cone on tests_automation/regs.v (see the comment in that file)."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")

    def test_fanin(self):
        r = analysis.get_cone(self.d, "y")
        self.assertEqual((r["count"], r["gates"], r["by_type"]), (2, ["g1", "g2"], {"AND": 1, "NOT": 1}))

    def test_fanin_through_flop(self):  # q1 = F1/Q: report F1's D cone (n3 <- g3 <- g1)
        r = analysis.get_cone(self.d, "q1")
        self.assertEqual((r["through_dff"], r["resolved_signal"], r["gates"]), ("F1", "n3", ["g1", "g3"]))

    def test_fanout(self):  # n1 feeds g2, g3; g3 reaches F1 (counted, not crossed)
        r = analysis.get_cone(self.d, "n1", "fanout")
        self.assertEqual(r["gates"], ["F1", "g2", "g3"])

    def test_shared(self):
        r = analysis.get_cone(self.d, "y", shared_with="q1")
        self.assertEqual((r["gates"], r["signal_cone_count"], r["other_cone_count"]), (["g1"], 2, 2))

    def test_largest_cone(self):
        r = analysis.get_endpoint_stats(self.d, "ANY_PO", "cone_size")
        self.assertEqual((r["max_value"], r["at_max"]), (2, ["y"]))


class FanoutAndGates(unittest.TestCase):
    """get_fanout / find_gates on tests_automation/regs.v (see the comment in that file)."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")

    def test_signal_fanout(self):  # n1 feeds g2.A and g3.A
        r = analysis.get_fanout(self.d, "n1")
        self.assertEqual((r["fanout"], [x["gate"] for x in r["loads"]]), (2, ["g2", "g3"]))

    def test_clock_fanout_and_gate_output(self):
        self.assertEqual(analysis.get_fanout(self.d, "clk")["fanout"], 2)  # F1.CK, F2.CK
        r = analysis.get_fanout(self.d, "g1")  # gate name -> its output n1
        self.assertEqual((r["output_signal"], r["fanout"]), ("n1", 2))

    def test_max_fanout(self):
        r = analysis.get_fanout(self.d, "ANY_SIGNAL")
        self.assertEqual((r["max_fanout"], r["at_max"]), (2, ["clk", "n1", "rst"]))
        self.assertEqual(analysis.get_fanout(self.d, "ANY_PI")["at_max"], ["clk", "rst"])

    def test_find_by_type_and_name(self):
        self.assertEqual([g["gate"] for g in analysis.find_gates(self.d, "DFF")["gates"]], ["F1", "F2"])
        g = analysis.find_gates(self.d, name="g3")["gates"][0]
        self.assertEqual((g["type"], g["pins"]), ("OR", {"Y": "n3", "A": "n1", "B": "q2"}))

    def test_find_by_connection(self):
        r = analysis.find_gates(self.d, "DFF", connected_to="clk", pin="CK")
        self.assertEqual(r["count"], 2)
        self.assertEqual([g["gate"] for g in analysis.find_gates(self.d, connected_to="n1")["gates"]],
                         ["g1", "g2", "g3"])  # driver g1 and loads g2, g3


class PortsCutsPatterns(unittest.TestCase):
    """get_ports / find_cut_points on tests_automation/regs.v, pattern queries on tests_automation/patterns.v."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")
        cls.p = read_design(HERE / "patterns.v")

    def test_ports(self):
        r = analysis.get_ports(self.d)
        self.assertEqual((r["inputs"]["ports"], r["inputs"]["bits"], r["outputs"]["ports"]), (4, 4, 1))
        self.assertEqual(analysis.get_ports(read_design(TEST10), "output")["outputs"]["list"],
                         [{"name": "n25", "width": 17, "range": "[16:0]"}])

    def test_constant_inputs(self):  # both flops have SN tied to 1'b1
        self.assertEqual(analysis.find_gates(self.d, connected_to="1'b1")["count"], 2)

    def test_articulation_points(self):  # a -> g1 -> n1 -> g2 -> y: both gates are unavoidable
        r = analysis.find_cut_points(self.d, "a", "y")
        self.assertEqual((r["path_exists"], r["gates"], r["signals"]), (True, ["g1", "g2"], ["n1", "y"]))

    def test_cut_signal(self):  # every input-to-output path passes n1
        r = analysis.find_cut_points(self.d, signal="n1")
        self.assertEqual((r["is_on_path"], r["is_cut"], r["cut_pairs"]), (True, True, [["a", "y"], ["b", "y"]]))

    def test_patterns(self):
        pairs = analysis.find_inverter_pairs(self.p)
        self.assertEqual([(m["first"], m["second"]) for m in pairs["matches"]], [("g1", "g2"), ("g11", "g12")])
        dup = analysis.find_duplicates(self.p)
        self.assertEqual((dup["count"], dup["matches"][0]["gates"]), (1, ["g3", "g4"]))
        eh = analysis.find_enable_hold(self.p)
        self.assertEqual({m["dff"]: m["classification"] for m in eh["matches"]},
                         {"F1": "mux_or_gated_hold_candidate", "F2": "direct_hold"})


class ViewHelpers(unittest.TestCase):
    """utils.netlist_view / depends_on / trace_buf_not on tests_automation/patterns.v."""

    @classmethod
    def setUpClass(cls):
        from eda_engine import utils
        cls.utils = utils
        cls.d = read_design(HERE / "patterns.v")
        cls.drivers, cls.dffs = utils.netlist_view(cls.d)

    def test_trace_buf_not(self):
        self.assertEqual(self.utils.trace_buf_not(self.d, self.drivers, "n2"), ("a", False))   # not(not(a))
        self.assertEqual(self.utils.trace_buf_not(self.d, self.drivers, "n1"), ("a", True))    # not(a)
        self.assertEqual(self.utils.trace_buf_not(self.d, self.drivers, "y2"), ("n4", False))  # buf -> stops at and
        self.assertEqual(self.utils.trace_buf_not(self.d, self.drivers, "hq2"), ("h", False))  # F2's direct hold

    def test_depends_on(self):
        self.assertTrue(self.utils.depends_on(self.drivers, "m", "q"))         # F1's D depends on its Q
        self.assertFalse(self.utils.depends_on(self.drivers, "m", "q", stop={"sel_q"}))
        self.assertFalse(self.utils.depends_on(self.drivers, "y1", "q"))


class Equations(unittest.TestCase):
    """derive_boolean_equation on tests_automation/regs.v and tests_automation/patterns.v."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(HERE / "regs.v")

    def test_combinational(self):
        r = analysis.derive_boolean_equation(self.d, "y")
        self.assertEqual((r["equation"], r["primary_inputs_used"], r["primary_input_only"]),
                         ("y = ~((a * b))", ["a", "b"], True))

    def test_state_input(self):  # n3 = or(n1, q2): depends on flip-flop F2's output
        r = analysis.derive_boolean_equation(self.d, "n3")
        self.assertEqual((r["equation"], r["state_inputs"], r["primary_input_only"]),
                         ("n3 = ((a * b) + q2)", ["q2"], False))

    def test_flop_and_input(self):
        self.assertEqual(analysis.derive_boolean_equation(self.d, "q1")["driver_type"], "dff")
        self.assertEqual(analysis.derive_boolean_equation(self.d, "a")["equation"], "a = a")

    def test_shared_terms(self):  # forced: shared signals stay named
        from eda_engine.analysis import expressions
        old, expressions.MAX_FLAT_CHARS = expressions.MAX_FLAT_CHARS, 0
        try:
            r = analysis.derive_boolean_equation(read_design(HERE / "patterns.v"), "y1")
        finally:
            expressions.MAX_FLAT_CHARS = old
        self.assertEqual(r["equation"], "y1 = ((a * b) + ~(~(a)))")  # nothing used twice here
        r = analysis.derive_boolean_equation(read_design(TEST10), "n25[16]")
        self.assertIn("shared_terms", r)


class Verification(unittest.TestCase):
    """verification package (BLIF + ABC) on tests_automation/regs.v and tests_automation/patterns.v."""

    @classmethod
    def setUpClass(cls):
        from eda_engine import verification
        cls.v = verification
        cls.p = read_design(HERE / "patterns.v")

    def test_unchanged_design(self):
        r = self.v.check_equivalence(read_design(HERE / "regs.v"))
        self.assertEqual((r["equivalent"], r["method"]), (True, "structurally identical"))

    def edited(self, old: str, new: str):
        """regs.v with one line changed, posing as a transformed regs.v."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "regs.v"
        path.write_text((HERE / "regs.v").read_text().replace(old, new))
        d = read_design(path)
        d.path = str(HERE / "regs.v")  # compare against the original file
        return d

    def test_changed_design(self):
        r = self.v.check_equivalence(self.edited("  not g2(y, n1);", "  buf g2(y, n1);"))
        self.assertEqual((r["equivalent"], r["status"]), (False, "FAIL"))
        r = self.v.check_equivalence(self.edited("or  g3(n3, n1, q2);", "or  g3(n3, q2, n1);"))
        self.assertEqual((r["equivalent"], r["status"]), (True, "PASS"))  # inputs swapped: proved, not identical
        self.assertNotEqual(r["method"], "structurally identical")
        r = self.v.check_equivalence(self.edited("  or  g3(n3, n1, q2);", "  or  g3(n3, n1, q1);"))
        self.assertEqual((r["status"], r["failing_outputs"]), ("FAIL", ["_cut_F1_D"]))

    def test_constant(self):  # nand(n1, 1) == not(n1): ABC must see the tie cell as a constant
        r = self.v.check_equivalence(self.edited("  not g2(y, n1);", "  nand g2(y, n1, 1'b1);"))
        self.assertEqual(r["status"], "PASS")
        r = self.v.check_equivalence(self.edited("  not g2(y, n1);", "  nand g2(y, n1, 1'b0);"))
        self.assertEqual(r["status"], "FAIL")

    def test_flop_pin_net_changed(self):  # a buffer before F1.D: cut points are named after the flop
        r = self.v.check_equivalence(self.edited(".D(n3), .Q(q1));", ".D(n3b), .Q(q1));\n  wire n3b;\n  buf gb(n3b, n3);"))
        self.assertEqual(r["status"], "PASS")

    def test_signal_equivalence(self):
        self.assertTrue(self.v.check_signal_equivalence(self.p, "n3", "n4")["equivalent"])   # and(a,b), and(b,a)
        self.assertTrue(self.v.check_signal_equivalence(self.p, "n2", "a")["equivalent"])    # not(not(a))
        self.assertFalse(self.v.check_signal_equivalence(self.p, "n1", "a")["equivalent"])   # not(a)

    def test_symmetry(self):
        self.assertTrue(self.v.check_symmetry(self.p, "n3", "a", "b")["symmetric"])    # a & b
        self.assertFalse(self.v.check_symmetry(self.p, "y1", "a", "b")["symmetric"])   # (a & b) | a == a
        with self.assertRaises(ValueError):
            self.v.check_symmetry(self.p, "y1", "n1", "b")


class FlopLinking(unittest.TestCase):
    """A dff whose pins are all nets (SN not tied to 1'b1) must link (primitives.lef power pins)."""

    def test_sn_on_a_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regs.v"
            path.write_text((HERE / "regs.v").read_text().replace(".SN(1'b1)", ".SN(a)"))
            d = read_design(path)
            out = Path(tmp) / "out.v"
            write_design(d, out)
            text = out.read_text()
        self.assertEqual(connectivity(d)["F1"][1][2], ("SN", "a"))
        self.assertNotIn("VDD", text)


class PinOrder(unittest.TestCase):
    """Positional connections must land on the right pins (ICCAD cell pin order)."""

    def test_pinmap(self):
        c = connectivity(read_design(HERE / "pinmap.v"))
        self.assertEqual(c["g_and"], ("AND", [("Y", "y_and"), ("A", "a"), ("B", "b")]))
        self.assertEqual(c["g_not"], ("NOT", [("Y", "y_not"), ("A", "a")]))
        self.assertEqual(c["g_dff"], ("DFF", [("CK", "ck"), ("RN", "rn"), ("SN", "sn"),
                                              ("D", "d"), ("Q", "q")]))


class Transformations(unittest.TestCase):
    """transformation package on tests_automation/patterns.v, tests_automation/transforms.v and test10; every
    transformed design must still be equivalent to its file."""

    @classmethod
    def setUpClass(cls):
        from eda_engine import transformation, verification
        cls.t, cls.v = transformation, verification

    def assertEquivalent(self, d):
        self.assertEqual(self.v.check_equivalence(d)["status"], "PASS")

    def test_full_list(self):  # inline_limit=None: everything, no file
        d = read_design(TEST10)
        r = analysis.find_gates(d, "AND", inline_limit=None)
        self.assertEqual((len(r["gates"]), r["count"], "file_path" in r), (903, 903, False))

    def test_rename(self):
        d = read_design(HERE / "patterns.v")
        self.assertEqual(self.t.rename_gate(d, "g3", "renamed_gate")["new_name"], "renamed_gate")
        self.t.rename_wire(d, "n3", "renamed_wire")
        self.assertEqual(connectivity(d)["renamed_gate"][1][0], ("Y", "renamed_wire"))
        with self.assertRaises(ValueError):
            self.t.rename_gate(d, "g4", "g5")      # name taken
        with self.assertRaises(ValueError):
            self.t.rename_wire(d, "y1", "z")       # a port
        self.assertEquivalent(d)

    def test_dangling(self):
        d = read_design(HERE / "transforms.v")
        r = self.t.remove_dangling_gates(d)
        self.assertEqual((r["gates_removed"], r["by_type"]), (2, {"AND": 1, "NOT": 1}))
        self.assertIsNone(d.block.findNet("d1"))
        self.assertEquivalent(d)

    def test_inverter_pairs(self):
        d = read_design(HERE / "patterns.v")  # g1->g2 (input a), g11->g12 (into F2.D)
        r = self.t.collapse_inverter_pairs(d)
        self.assertEqual((r["pairs_collapsed"], r["gates_removed"]), (2, 4))
        self.assertEqual(connectivity(d)["F2"][1][3], ("D", "h"))  # F2.D now on F2.Q's net
        self.assertEquivalent(d)

    def test_inverter_pair_between_ports(self):  # both ends are ports: a buffer joins them
        d = read_design(HERE / "transforms.v")
        self.assertEqual(self.t.collapse_inverter_pairs(d)["pairs_collapsed"], 1)
        self.assertEqual([c for c in connectivity(d).values() if ("Y", "y3") in c[1]],
                         [("BUF", [("Y", "y3"), ("A", "a")])])
        self.assertEquivalent(d)

    def test_duplicates(self):
        d = read_design(HERE / "patterns.v")  # g3 = and(a, b), g4 = and(b, a)
        self.assertEqual(self.t.merge_duplicate_gates(d)["gates_merged"], 1)
        self.assertEqual(analysis.find_duplicates(d)["count"], 0)
        self.assertEquivalent(d)

    def test_constants(self):
        d = read_design(HERE / "transforms.v")
        r = self.t.simplify_gates_with_constant_inputs(d)
        self.assertEqual((r["simplified_gates"], r["replacement_breakdown"], r["skipped"]),
                         (1, {"NAND(x, 1'b1) -> NOT": 1}, {"AND(x, 1'b0) -> 1'b0": 1}))
        self.assertEqual(connectivity(d)["g1"], ("NOT", [("Y", "y1"), ("A", "a")]))
        self.assertEquivalent(d)

    def test_fanout(self):  # fewest buffers: ceil((F - k) / (k - 1)) per signal
        d = read_design(TEST10)
        fanouts = [sum(1 for it in n.getITerms() if it.isInputSignal()) for n in d.block.getNets()]
        r = self.t.insert_buffers_for_fanout(d, 4)
        self.assertEqual(r["buffers_inserted"], sum(-(-(f - 4) // 3) for f in fanouts if f > 4))
        self.assertLessEqual(max(sum(1 for it in n.getITerms() if it.isInputSignal())
                                 for n in d.block.getNets()), 4)
        self.assertEquivalent(d)

    def test_dedicated_buffers(self):
        d = read_design(HERE / "patterns.v")  # a feeds g1, g3, g4
        self.assertEqual(self.t.insert_dedicated_buffers_for_loads(d, "a")["buffers_inserted"], 3)
        loads = analysis.get_fanout(d, "a")["loads"]
        self.assertEqual(({x["type"] for x in loads}, len(loads)), ({"BUF"}, 3))
        self.assertEquivalent(d)
        with tempfile.TemporaryDirectory() as tmp:  # new cells and nets write and read back
            write_design(d, Path(tmp) / "out.v")
            self.assertEqual(analysis.count_gates(read_design(Path(tmp) / "out.v"))["breakdown"]["BUF"], 4)


class Resynthesis(unittest.TestCase):
    """transformation.resynthesis (OpenROAD BLIF writer/reader + ABC) on test10 and tests_automation/regs.v;
    every result must still be equivalent to its file."""

    @classmethod
    def setUpClass(cls):
        from eda_engine import transformation, verification
        cls.t, cls.v = transformation, verification

    def assertEquivalent(self, d):
        self.assertEqual(self.v.check_equivalence(d)["status"], "PASS")

    def logic_types(self, d, signal=None):
        from eda_engine.transformation.resynthesis import scope
        return {d.cell(d.inst(g)).logic for g in scope(d, signal)}

    def test_decompositions(self):  # every NAND-only / NOR-only circuit computes its gate
        from eda_engine.transformation.resynthesis import DECOMPOSITIONS
        ops = {"NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b)}
        want = {"NOT": lambda a, b: 1 - a, "BUF": lambda a, b: a, "AND": lambda a, b: a & b,
                "OR": lambda a, b: a | b, "NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b),
                "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: 1 - (a ^ b)}
        for target, circuits in DECOMPOSITIONS.items():
            for gate, steps in circuits.items():
                for a in (0, 1):
                    for b in (0, 1):
                        v = {"A": a, "B": b}
                        for out, x, y in steps:
                            v[out] = ops[target](v[x], v[y])
                        self.assertEqual(v["Y"], want[gate](a, b), f"{gate} as {target}")

    def test_reduce_depth(self):
        d = read_design(TEST10)
        r = self.t.reduce_depth(d)
        self.assertLess(r["depth_after"], r["depth_before"])
        self.assertEqual(r["depth_after"], analysis.get_max_depth(d, "ANY_START", "ANY_END")["depth"])
        self.assertEquivalent(d)

    def test_remap(self):  # whole design with flip-flops, then again (a second call id)
        d = read_design(HERE / "regs.v")
        self.t.remap_gates(d, ["NAND", "NOT"])
        self.assertEqual(self.logic_types(d), {"NAND", "NOT"})
        self.t.remap_gates(d, ["NOR"])  # a single universal gate: inverters become NOR(x, x)
        self.assertEqual(self.logic_types(d), {"NOR"})
        self.assertEqual(analysis.count_gates(d)["breakdown"]["DFF"], 2)
        self.assertEquivalent(d)

    def test_remap_cone(self):  # gates outside the cone are untouched
        from eda_engine.transformation.resynthesis import scope
        d = read_design(TEST10)
        cone = scope(d, "n25[0]")
        outside = {i.getName(): i.getMaster().getName() for i in d.block.getInsts() if i.getName() not in cone}
        self.t.remap_gates(d, ["NOR", "NOT"], "n25[0]")
        self.assertEqual(self.logic_types(d, "n25[0]"), {"NOR", "NOT"})
        self.assertEqual({n: d.inst(n).getMaster().getName() for n in outside}, outside)
        self.assertEquivalent(d)

    def test_replace_gate_type(self):
        d = read_design(TEST10)  # 82 XOR
        r = self.t.replace_gate_type(d, "XOR", ["NAND"])
        self.assertEqual((r["replaced"], r["removed"], r["added"]), (82, {"XOR": 82}, {"NAND": 328}))
        r = self.t.replace_gate_type(d, "OR", ["NAND", "NOT"])  # through ABC
        self.assertEqual((r["removed"]["OR"], set(r["added"]) <= {"NAND", "NOT"}), (1076, True))
        self.assertEquivalent(d)


@unittest.skipUnless(NANGATE.is_dir() and EXAMPLE.exists(), "needs the Nangate45 files and examples/")
class Generic(unittest.TestCase):
    """The generic reader on a Synopsys DC netlist mapped to Nangate45."""

    @classmethod
    def setUpClass(cls):
        cls.d = read_design(EXAMPLE, **NANGATE_CONFIG)

    def test_load(self):  # no source header kept: ports come from OpenSTA, ranges as declared
        self.assertEqual((self.d.top, len(list(self.d.block.getInsts())), self.d.source_header), ("dsp_circuit_1_16_bit", 415, None))
        inputs = analysis.get_ports(self.d)["inputs"]["list"]
        self.assertEqual(next(p for p in inputs if p["name"] == "a"), {"name": "a", "width": 16, "range": "[0:15]"})
        self.assertEqual((len(self.d.port_list()), len(list(self.d.block.getBTerms()))), (9, 99))

    def test_cells(self):  # classified from LEF / Liberty, not by name
        kinds = {m: self.d.cells[m].kind for m in ("DFF_X1", "BUF_X1", "INV_X1", "AOI22_X1", "LOGIC1_X1", "FILLCELL_X1")}
        self.assertEqual(kinds, {"DFF_X1": "sequential", "BUF_X1": "buffer", "INV_X1": "inverter",
                                 "AOI22_X1": "combinational", "LOGIC1_X1": "constant1", "FILLCELL_X1": "other"})
        self.assertEqual(self.d.cells["AOI22_X1"].functions, {"ZN": "!((A1*A2)+(B1*B2))"})
        self.assertEqual(self.d.buffer_cell(), ("BUF_X1", "A", "Z"))

    def test_single_module(self):  # the top is inferred; two modules (not flattened) are refused
        self.assertEqual(self.d.top, "dsp_circuit_1_16_bit")
        with tempfile.TemporaryDirectory() as tmp:
            two = Path(tmp) / "two.v"
            two.write_text("// module commented_out (a);\nmodule a (x); input x; endmodule\nmodule b (y); input y; endmodule\n")
            with self.assertRaisesRegex(ValueError, "expected one .flattened. module, found 2: a, b"):
                read_design(two, **NANGATE_CONFIG)

    def test_escaped_names(self):  # \r3_reg[0] in the netlist; a bus bit r3[0]
        self.assertEqual(self.d.inst("r3_reg[0]").getMaster().getName(), "DFF_X1")
        self.assertEqual(self.d.net("r3[0]").getName(), "r3[0]")

    def test_tie_cell(self):
        d = read_design(HERE / "logic1.v", **NANGATE_CONFIG)
        self.assertEqual(d.constant_nets, {"one": "1'b1"})

    def test_write_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_design(self.d, Path(tmp) / "out.v")
            d2 = read_design(Path(tmp) / "out.v", **NANGATE_CONFIG)
        self.assertEqual(connectivity(d2), connectivity(self.d))

    def test_find_gates_by_logic(self):  # logic types by function, anything else by name pattern
        count = lambda t: (analysis.find_gates(self.d, t)["count"], analysis.find_gates(self.d, t)["by_cell"])
        self.assertEqual(count("NAND"), (44, {"NAND2_X1": 41, "NAND4_X1": 3}))
        self.assertEqual(count("AOI21"), (19, {"AOI21_X1": 19}))
        self.assertEqual(count("NAND2_*")[0], 41)
        self.assertEqual(count("FA")[0], 0)
        self.assertEqual({c: self.d.cells[c].logic for c in ("FA_X1", "HA_X1", "MUX2_X1", "OAI222_X1")},
                         {"FA_X1": "FA", "HA_X1": "HA", "MUX2_X1": "MUX2", "OAI222_X1": "OAI222"})

    def test_flop_pins(self):  # from setup arcs and the ff (IQ, IQN) declaration, not pin names
        roles = lambda c: (self.d.cells[c].outputs, self.d.cells[c].data_inputs, self.d.cells[c].inverted_outputs)
        self.assertEqual(roles("DFF_X1"), (["Q", "QN"], ["D"], ["QN"]))
        self.assertEqual(roles("SDFF_X1"), (["Q", "QN"], ["D", "SE", "SI"], ["QN"]))
        self.assertEqual(read_design(HERE / "demo.v").cells["DFF"].data_inputs, ["D"])

    def test_flop_endpoints(self):  # QN outputs start paths too: 45 Q + 12 QN connected
        from eda_engine import utils
        starts = [label for label, _ in utils.endpoints(self.d, "ANY_DFF_Q", "source")]
        self.assertEqual((len(starts), sum(s.endswith("/QN") for s in starts)), (57, 12))
        self.assertEqual(len(utils.endpoints(self.d, "ANY_DFF_D", "sink")), 48)
        self.assertEqual(analysis.get_max_depth(self.d, "ANY_DFF_Q", "ANY_DFF_D")["depth"], 27)

    def test_cell_build(self):  # Cell.build alone, from library facts: kind and logic by function
        from eda_engine.design import Cell
        cases = [((True, ["D", "CK"], ["Q"], {"Q": "IQ"}), ("sequential", None)),
                 ((False, [], ["Z"], {"Z": "1"}), ("constant1", None)), ((False, [], ["Z"], {"Z": "0"}), ("constant0", None)),
                 ((False, ["A"], ["Z"], {"Z": "A"}), ("buffer", "BUF")), ((False, ["A"], ["ZN"], {"ZN": "!(A)"}), ("inverter", "NOT")),
                 ((False, ["A", "B"], ["Z"], {"Z": "A^B"}), ("combinational", "XOR")),
                 ((False, ["A1", "A2", "A3"], ["ZN"], {"ZN": "!((A1*A2)*A3)"}), ("combinational", "NAND")),
                 ((False, ["A", "B1", "B2"], ["ZN"], {"ZN": "!(A+(B1*B2))"}), ("combinational", "AOI21")),
                 ((False, ["A", "B", "C"], ["Z"], {"Z": "(A*!B)+C"}), ("combinational", None)),
                 ((False, [], ["Z"], {}), ("other", None))]
        self.assertEqual([(lambda c: (c.kind, c.logic))(Cell.build(*args, 1.0)) for args, _ in cases],
                         [expected for _, expected in cases])
        aoi21 = Cell.build(False, ["A", "B1", "B2"], ["ZN"], {"ZN": "!(A+(B1*B2))"}, 1.0)
        self.assertEqual(aoi21.swappable, [["A"], ["B1", "B2"]])  # A is not interchangeable with B1
        self.assertEqual(aoi21.logic_pins, {"A": "A", "B": "B1", "C": "B2", "Y": "ZN"})
        aoi21_other = Cell.build(False, ["A1", "A2", "B"], ["Y"], {"Y": "!((A1*A2)+B)"}, 1.0)  # pins shuffled
        self.assertEqual((aoi21_other.logic, aoi21_other.logic_pins["A"]), ("AOI21", "B"))
        fa = Cell.build(False, ["A", "B", "CI"], ["SUM", "COUT"], {"COUT": "(A*B)+(CI*(A+B))", "SUM": "CI^(A^B)"}, 1.0)
        self.assertEqual((fa.kind, fa.logic, fa.swappable), ("combinational", "FA", [["A", "B", "CI"]]))
        self.assertEqual((fa.logic_pins["CO"], fa.logic_pins["S"]), ("COUT", "SUM"))  # outputs paired by function

    def test_iccad_cells(self):  # the ICCAD cell library classifies the same way
        d = read_design(HERE / "demo.v")
        self.assertEqual({m: d.cells[m].kind for m in ("DFF", "BUF", "NOT", "AND", "TIEHI", "TIELO")},
                         {"DFF": "sequential", "BUF": "buffer", "NOT": "inverter", "AND": "combinational",
                          "TIEHI": "constant1", "TIELO": "constant0"})


@unittest.skipUnless(NANGATE.is_dir(), "needs the Nangate45 files")
class GenericVerification(unittest.TestCase):
    """verification on Nangate45 cells (tests_automation/mix.v), including multi-output FA / HA."""

    @classmethod
    def setUpClass(cls):
        from eda_engine import verification
        cls.v = verification
        cls.config = NANGATE_CONFIG

    def edited(self, old: str, new: str):
        """mix.v with one line changed, posing as a transformed mix.v."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "mix.v"
        text = (HERE / "mix.v").read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))
        d = read_design(path, **self.config)
        d.path = str(HERE / "mix.v")  # compare against the original file
        return d

    def test_equivalent_rewrites(self):  # interchangeable inputs swapped
        for old, new in [(".B1(b), .B2(nq)", ".B1(nq), .B2(b)"),
                         (".A(a), .B(b), .CI(c)", ".A(c), .B(a), .CI(b)")]:
            self.assertEqual(self.v.check_equivalence(self.edited(old, new))["status"], "PASS", new)

    def test_changed_designs(self):  # the failing outputs say where
        for old, new, failing in [(".A(a), .B1(b)", ".A(b), .B1(a)", ["_cut_f1_D"]),   # AOI21 A <-> B1
                                  (".CO(co), .S(sum)", ".CO(sum), .S(co)", ["co", "h", "y"]),  # FA outputs
                                  ("INV_X1 i1 ( .A(qn), .ZN(nq) );", "BUF_X1 i1 ( .A(qn), .Z(nq) );", ["q", "y"])]:
            r = self.v.check_equivalence(self.edited(old, new))
            self.assertEqual((r["status"], r["failing_outputs"]), ("FAIL", failing), new)

    def test_symmetry(self):
        d = read_design(HERE / "mix.v", **self.config)
        self.assertTrue(self.v.check_symmetry(d, "co", "a", "b")["symmetric"])   # full-adder carry
        self.assertFalse(self.v.check_symmetry(d, "m", "a", "s")["symmetric"])   # mux data vs select

    def test_syntax_error(self):  # a clear message, not OpenROAD's bare message code
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.v"
            bad.write_text("module bad (a, y);\n input a;\n output y;\n INV_X1 i1 ( .A(a) .ZN(y) );\nendmodule\n")
            with self.assertRaisesRegex(ValueError, "syntax error"):
                read_design(bad, **NANGATE_CONFIG)


class Demo(unittest.TestCase):
    """Small netlist with a dff, named ports and a 1'b1 constant."""

    def test_write_and_reload(self):
        d = read_design(HERE / "demo.v")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.v"
            write_design(d, out)
            text = out.read_text()
            d2 = read_design(out)
        self.assertIn("dff g4 (.CK(clk), .RN(rst), .SN(1'b1), .D(y), .Q(q));", text)
        self.assertIn("and g1 (n1, a, b);", text)
        self.assertEqual(connectivity(d2), connectivity(d))
        self.assertEqual(analysis.count_gates(d2)["breakdown"]["DFF"], 1)

    def test_depth_and_flop_cut(self):
        d = read_design(HERE / "demo.v")
        r = analysis.get_max_depth(d, "a", "y")  # a->g1->g2->g3 beats a->g3
        self.assertEqual((r["depth"], r["path"]), (3, ["a", "n1", "n2", "y"]))
        self.assertEqual(analysis.find_all_paths(d, "a", "y")["count"], 2)
        self.assertEqual(analysis.get_max_depth(d, "y", "q")["depth"], -1)  # dff cuts


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="circuitant_tests_") as tmp:
        os.chdir(tmp)  # result lists that tools write to the current directory land here
        unittest.main(verbosity=2)
