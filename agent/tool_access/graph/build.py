"""The tool knowledge graph: ontology.ttl + knowledge.ttl + facts generated from tools.py.

    ./openroad_python.sh agent/tool_access/graph/build.py [out.ttl]   writes the whole graph as Turtle

Generated (see README.md, "Comes from"): every Tools method as a ca:Tool with its summary,
description, category (engine package > engine file), parameters (kinds, keywords, allowed
values, defaults), examples, alternatives (and confusableWith both ways), changesDesign and
requiresDesign.  Only the Session category (tools that call no engine package) is named
here; everything else comes from the code, so the graph follows it.
"""
import inspect
import re
import sys
import typing
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, SKOS, XSD

if __name__ == "__main__":  # run as a script: make circuitAnt's packages importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from agent.tools import Tools, changes_design, parameter  # noqa: E402
from eda_engine import analysis, transformation, verification  # noqa: E402

HERE = Path(__file__).resolve().parent
CA = Namespace("https://w3id.org/circuitant/ontology#")
CT = Namespace("https://w3id.org/circuitant/tools#")
PACKAGES = {"Analysis": analysis, "Transformation": transformation, "Verification": verification}


def build(knowledge: bool = True) -> Graph:
    """The graph; knowledge=False leaves out knowledge.ttl (concepts, followedBy)."""
    g = Graph()
    g.bind("ca", CA)
    g.bind("ct", CT)
    g.bind("skos", SKOS)
    g.parse(HERE / "ontology.ttl")
    if knowledge:
        g.parse(HERE / "knowledge.ttl")

    def node(label: str, kind, description: str = "") -> None:
        g.add((CT[label], RDF.type, kind))
        g.add((CT[label], RDFS.label, Literal(label)))
        if description:
            g.add((CT[label], CA.description, Literal(description)))

    def first_line(doc: str | None) -> str:
        return (inspect.cleandoc(doc or "").splitlines() or [""])[0]

    node("Session", CA.Category, "Starting a testcase, reading and writing the design.")
    for package, module in PACKAGES.items():
        node(package, CA.Category, first_line(module.__doc__))

    keywords = {}  # keyword -> meaning, to check every kind gives the same meaning
    for name, fn in inspect.getmembers(Tools, inspect.isfunction):
        if name.startswith("_"):
            continue
        tool = CT[name]
        doc = inspect.getdoc(fn)
        node(name, CA.Tool, doc)
        g.add((tool, CA.summary, Literal(summary(doc))))

        # Category: the engine package and file of the function the method calls.
        names = fn.__code__.co_names
        package = next((p for p in PACKAGES if PACKAGES[p].__name__.split(".")[-1] in names), None)
        engine = next((getattr(PACKAGES[package], n) for n in names
                       if package and callable(getattr(PACKAGES[package], n, None))), None)
        if engine:
            module = sys.modules[engine.__module__]
            topic = engine.__module__.split(".")[-1].capitalize()
            node(topic, CA.Category, first_line(module.__doc__))
            g.add((CT[topic], SKOS.broader, CT[package]))
            g.add((tool, CA.inCategory, CT[topic]))
            g.add((tool, CA.engineFunction, Literal(f"{engine.__module__}.{engine.__name__}")))
        else:
            g.add((tool, CA.inCategory, CT.Session))
        g.add((tool, CA.changesDesign, Literal(changes_design(fn))))
        g.add((tool, CA.requiresDesign, Literal("_d" in names)))

        # Parameters, with their kinds, keywords and allowed values.
        hints = typing.get_type_hints(fn, include_extras=True)
        for position, (pname, p) in enumerate(list(inspect.signature(fn).parameters.items())[1:], 1):
            par = CT[f"{name}.{pname}"]
            base, notes = parameter(hints[pname])
            g.add((par, RDF.type, CA.Parameter))
            g.add((par, RDFS.label, Literal(pname)))
            g.add((tool, CA.hasParameter, par))
            g.add((par, CA.position, Literal(position)))
            g.add((par, CA.required, Literal(p.default is p.empty)))
            g.add((par, CA.datatype, {str: XSD.string, int: XSD.integer, bool: XSD.boolean}[base]))
            if p.default not in (p.empty, None):
                g.add((par, CA.default, Literal(str(p.default))))
            own = "; ".join(n.text() for n in notes if n.description and not n.kind)
            if own:
                g.add((par, CA.description, Literal(own)))
            for value in next((n.values for n in notes if n.values), ()):
                g.add((par, CA.allowedValue, Literal(value)))
            for kind in (n for n in notes if n.kind):
                node(kind.kind, CA.ValueKind, kind.description)
                g.add((par, CA.ofKind, CT[kind.kind]))
                for keyword, meaning in kind.keywords:
                    if keywords.setdefault(keyword, meaning) != meaning:
                        raise ValueError(f"{keyword} means {keywords[keyword]!r} and {meaning!r}")
                    node(keyword, CA.Keyword, meaning)
                    g.add((CT[kind.kind], CA.hasKeyword, CT[keyword]))

        # Examples (with their argument values) and alternatives.
        for i, (phrase, args) in enumerate(getattr(fn, "examples", []), 1):
            example = CT[f"{name}.example{i}"]
            g.add((example, RDF.type, CA.Example))
            g.add((example, RDFS.label, Literal(phrase)))
            g.add((tool, CA.hasExample, example))
            for pname, value in args.items():
                binding = BNode()
                g.add((binding, RDF.type, CA.Binding))
                g.add((binding, CA.forParameter, CT[f"{name}.{pname}"]))
                g.add((binding, CA.value, Literal(str(value))))
                g.add((example, CA.hasBinding, binding))
        for other, when in getattr(fn, "instead", {}).items():
            alternative = BNode()
            g.add((alternative, RDF.type, CA.Alternative))
            g.add((alternative, CA.useTool, CT[other]))
            g.add((alternative, CA.when, Literal(when)))
            g.add((tool, CA.hasAlternative, alternative))
            g.add((tool, CA.confusableWith, CT[other]))
            g.add((CT[other], CA.confusableWith, tool))
    return g


def summary(doc: str) -> str:
    """The first sentence of a docstring: up to a period followed by two spaces, a line break or the end
    (a period followed by one space, as in "e.g. g12", does not end it)."""
    text = inspect.cleandoc(doc)
    end = re.search(r"\.(?=  |\n|$)", text)
    return " ".join(text[:end.end() if end else len(text)].split())


if __name__ == "__main__":
    turtle = build().serialize(format="turtle")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(turtle)
    else:
        print(turtle)
