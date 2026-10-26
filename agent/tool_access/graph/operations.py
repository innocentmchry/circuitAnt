"""Questions to the tool knowledge graph, each a fixed SPARQL query (plus read-only free SPARQL).

    catalog()          the tool names (optionally each with its one-line summary)
    describe(names)    everything about some tools: arguments, examples, alternatives, tools easily
                       confused with them, what usually follows
    search(text)       TEMPORARY, do not use: the tools best matching a request, by keyword matching
    sparql(query)      the rows of a read-only SELECT / ASK query
"""
import difflib
import math
import re
from collections import defaultdict

from rdflib import Graph, Literal
from rdflib.plugins.sparql import prepareQuery

PREFIXES = """PREFIX ca: <https://w3id.org/circuitant/ontology#>
PREFIX ct: <https://w3id.org/circuitant/tools#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""
MAX_ROWS = 100  # free SPARQL: rows returned at most


class Operations:
    def __init__(self, graph: Graph) -> None:
        self.g = graph
        self.index = self._search_index()

    def rows(self, query: str, **bindings) -> list[tuple]:
        """Rows of a fixed query, as Python values (None for unbound)."""
        return [tuple(v.toPython() if v is not None else None for v in row)
                for row in self.g.query(PREFIXES + query, initBindings=bindings)]

    def tools(self) -> list[str]:
        return sorted(name for (name,) in self.rows("SELECT ?name WHERE { ?t a ca:Tool ; rdfs:label ?name }"))

    def catalog(self, leave_out: set = frozenset(), summaries: bool = False) -> str:
        """The tools, one line each, by name: the name (with summaries: name -- summary)."""
        rows = self.rows("SELECT ?name ?summary WHERE { ?t a ca:Tool ; rdfs:label ?name ; ca:summary ?summary } "
                         "ORDER BY ?name")
        return "\n".join(f"{name} -- {summary}" if summaries else name
                         for name, summary in rows if name not in leave_out)

    def describe(self, names: list[str]) -> dict:
        """{name: everything the graph knows about the tool}; unknown names get suggestions."""
        known = set(self.tools())
        return {name: self._describe(name) if name in known else
                {"error": f"no tool {name!r}", "did_you_mean": close(name, known)} for name in names}

    def _describe(self, name: str) -> dict:
        (summary, description, changes), = self.rows("""SELECT ?s ?d ?c WHERE { ?t rdfs:label ?name ; a ca:Tool ;
            ca:summary ?s ; ca:description ?d ; ca:changesDesign ?c }""", name=Literal(name))
        info = {"description": description, "changes_design": changes, "arguments": {}}
        for pname, required, default, own, kind, kind_text in self.rows("""
                SELECT ?p ?req ?def ?own ?kind ?kindtext WHERE {
                  ?t rdfs:label ?name ; a ca:Tool ; ca:hasParameter ?par .
                  ?par rdfs:label ?p ; ca:required ?req ; ca:position ?pos .
                  OPTIONAL { ?par ca:default ?def } OPTIONAL { ?par ca:description ?own }
                  OPTIONAL { ?par ca:ofKind ?k . ?k rdfs:label ?kind ; ca:description ?kindtext } } ORDER BY ?pos""",
                name=Literal(name)):
            arg = {"required": required}
            if default is not None:
                arg["default"] = default
            text = own or kind_text
            if kind and own:
                text = f"{own} (a {kind})"
            if text:
                arg["meaning"] = text
            values = [v for (v,) in self.rows("""SELECT ?v WHERE { ?t rdfs:label ?name ; a ca:Tool ; ca:hasParameter ?par .
                ?par rdfs:label ?p ; ca:allowedValue ?v }""", name=Literal(name), p=Literal(pname))]
            if values:
                arg["allowed"] = values
            keywords = {k: m for k, m in self.rows("""SELECT ?k ?m WHERE { ?t rdfs:label ?name ; a ca:Tool ;
                ca:hasParameter ?par . ?par rdfs:label ?p ; ca:ofKind/ca:hasKeyword ?kw . ?kw rdfs:label ?k ;
                ca:description ?m }""", name=Literal(name), p=Literal(pname))}
            if keywords:
                arg["keywords"] = keywords
            info["arguments"][pname] = arg
        examples = defaultdict(dict)
        for phrase, pname, value in self.rows("""SELECT ?phrase ?p ?v WHERE { ?t rdfs:label ?name ; a ca:Tool ;
                ca:hasExample ?e . ?e rdfs:label ?phrase .
                OPTIONAL { ?e ca:hasBinding ?b . ?b ca:forParameter/rdfs:label ?p ; ca:value ?v } }""", name=Literal(name)):
            examples[phrase]
            if pname:
                examples[phrase][pname] = value
        if examples:
            info["examples"] = [{"request": p, "arguments": a} if a else {"request": p} for p, a in examples.items()]
        instead = {other: when for other, when in self.rows("""SELECT ?o ?w WHERE { ?t rdfs:label ?name ; a ca:Tool ;
                ca:hasAlternative ?a . ?a ca:useTool/rdfs:label ?o ; ca:when ?w }""", name=Literal(name))}
        if instead:
            info["use_instead"] = instead
        confused = sorted(n for (n,) in self.rows("""SELECT ?n WHERE { ?t rdfs:label ?name ; a ca:Tool ;
                ca:confusableWith/rdfs:label ?n }""", name=Literal(name)) if n not in instead)
        if confused:
            info["easily_confused_with"] = confused  # their use_instead says when to use them
        following = [n for (n,) in self.rows("""SELECT ?n WHERE { ?t rdfs:label ?name ; a ca:Tool ;
                ca:followedBy/rdfs:label ?n }""", name=Literal(name))]
        if following:
            info["usually_followed_by"] = sorted(following)
        return info

    def search(self, text: str, limit: int = 5) -> list[dict]:
        """TEMPORARY -- do not use; kept only until embedding-based search replaces it.  It is keyword
        matching against the hand-written synonyms of knowledge.ttl: it finds only what those word lists
        anticipate, follows the wording they were written from, and understands neither meaning nor
        direction ("does output X depend on input Y" matches paths, but not which way they go).

        The tools best matching a request.  A concept whose label or synonym occurs in the text
        counts most, the more words the more (longest match: a phrase inside a longer matched phrase
        is ignored); then the words shared with each tool's name, summary and examples (weighted by
        how rare the word is among the tools)."""
        query = words(text)
        lowered = f" {normalize(text)} "
        matched = {phrase for phrase, _, _ in self.index["concepts"]
                   if f" {phrase} " in lowered or (len(phrase) > 6 and phrase in lowered)}
        matched = {m for m in matched if not any(m != o and m in o for o in matched)}
        scores, why = defaultdict(float), defaultdict(set)
        for phrase, concept, tool in self.index["concepts"]:
            if phrase in matched:
                scores[tool] += 2.0 + len(phrase.split())
                why[tool].add(concept)
        for tool, bag in self.index["words"].items():
            scores[tool] += sum(self.index["idf"][w] for w in query & bag)
        ranked = sorted((t for t in scores if scores[t] > 0), key=lambda t: (-scores[t], t))[:limit]
        summaries = dict(self.rows("SELECT ?n ?s WHERE { ?t a ca:Tool ; rdfs:label ?n ; ca:summary ?s }"))
        return [{"tool": t, "summary": summaries[t], **({"concepts": sorted(why[t])} if why[t] else {})}
                for t in ranked]

    def _search_index(self) -> dict:
        concepts = [(normalize(phrase), concept, tool) for phrase, concept, tool in self.rows("""
            SELECT ?phrase ?concept ?tool WHERE { ?c a ca:Concept ; rdfs:label ?concept ; ca:answeredBy/rdfs:label ?tool .
            { ?c rdfs:label ?phrase } UNION { ?c skos:altLabel ?phrase } }""")]
        bags = defaultdict(set)
        for tool, text in self.rows("""SELECT ?n ?text WHERE { ?t a ca:Tool ; rdfs:label ?n .
                { ?t ca:summary ?text } UNION { ?t ca:hasExample/rdfs:label ?text } UNION { ?t rdfs:label ?text } }"""):
            bags[tool] |= words(text.replace("_", " "))
        df = defaultdict(int)
        for bag in bags.values():
            for w in bag:
                df[w] += 1
        return {"concepts": concepts, "words": bags,
                "idf": defaultdict(float, {w: math.log(1 + len(bags) / n) for w, n in df.items()})}

    def sparql(self, query: str) -> dict:
        """Rows of a read-only query (SELECT or ASK); the prefixes ca:, ct:, rdfs:, skos: are known."""
        text = query if "PREFIX" in query.upper() else PREFIXES + query
        try:
            prepared = prepareQuery(text)
        except Exception as exc:
            return {"error": f"invalid SPARQL: {exc}"}
        kind = prepared.algebra.name
        if kind not in ("SelectQuery", "AskQuery"):
            return {"error": "only SELECT and ASK queries are allowed"}
        result = self.g.query(prepared)
        if kind == "AskQuery":
            return {"answer": bool(result.askAnswer)}
        names = [str(v) for v in result.vars]
        rows = [[str(v) if v is not None else None for v in row] for row in result]
        return {"columns": names, "rows": rows[:MAX_ROWS], **({"truncated": len(rows)} if len(rows) > MAX_ROWS else {})}


def normalize(text: str) -> str:
    """Lower case, punctuation around words removed, a final s removed (depends -> depend, gates -> gate),
    so phrases match whatever the inflection."""
    out = []
    for w in text.lower().split():
        w = w.strip(".,;:!?()\"'")
        out.append(w[:-1] if w.endswith("s") and len(w) > 3 and not w.endswith("ss") else w)
    return " ".join(w for w in out if w)


def words(text: str) -> set:
    """Lower-case words of 3+ letters, plural s removed, without very common words."""
    stop = {"the", "and", "for", "with", "all", "any", "each", "this", "that", "its", "are", "from", "into",
            "only", "use", "using", "how", "many", "what", "which", "does", "design", "netlist", "gate", "gates",
            "signal", "signals", "report", "list", "find", "compute", "determine", "check", "whether"}
    out = set()
    for w in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()):
        w = w[:-1] if w.endswith("s") and len(w) > 4 and not w.endswith("ss") else w
        if w not in stop:
            out.add(w)
    return out


def close(name: str, known: set) -> list[str]:
    """Up to 3 known names close to a misspelled one."""
    return difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.5)
