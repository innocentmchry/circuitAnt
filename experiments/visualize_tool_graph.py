"""Interactive picture of the tool knowledge graph: every node and edge, in one HTML page.

    ./openroad_python.sh experiments/visualize_tool_graph.py [out.html]

Builds the graph exactly as access mode "graph" does (agent/tool_access/graph/build.py) and
writes a self-contained page (default results/tool_graph.html next to this file): nodes by type
(tool, category, parameter, value kind, keyword, concept, example, design object), edges by
property.  Blank nodes are folded in: an example's argument bindings become its details, and
an alternative ("use X instead when ...") becomes a use_instead edge.  The ontology's own class
and property declarations are left out; only the facts are drawn.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdflib import BNode, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF, RDFS, SKOS  # noqa: E402

from agent.tool_access.graph.build import CA, CT, build  # noqa: E402

TYPES = {CA.Tool: "Tool", CA.Category: "Category", CA.Parameter: "Parameter", CA.ValueKind: "Value kind",
         CA.Keyword: "Keyword", CA.Concept: "Concept", CA.Example: "Example"}
SKIP = {RDF.type, RDFS.label}


def short(term) -> str:
    text = str(term)
    for prefix, ns in (("ca:", str(CA)), ("ct:", str(CT)), ("skos:", str(SKOS)), ("rdfs:", str(RDFS))):
        text = text.replace(ns, prefix)
    return text


def export() -> dict:
    g = build()
    nodes, edges = {}, []

    def node(term, kind: str) -> str:
        key = short(term)
        if key not in nodes:
            label = g.value(term, RDFS.label)
            nodes[key] = {"id": key, "type": kind, "label": str(label) if label else key.split(":")[-1],
                          "details": {}}
        return key

    for cls, kind in TYPES.items():
        for s in g.subjects(RDF.type, cls):
            node(s, kind)
    for s, o in g.subject_objects(CA.denotes):  # design-object classes a value kind refers to
        node(o, "Design object")

    for s in set(g.subjects()):  # (subjects() yields a subject once per triple)
        if isinstance(s, BNode) or short(s) not in nodes:
            continue
        key = short(s)
        for p, o in g.predicate_objects(s):
            if p in SKIP:
                continue
            name = short(p).split(":")[-1]
            if isinstance(o, Literal):
                details = nodes[key]["details"]
                details.setdefault(name, []).append(str(o))
            elif isinstance(o, BNode) and (o, RDF.type, CA.Alternative) in g:  # use X instead when ...
                other, when = g.value(o, CA.useTool), g.value(o, CA.when)
                edges.append({"source": key, "target": short(other), "label": "use_instead", "note": str(when)})
            elif isinstance(o, BNode) and (o, RDF.type, CA.Binding) in g:  # an example's argument value
                arg = g.value(g.value(o, CA.forParameter), RDFS.label)
                nodes[key]["details"].setdefault("arguments", []).append(f"{arg} = {g.value(o, CA.value)}")
            elif isinstance(o, URIRef) and short(o) in nodes:
                edges.append({"source": key, "target": short(o), "label": name})
            elif isinstance(o, URIRef):  # e.g. a datatype IRI
                nodes[key]["details"].setdefault(name, []).append(short(o).split("#")[-1])
    for n in nodes.values():
        n["details"] = {k: v[0] if len(v) == 1 else v for k, v in n["details"].items()}
    return {"nodes": list(nodes.values()), "edges": edges}


PAGE = r"""<title>circuitAnt Tool Graph</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: a header band with type filters, then the graph canvas beside a details panel (stacked on phones). */
:root {
  --bg: #f6f7f8; --panel: #ffffff; --line: #d9dde2; --ink: #1b1f24; --ink-2: #4a525c; --ink-3: #7a838e;
  --edge: #9aa3ad; --focus: #1b1f24;
  --t1: #2a78d6; --t2: #eb6834; --t3: #1baf7a; --t4: #eda100; --t5: #e87ba4; --t6: #008300; --t7: #6250d6; --t8: #e34948;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141618; --panel: #1c1f22; --line: #33383e; --ink: #eef0f2; --ink-2: #b9c0c8; --ink-3: #8b939c;
  --edge: #5c646d; --focus: #eef0f2;
  --t1: #3987e5; --t2: #d95926; --t3: #199e70; --t4: #c98500; --t5: #d55181; --t6: #008300; --t7: #9085e9; --t8: #e66767;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141618; --panel: #1c1f22; --line: #33383e; --ink: #eef0f2; --ink-2: #b9c0c8; --ink-3: #8b939c;
  --edge: #5c646d; --focus: #eef0f2;
  --t1: #3987e5; --t2: #d95926; --t3: #199e70; --t4: #c98500; --t5: #d55181; --t6: #008300; --t7: #9085e9; --t8: #e66767;
  color-scheme: dark }
html, body { height: 100%; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 14px; display: flex; flex-direction: column; }
header { padding-block: 14px 10px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  border-bottom: 1px solid var(--line); display: flex; flex-direction: column; gap: 10px; background: var(--panel); }
.top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
h1 { font-size: 17px; font-weight: 600; margin: 0; text-wrap: balance; }
.count { font-family: var(--mono); font-size: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
.controls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip { display: inline-flex; align-items: center; gap: 6px; font: 500 12px var(--sans); color: var(--ink);
  background: transparent; border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; cursor: pointer; }
.chip[aria-pressed="false"] { color: var(--ink-3); border-style: dashed; }
.chip[aria-pressed="false"] svg { opacity: .3; }
.chip:focus-visible, input:focus-visible, button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.chip .n { font-family: var(--mono); color: var(--ink-3); font-variant-numeric: tabular-nums; }
input[type=search] { font: 13px var(--sans); color: var(--ink); background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 9px; min-width: 0; width: 220px; max-width: 100%; }
.toggle { font-size: 12px; color: var(--ink-2); display: inline-flex; gap: 5px; align-items: center; }
main { flex: 1; display: flex; min-height: 0; }
#graph { flex: 1; min-width: 0; position: relative; }
svg#canvas { width: 100%; height: 100%; display: block; cursor: grab; }
svg#canvas:active { cursor: grabbing; }
.edge { stroke: var(--edge); stroke-width: 1; fill: none; }
.edge.hot { stroke: var(--ink); stroke-width: 1.6; }
.edge-label { font: 9px var(--mono); fill: var(--ink-3); pointer-events: none; }
.node path { stroke: var(--panel); stroke-width: 1.5; cursor: pointer; }
.node text { font: 10px var(--sans); fill: var(--ink-2); pointer-events: none; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }
.node.tool text { font-weight: 600; fill: var(--ink); font-size: 11px; }
.dim { opacity: .12; }
.node.sel path { stroke: var(--focus); stroke-width: 2.5; }
aside { width: 340px; flex: none; border-left: 1px solid var(--line); background: var(--panel); overflow-y: auto;
  padding-block: 14px; padding-left: 16px; padding-right: max(16px, env(safe-area-inset-right, 0px)); }
aside h2 { font-size: 15px; margin: 2px 0 2px; font-family: var(--mono); font-weight: 500; overflow-wrap: anywhere; }
.kind { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); display: flex; align-items: center; gap: 6px; }
dl { margin: 10px 0 0; display: grid; grid-template-columns: max-content 1fr; gap: 6px 10px; font-size: 12.5px; }
dt { color: var(--ink-3); font-family: var(--mono); font-size: 11.5px; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; white-space: pre-wrap; line-height: 1.45; }
h3 { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); margin: 18px 0 6px; font-weight: 600; }
ul.links { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; }
ul.links button { all: unset; cursor: pointer; font-family: var(--mono); color: var(--ink); border-bottom: 1px dotted var(--ink-3); }
ul.links .p { color: var(--ink-3); font-family: var(--mono); font-size: 11px; }
ul.links .note { color: var(--ink-2); display: block; padding-left: 12px; font-size: 12px; }
.hint { color: var(--ink-2); line-height: 1.55; font-size: 13px; max-width: 34ch; }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  padding: 5px 8px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.12); max-width: 260px; }
#tip b { font-family: var(--mono); font-weight: 500; }
@media (max-width: 760px) {
  main { flex-direction: column; }
  #graph { min-height: 60vh; }
  aside { width: auto; border-left: 0; border-top: 1px solid var(--line); padding-left: max(16px, env(safe-area-inset-left, 0px)); max-height: 40vh; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<header>
  <div class="top"><h1>circuitAnt Tool Graph</h1><span class="count" id="count"></span></div>
  <div class="controls" id="chips"></div>
  <div class="controls">
    <input type="search" id="search" placeholder="Find a node (e.g. get_cone, ANY_PI)" aria-label="Find a node">
    <label class="toggle"><input type="checkbox" id="labels"> edge names</label>
    <button class="chip" id="reset" type="button">Reset view</button>
  </div>
</header>
<main>
  <div id="graph"><svg id="canvas" role="img" aria-label="Tool knowledge graph"></svg><div id="tip" hidden></div></div>
  <aside id="panel" aria-live="polite">
    <p class="hint">Click a node to see everything the graph stores about it and what it connects to.
    Drag to move nodes, scroll or pinch to zoom, and use the chips to hide node types.</p>
  </aside>
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA = __DATA__;
const TYPES = [  // fixed order: colour slot and shape per node type
  ["Tool", "--t1", d3.symbolCircle, 340], ["Category", "--t2", d3.symbolSquare, 190],
  ["Parameter", "--t3", d3.symbolCircle, 70], ["Value kind", "--t4", d3.symbolDiamond, 190],
  ["Keyword", "--t5", d3.symbolTriangle, 110], ["Concept", "--t6", d3.symbolStar, 170],
  ["Example", "--t7", d3.symbolCircle, 45], ["Design object", "--t8", d3.symbolWye, 190]];
const TYPE = Object.fromEntries(TYPES.map(([n, c, s, a]) => [n, {color: `var(${c})`, symbol: s, area: a}]));
const shown = new Set(TYPES.map(t => t[0]));
const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const links = DATA.edges.map(e => ({...e}));
const nbrs = new Map(DATA.nodes.map(n => [n.id, new Set()]));
links.forEach(e => { nbrs.get(e.source).add(e.target); nbrs.get(e.target).add(e.source); });

document.getElementById("count").textContent =
  `${DATA.nodes.length} nodes · ${DATA.edges.length} edges · built from agent/tools.py, ontology.ttl and knowledge.ttl`;

const svg = d3.select("#canvas"), root = svg.append("g");
svg.append("defs").append("marker").attr("id", "arrow").attr("viewBox", "0 -4 8 8").attr("refX", 14).attr("refY", 0)
  .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
  .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "var(--edge)");
const zoom = d3.zoom().scaleExtent([0.15, 4]).on("zoom", ev => root.attr("transform", ev.transform));
svg.call(zoom);

const edgeSel = root.append("g").selectAll("line").data(links).join("line").attr("class", "edge").attr("marker-end", "url(#arrow)");
const edgeLabel = root.append("g").selectAll("text").data(links).join("text").attr("class", "edge-label").text(d => d.label).attr("hidden", true);
const nodeSel = root.append("g").selectAll("g").data(DATA.nodes).join("g")
  .attr("class", d => "node" + (d.type === "Tool" ? " tool" : ""))
  .attr("tabindex", 0).attr("role", "button").attr("aria-label", d => `${d.type} ${d.label}`);
nodeSel.append("path").attr("d", d => d3.symbol(TYPE[d.type].symbol, TYPE[d.type].area)()).attr("fill", d => TYPE[d.type].color);
nodeSel.append("text").attr("x", 9).attr("y", 3.5).text(d => d.type === "Example" ? "" : d.label);

const sim = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(l => l.label === "hasExample" ? 26 : l.label === "hasParameter" ? 34 : 60).strength(0.5))
  .force("charge", d3.forceManyBody().strength(d => d.type === "Tool" ? -260 : -90))
  .force("collide", d3.forceCollide(d => d.type === "Tool" ? 16 : 8))
  .force("x", d3.forceX().strength(0.03)).force("y", d3.forceY().strength(0.03))
  .on("tick", () => {
    edgeSel.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    edgeLabel.attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2);
    nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
  });
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { sim.stop(); for (let i = 0; i < 300; i++) sim.tick(); sim.on("tick")(); }

nodeSel.call(d3.drag()
  .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
  .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
  .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

function fit() {
  const box = document.getElementById("graph").getBoundingClientRect();
  const xs = DATA.nodes.map(n => n.x), ys = DATA.nodes.map(n => n.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const k = Math.min(box.width / (x1 - x0 + 80), box.height / (y1 - y0 + 80), 1.5);
  svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(box.width / 2, box.height / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
}
sim.on("end", fit); setTimeout(fit, 1600);
document.getElementById("reset").onclick = () => { select(null); fit(); };

const tip = document.getElementById("tip");
nodeSel.on("mouseenter", (ev, d) => { highlight(d.id);
    tip.hidden = false; tip.innerHTML = `<b></b><div></div>`; tip.firstChild.textContent = d.label; tip.lastChild.textContent = d.type +
      (d.details.summary ? " · " + d.details.summary : ""); move(ev); })
  .on("mousemove", move).on("mouseleave", () => { tip.hidden = true; highlight(selected); })
  .on("click", (ev, d) => select(d.id)).on("keydown", (ev, d) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(d.id); } });
function move(ev) { const b = document.getElementById("graph").getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - b.left + 12, b.width - 270) + "px"; tip.style.top = (ev.clientY - b.top + 12) + "px"; }

let selected = null;
function highlight(id) {
  const near = id ? new Set([id, ...nbrs.get(id)]) : null;
  nodeSel.classed("dim", d => !shown.has(d.type) ? false : near ? !near.has(d.id) : false);
  edgeSel.classed("hot", d => near && (d.source.id === id || d.target.id === id))
         .classed("dim", d => near ? !(d.source.id === id || d.target.id === id) : false);
}
function visible() {
  nodeSel.attr("hidden", d => shown.has(d.type) ? null : true);
  const on = d => shown.has(d.source.type) && shown.has(d.target.type);
  edgeSel.attr("hidden", d => on(d) ? null : true);
  edgeLabel.attr("hidden", d => on(d) && document.getElementById("labels").checked ? null : true);
}
document.getElementById("labels").onchange = visible;

function select(id) {
  selected = id; nodeSel.classed("sel", d => d.id === id); highlight(id);
  const panel = document.getElementById("panel");
  if (!id) { panel.innerHTML = `<p class="hint">Click a node to see everything the graph stores about it and what it connects to. Drag to move nodes, scroll or pinch to zoom, and use the chips to hide node types.</p>`; return; }
  const n = byId.get(id), t = TYPE[n.type];
  panel.innerHTML = "";
  const kind = document.createElement("div"); kind.className = "kind";
  kind.innerHTML = `<svg width="12" height="12" viewBox="-6 -6 12 12"><path></path></svg><span></span>`;
  kind.querySelector("path").setAttribute("d", d3.symbol(t.symbol, 60)()); kind.querySelector("path").setAttribute("fill", t.color);
  kind.querySelector("span").textContent = n.type;
  const h = document.createElement("h2"); h.textContent = n.label;
  panel.append(kind, h);
  const dl = document.createElement("dl");
  Object.entries(n.details).forEach(([k, v]) => { const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = Array.isArray(v) ? v.join("\n") : v; dl.append(dt, dd); });
  const idt = document.createElement("dt"), idd = document.createElement("dd"); idt.textContent = "IRI"; idd.textContent = n.id; dl.append(idt, idd);
  panel.append(dl);
  for (const [title, list] of [["Points to", links.filter(l => l.source.id === id).map(l => [l, l.target])],
                               ["Pointed to by", links.filter(l => l.target.id === id).map(l => [l, l.source])]]) {
    if (!list.length) continue;
    const h3 = document.createElement("h3"); h3.textContent = `${title} (${list.length})`;
    const ul = document.createElement("ul"); ul.className = "links";
    list.sort((a, b) => a[0].label.localeCompare(b[0].label) || a[1].label.localeCompare(b[1].label)).forEach(([l, other]) => {
      const li = document.createElement("li"), p = document.createElement("span"), b = document.createElement("button");
      p.className = "p"; p.textContent = l.label + " → "; b.textContent = other.label; b.onclick = () => { focusOn(other.id); select(other.id); };
      li.append(p, b);
      if (l.note) { const note = document.createElement("span"); note.className = "note"; note.textContent = l.note; li.append(note); }
      ul.append(li); });
    panel.append(h3, ul);
  }
}
function focusOn(id) { const n = byId.get(id); if (!shown.has(n.type)) { shown.add(n.type); chips(); visible(); }
  const b = document.getElementById("graph").getBoundingClientRect(), k = d3.zoomTransform(svg.node()).k;
  svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(b.width / 2, b.height / 2).scale(Math.max(k, 1)).translate(-n.x, -n.y)); }

function chips() {
  const box = document.getElementById("chips"); box.innerHTML = "";
  TYPES.forEach(([name]) => { const t = TYPE[name], n = DATA.nodes.filter(d => d.type === name).length;
    if (!n) return;
    const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.setAttribute("aria-pressed", shown.has(name));
    b.innerHTML = `<svg width="12" height="12" viewBox="-6 -6 12 12"><path></path></svg><span></span><span class="n"></span>`;
    b.querySelector("path").setAttribute("d", d3.symbol(t.symbol, 60)()); b.querySelector("path").setAttribute("fill", t.color);
    b.querySelector("span").textContent = name; b.querySelector(".n").textContent = n;
    b.onclick = () => { shown.has(name) ? shown.delete(name) : shown.add(name); chips(); visible(); highlight(selected); };
    box.append(b); });
}
chips(); visible();

document.getElementById("search").addEventListener("keydown", ev => { if (ev.key !== "Enter") return;
  const q = ev.target.value.trim().toLowerCase(); if (!q) return;
  const hit = DATA.nodes.find(n => n.label.toLowerCase() === q) || DATA.nodes.find(n => n.label.toLowerCase().includes(q));
  if (hit) { focusOn(hit.id); select(hit.id); } });
select("ct:get_fanout");
</script>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "tool_graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = export()
    out.write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    kinds = {}
    for n in data["nodes"]:
        kinds[n["type"]] = kinds.get(n["type"], 0) + 1
    print(f"wrote {out}: {len(data['nodes'])} nodes ({kinds}), {len(data['edges'])} edges")


if __name__ == "__main__":
    main()
