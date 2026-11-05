"""Dynamic Tool Synthesizer for circuitAnt.

Implements the KTCE (AAAI-25) Problem -> Knowledge -> Tool paradigm for EDA.
Synthesizes generalized, reusable Tools methods grounded in the Tool Knowledge Graph
and OpenROAD/OpenDB engine primitives, then hot-loads them into the runtime session.
"""
import ast
import inspect
import json
from pathlib import Path
import re
import types
from typing import Any, Dict, Optional, Tuple

from agent.llm import OpenAIClient
from agent.synthesis.kg_context import KGContextExtractor
from agent.tools import (
    EndPoint,
    FilePath,
    Gate,
    GateType,
    GateTypes,
    Param,
    Signal,
    StartPoint,
    Tools,
    changes_design,
    schema,
    tool,
)
from eda_engine import Design, analysis, blif, edit, transformation, utils, verification

SYNTHESIS_SYSTEM_PROMPT = """You are an expert EDA software engineer extending circuitAnt, a framework for netlist analysis, transformation, and optimization on OpenROAD's OpenDB.
Your task is to generate a single, highly reusable, production-ready Python method to be added to the `Tools` class.

Follow these strict requirements:
1. Method definition: `def <function_name>(self, ...) -> dict:`
2. Type annotations: Use `Signal`, `Gate`, `Annotated[str, Param(...)]`, `int`, etc.
   Note: `Param`, `tool`, `Signal`, `Gate`, `StartPoint`, `EndPoint` are from `agent.tools` (NOT from `typing`).
   `Annotated` is from `typing`.
   All of these, plus `self._d()`, `edit`, `utils`, `transformation`, `analysis`, and `verification`, are already imported and in scope!
3. Decorator: Always decorate with `@tool(examples=(...))` giving representative example requests.
4. Docstring: Write a clean docstring whose first sentence summarizes what the tool does and returns.
5. In-place & Functional: Return a JSON-serializable dictionary with informative fields (e.g., counts before/after, status).
6. REUSABILITY CONTRACT: The method MUST operate on any loaded netlist. NEVER hardcode net names, instance names, or numbers from a specific testcase.
7. Standalone Definition: Define the method as a top-level function starting with @tool and def <name>(self, ...). DO NOT wrap it in `class Tools:`.
8. LEVERAGE ENGINE LIBRARY: Always check the provided engine API context bundle. If an established implementation in `transformation.<func>(self._d(), ...)` or `analysis.<func>(self._d(), ...)` exists (such as `transformation.insert_dedicated_buffers_for_loads`, `transformation.collapse_inverter_pairs`, or `analysis.derive_boolean_equation`), call it directly instead of reinventing low-level pointer logic or guessing OpenDB methods.
9. THE "BIG 4" OPENDB INVARIANTS (STRICT):
   a. Loads: `loads = [it for it in net.getITerms() if it.isInputSignal()]`. (NEVER call net.getLoads()!).
   b. Drivers: `drivers = [it for it in net.getITerms() if it.isOutputSignal()]`. (NEVER call net.getDriver()!).
   c. Add Gate: `inst = edit.add_gate(d, master_name, {pin_name: dbNet_obj}, base_name)`.
   d. Rewire: `iterm.disconnect()`; `iterm.connect(new_net_obj)`. (NEVER call iterm.setNet()!).
10. SELF-VERIFYING TRANSFORMATIONS: For design transformations, return informative dictionaries with before/after counts, and feel free to include equivalence verification status via `verification.check_equivalence(self._d())` or `self.check_equivalence()`.
11. Return ONLY valid executable Python code enclosed in a ```python ... ``` block. No conversational preamble.
"""


class DynamicToolSynthesizer:
    """Synthesizes, validates, and hot-loads missing tools at runtime."""

    def __init__(self, api_key: str, model: str = "gpt-4o", tools_instance: Optional[Tools] = None):
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.model = model
        self.tools = tools_instance
        self.kg = KGContextExtractor(tools_instance)
        self.synthesized_tools: list[str] = []
        self.synthesis_attempts: dict[str, int] = {}
        self.last_failed_tool: str = ""
        self.last_error: str = ""

    def synthesize_tool(
        self,
        category: str,
        intent: str,
        suggested_name: str,
        expected_inputs: str = "",
        expected_output_fields: str = "",
        mutation_feedback: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Prompts the LLM to synthesize the function code.

        Returns (function_name, python_code_str).
        """
        # Sanitize specific signal/gate references from suggested_name into generic names
        clean_name = re.sub(r'_(n|g)\d+(\b|_)', r'_signal\2', suggested_name)
        if clean_name != suggested_name:
            suggested_name = clean_name

        bundle = self.kg.build_context_bundle(category, intent, suggested_name)

        user_content = f"""Please synthesize the missing EDA tool method:
Target Function Name: {suggested_name}
Category: {category}
Formal Operational Intent: {intent}
Expected Input Parameters: {expected_inputs or 'Infer typed parameters (e.g. signal_name: Signal)'}
Expected Output Fields: {expected_output_fields or 'Dictionary with status/counts'}

### GENERALIZATION CONTRACT (CRITICAL):
1. The function MUST operate on ANY circuit netlist. NEVER hardcode net names (like 'n0') or gate instances.
2. If the user request mentions a specific net or gate, you MUST parameterize it as an input argument (e.g. `signal_name: Signal`).
3. Always include `@tool(examples=(...))` showing representative, generalized usage requests.

{bundle}
"""
        if mutation_feedback:
            user_content += f"""
### MUTATION REPAIR REQUIRED:
The previous implementation failed with the following verification or runtime error:
{mutation_feedback}
Please fix the implementation to address this failure while preserving general EDA invariants.
"""

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        reply = self.client.complete(messages, tools=[])
        code = self._extract_code(reply.content)
        return suggested_name, code

    def validate_code(self, code: str) -> None:
        """Validates Python syntax, security constraints, and generalization invariants."""
        tree = ast.parse(code)
        # Security sanity check: forbid eval, exec, subprocess, os.system
        forbidden_calls = {"eval", "exec", "system", "popen", "spawn"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                    raise ValueError(f"Security error: forbidden call '{node.func.id}' in synthesized tool.")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                    raise ValueError(f"Security error: forbidden attribute call '{node.func.attr}'.")

        # Generalization contract check: forbid hardcoded testcase literals (e.g. "n0", "g12") inside function body
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start_idx = 1 if func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant) else 0
            for stmt in func.body[start_idx:]:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if re.match(r'^(n|g)\d+(\[\d+\])?$', node.value):
                            raise ValueError(
                                f"Generalization Violation: Hardcoded netlist literal '{node.value}' found in synthesized tool logic. "
                                f"Tools must be generalized to accept parameters (e.g. signal_name: Signal, gate_name: Gate) "
                                f"rather than hardcoding instance/net names."
                            )

    def hot_load(self, code: str, suggested_name: Optional[str] = None) -> Any:
        """Executes code in the circuitAnt runtime namespace and binds it to `self.tools`."""
        self.validate_code(code)

        exec_globals = {
            "Tools": Tools,
            "tool": tool,
            "Param": Param,
            "Signal": Signal,
            "Gate": Gate,
            "StartPoint": StartPoint,
            "EndPoint": EndPoint,
            "GateType": GateType,
            "GateTypes": GateTypes,
            "FilePath": FilePath,
            "Design": Design,
            "analysis": analysis,
            "transformation": transformation,
            "verification": verification,
            "edit": edit,
            "utils": utils,
            "blif": blif,
            "re": __import__("re"),
            "Path": __import__("pathlib").Path,
            "typing": __import__("typing"),
            "Annotated": __import__("typing").Annotated,
        }
        exec_locals: Dict[str, Any] = {}
        exec(code, exec_globals, exec_locals)

        # Find the defined function
        fn = None
        if suggested_name and suggested_name in exec_locals and inspect.isfunction(exec_locals[suggested_name]):
            fn = exec_locals[suggested_name]

        # Check if model wrapped inside a class (e.g. class Tools)
        if not fn and "Tools" in exec_locals and inspect.isclass(exec_locals["Tools"]):
            cls = exec_locals["Tools"]
            for m_name, m_val in inspect.getmembers(cls, inspect.isfunction):
                if not m_name.startswith("_") and m_name not in ("tool", "schema"):
                    fn = m_val
                    break

        if not fn:
            for k, val in exec_locals.items():
                if k not in ("tool", "changes_design", "schema", "parameter") and inspect.isfunction(val):
                    fn = val
                    break

        if not fn:
            raise ValueError("No function definition found in synthesized code.")

        fn_name = fn.__name__
        # Bind method dynamically to Tools class and instance
        setattr(Tools, fn_name, fn)
        if self.tools:
            bound_method = types.MethodType(fn, self.tools)
            setattr(self.tools, fn_name, bound_method)

        return fn

    def synthesize_and_register(
        self,
        category: str,
        intent: str,
        suggested_name: str,
        expected_inputs: str = "",
        expected_output_fields: str = "",
        access_instance: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """End-to-end: synthesize, validate, hot-load, and update access registry."""
        try:
            # Clean specific testcase signal/gate literals from name
            clean_name = re.sub(r'_(n|g)\d+(\b|_)', r'_signal\2', suggested_name)
            if clean_name != suggested_name:
                suggested_name = clean_name

            # 1. Deduplication Pre-Check: if already bound to self.tools
            if hasattr(self.tools, clean_name):
                bound_fn = getattr(self.tools, clean_name)
                if access_instance:
                    self._update_access(access_instance, bound_fn)
                return {
                    "status": "SUCCESS",
                    "tool_name": clean_name,
                    "message": f"Tool '{clean_name}' already exists and is ready for use. Call it directly with your arguments.",
                }

            # 2. Disk Persistence Pre-Check: if already exists on disk
            tool_disk_path = Path("agent/dynamic_tools") / f"{clean_name}.py"
            if tool_disk_path.is_file():
                try:
                    code = tool_disk_path.read_text(encoding="utf-8")
                    fn = self.hot_load(code, suggested_name=clean_name)
                    if access_instance:
                        self._update_access(access_instance, fn)
                    if clean_name not in self.synthesized_tools:
                        self.synthesized_tools.append(clean_name)
                    return {
                        "status": "SUCCESS",
                        "tool_name": clean_name,
                        "saved_path": str(tool_disk_path),
                        "code": code,
                        "message": f"Tool '{clean_name}' loaded from persistent cache and hot-loaded into active session.",
                    }
                except Exception:
                    pass  # If disk file was corrupted, proceed to synthesize

            # 3. Strict 2-Attempt Synthesis Budget
            base_key = re.sub(r'_v\d+$', '', clean_name)
            attempts = self.synthesis_attempts.get(base_key, 0)
            if attempts >= 2:
                return {
                    "status": "ERROR",
                    "tool_name": suggested_name,
                    "error": f"Synthesis budget exceeded for '{base_key}' (maximum 2 attempts reached).",
                }
            self.synthesis_attempts[base_key] = attempts + 1

            mutation_fb = self.last_error or None
            self.last_error = ""  # Consume error
            fn_name, code = self.synthesize_tool(
                category,
                intent,
                suggested_name,
                expected_inputs,
                expected_output_fields,
                mutation_feedback=mutation_fb,
            )
            fn = self.hot_load(code, suggested_name=suggested_name)

            # 1. Persist synthesized function to disk repository (agent/dynamic_tools/<name>.py)
            saved_path = self._persist_to_disk(fn.__name__, code)

            # 2. Persist metadata and concept mapping to Tool Knowledge Graph (.ttl)
            self._persist_to_kg(fn, category, intent, access_instance)

            # 3. Update Access mode schemas if an access instance is provided
            if access_instance:
                self._update_access(access_instance, fn)

            self.synthesized_tools.append(fn.__name__)

            return {
                "status": "SUCCESS",
                "tool_name": fn.__name__,
                "saved_path": str(saved_path),
                "code": code,
                "message": f"Tool '{fn.__name__}' synthesized, saved to {saved_path}, and hot-loaded into active session.",
            }
        except Exception as exc:
            return {"status": "ERROR", "tool_name": suggested_name, "error": str(exc)}

    def mutate_and_reregister(
        self,
        category: str,
        intent: str,
        suggested_name: str,
        error_feedback: str,
        expected_inputs: str = "",
        expected_output_fields: str = "",
        access_instance: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Performs a KTCE Mutation pass using error feedback."""
        try:
            fn_name, code = self.synthesize_tool(
                category,
                intent,
                suggested_name,
                expected_inputs,
                expected_output_fields,
                mutation_feedback=error_feedback,
            )
            fn = self.hot_load(code, suggested_name=suggested_name)
            saved_path = self._persist_to_disk(fn.__name__, code)
            self._persist_to_kg(fn, category, intent, access_instance)
            if access_instance:
                self._update_access(access_instance, fn)
            return {
                "status": "SUCCESS",
                "tool_name": fn.__name__,
                "saved_path": str(saved_path),
                "code": code,
                "message": f"Tool '{fn.__name__}' mutated, saved to {saved_path}, and hot-loaded after repair.",
            }
        except Exception as exc:
            return {"status": "ERROR", "tool_name": suggested_name, "error": str(exc)}

    def _persist_to_disk(self, fn_name: str, code: str) -> Path:
        """Saves synthesized tool Python file to agent/dynamic_tools/."""
        tools_dir = Path("agent/dynamic_tools")
        tools_dir.mkdir(parents=True, exist_ok=True)
        out_file = tools_dir / f"{fn_name}.py"
        header = f'''"""Dynamically synthesized tool: {fn_name}.
Generated by CircuitAnt Knowledge-Grounded Dynamic Tool Synthesizer.
"""
from typing import Annotated
from agent.tools import EndPoint, FilePath, Gate, GateType, GateTypes, Param, Signal, StartPoint, Tools, changes_design, tool
from eda_engine import Design, analysis, blif, edit, transformation, utils, verification

'''
        clean_code = code
        if "from eda_engine import" not in clean_code:
            clean_code = "from eda_engine import Design, analysis, blif, edit, transformation, utils, verification\n" + clean_code
        if "from agent.tools import" not in clean_code:
            clean_code = "from typing import Annotated\nfrom agent.tools import EndPoint, FilePath, Gate, GateType, GateTypes, Param, Signal, StartPoint, Tools, changes_design, tool\n" + clean_code
        out_file.write_text(clean_code, encoding="utf-8")
        return out_file

    def _persist_to_kg(self, fn: Any, category: str, intent: str, access: Optional[Any] = None) -> None:
        """Persists the tool metadata and concept mapping to dynamic_knowledge.ttl and active Graph."""
        fn_name = fn.__name__
        doc = inspect.getdoc(fn) or intent
        first_line = (doc.splitlines() or [intent])[0].strip()
        summary = first_line.replace('"', '\\"')
        doc_escaped = doc.replace('"', '\\"').replace("\n", " ")

        ttl_path = Path("agent/tool_access/graph/dynamic_knowledge.ttl")
        intent_escaped = intent.replace('"', r'\"')
        triple_block = f"""
ct:{fn_name} a ca:Tool ;
    rdfs:label "{fn_name}" ;
    ca:summary "{summary}" ;
    ca:description "{doc_escaped}" ;
    ca:inCategory ca:{category.capitalize()} .

ct:concept_{fn_name} a ca:Concept ;
    rdfs:label "{intent_escaped}" ;
    ca:answeredBy ct:{fn_name} .
"""
        try:
            with open(ttl_path, "a", encoding="utf-8") as f:
                f.write(triple_block)
        except Exception as exc:
            print(f"[WARNING] Failed to write dynamic_knowledge.ttl: {exc}")

        # Update live rdflib Graph if access has graph
        if access and hasattr(access, "graph"):
            try:
                from rdflib import Namespace, Literal, RDF, RDFS
                CA = Namespace("https://w3id.org/circuitant/ontology#")
                CT = Namespace("https://w3id.org/circuitant/tools#")
                access.graph.add((CT[fn_name], RDF.type, CA.Tool))
                access.graph.add((CT[fn_name], RDFS.label, Literal(fn_name)))
                access.graph.add((CT[fn_name], CA.summary, Literal(summary)))
                access.graph.add((CT[fn_name], CA.description, Literal(doc)))
                access.graph.add((CT[fn_name], CA.inCategory, CA[category.capitalize()]))
            except Exception as exc:
                print(f"[WARNING] Failed to update live rdflib graph: {exc}")

    def _update_access(self, access: Any, fn: Any) -> None:
        """Reflects the newly synthesized tool in DirectAccess or GraphAccess."""
        fn_name = fn.__name__
        bound_fn = getattr(self.tools, fn_name, fn) if self.tools else fn
        fn_schema = schema(bound_fn)

        # If DirectAccess
        if hasattr(access, "functions") and hasattr(access, "schemas"):
            access.functions[fn_name] = bound_fn
            access.schemas = [s for s in access.schemas if s["function"]["name"] != fn_name]
            access.schemas.append(fn_schema)

        # If GraphAccess
        if hasattr(access, "direct"):
            access.direct.functions[fn_name] = bound_fn
            access.direct.schemas = [s for s in access.direct.schemas if s["function"]["name"] != fn_name]
            access.direct.schemas.append(fn_schema)
            # In GraphAccess, mark described so model can immediately run it via run_tool
            if hasattr(access, "described"):
                access.described.add(fn_name)
            if hasattr(access, "synthesized_names"):
                access.synthesized_names.add(fn_name)
            if hasattr(access, "_format_system_text"):
                access.system_text = access._format_system_text()

    def _extract_code(self, response_text: str) -> str:
        """Extracts and sanitizes python code from markdown block."""
        match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
            code = match.group(1).strip() if match else response_text.strip()
        # Sanitize common import errors: Param is from agent.tools, not typing or eda_engine
        code = re.sub(r'from\s+(?:typing|eda_engine)\s+import\s+([^;\n]*\bParam\b[^;\n]*)', r'from agent.tools import Param', code)
        code = re.sub(r'from\s+eda_engine\.utils\s+import\s+verification', r'from eda_engine import verification', code)
        code = re.sub(r'\bkeep_net\s*=', 'keep=', code)
        code = re.sub(r'\bdrop_net\s*=', 'drop=', code)
        return code
