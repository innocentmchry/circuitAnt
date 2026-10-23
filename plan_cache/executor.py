"""Plan execution with reference resolution, verification, and transactional snapshot fallback."""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from eda_engine import read_design, write_design
from .generalize import PlanTemplate
from .trace import ToolCallRecord


def resolve_reference_path(ref_expr: str, earlier_results: list[Any]) -> Any:
    """Resolve a reference string like '$step_0.result.at_max[0]' against earlier results."""
    m = re.match(r"^\$step_(\d+)\.result(?:\.(.*)|(\[.*))?$", ref_expr)
    if not m:
        raise ValueError(f"Malformed reference expression: {ref_expr}")

    step_idx = int(m.group(1))
    if step_idx >= len(earlier_results):
        raise IndexError(f"Reference to step {step_idx} exceeds available results ({len(earlier_results)})")

    current_val = earlier_results[step_idx]
    path_suffix = (m.group(2) or "") + (m.group(3) or "")
    if not path_suffix:
        return current_val

    # Parse tokens: keys (.foo) and array indices ([0])
    tokens = re.findall(r"([^.\[\]]+)|\[(\d+)\]", path_suffix)
    for key_token, idx_token in tokens:
        if key_token:
            if not isinstance(current_val, dict) or key_token not in current_val:
                raise KeyError(f"Key {key_token!r} not found in step {step_idx} result")
            current_val = current_val[key_token]
        elif idx_token:
            idx = int(idx_token)
            if not isinstance(current_val, (list, tuple)) or idx >= len(current_val):
                raise IndexError(f"Index {idx} out of range in step {step_idx} result")
            current_val = current_val[idx]

    return current_val


class PlanExecutor:
    """Executes a generalized plan template on a loaded OpenDB design."""

    @staticmethod
    def execute(
        template: PlanTemplate,
        filled_slots: dict[str, Any],
        tools: Any,
    ) -> tuple[bool, list[ToolCallRecord], str | None]:
        """Execute template steps. Returns (success, tool_call_records, error_or_none).

        If design-mutating, takes a snapshot before execution and restores on any error
        or equivalence verification failure.
        """
        snapshot_file: Path | None = None
        orig_design = tools.design

        if template.mutates_design and orig_design is not None:
            # Create a temporary snapshot to restore in case of failure
            fd, tmp_path = tempfile.mkstemp(prefix="plan_snap_", suffix=".v")
            import os
            os.close(fd)
            snapshot_file = Path(tmp_path)
            try:
                write_design(orig_design, snapshot_file)
            except Exception as exc:
                if snapshot_file.exists():
                    snapshot_file.unlink()
                return False, [], f"Failed to create design snapshot: {exc}"

        earlier_results: list[Any] = []
        recorded_calls: list[ToolCallRecord] = []

        try:
            for step in template.steps:
                tool_name = step["tool"]
                raw_args = step["args"]

                # 1. Resolve arguments
                resolved_args: dict[str, Any] = {}
                for arg_name, arg_val in raw_args.items():
                    if isinstance(arg_val, str):
                        if arg_val.startswith("<") and arg_val.endswith(">"):
                            resolved_args[arg_name] = filled_slots[arg_val]
                        elif arg_val.startswith("$step_"):
                            resolved_args[arg_name] = resolve_reference_path(arg_val, earlier_results)
                        else:
                            resolved_args[arg_name] = arg_val
                    else:
                        resolved_args[arg_name] = arg_val

                # 2. Invoke tool
                tool_fn = getattr(tools, tool_name, None)
                if tool_fn is None:
                    raise AttributeError(f"Tool {tool_name!r} not found on Tools object")

                t_start = time.monotonic()
                result = tool_fn(**resolved_args)
                t_elapsed = time.monotonic() - t_start

                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(f"Tool {tool_name} returned error: {result['error']}")

                earlier_results.append(result)
                recorded_calls.append(
                    ToolCallRecord(
                        name=tool_name,
                        arguments=resolved_args,
                        result=result,
                        elapsed=t_elapsed,
                    )
                )

            # 3. If mutating plan, run equivalence check
            if template.mutates_design:
                equiv_fn = getattr(tools, "check_equivalence", None)
                if equiv_fn:
                    equiv_res = equiv_fn()
                    if equiv_res.get("status") != "PASS":
                        raise RuntimeError(f"Equivalence check failed: status={equiv_res.get('status')}")

            # Succeeded: clean up snapshot
            if snapshot_file and snapshot_file.exists():
                snapshot_file.unlink()

            return True, recorded_calls, None

        except Exception as exc:
            # Fallback: restore snapshot if mutated
            if snapshot_file and snapshot_file.exists():
                try:
                    tools.design = read_design(snapshot_file, **tools.design_config)
                except Exception:
                    pass
                try:
                    snapshot_file.unlink()
                except Exception:
                    pass
            return False, recorded_calls, str(exc)
