"""SQLite storage for generalized plan templates and embeddings."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .generalize import PlanTemplate


class PlanStore:
    """Manages SQLite storage for semantic plan templates."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masked_intent TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    slot_schema_json TEXT NOT NULL,
                    mutates_design INTEGER NOT NULL,
                    tool_sequence TEXT NOT NULL,
                    use_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_seq_schema 
                ON templates (tool_sequence, slot_schema_json);
                """
            )
            conn.commit()

    def save_or_update(self, template: PlanTemplate, embedding: list[float]) -> int:
        """Insert a new template or update stats if an equivalent one already exists."""
        schema_json = json.dumps(template.slot_schema, sort_keys=True)
        steps_json = json.dumps(template.steps)
        emb_json = json.dumps(embedding)

        with self._get_conn() as conn:
            # Check if a template with the same tool sequence and slot schema already exists
            row = conn.execute(
                """
                SELECT id, use_count FROM templates
                WHERE tool_sequence = ? AND slot_schema_json = ?
                LIMIT 1;
                """,
                (template.tool_sequence, schema_json),
            ).fetchone()

            if row:
                tid = row["id"]
                conn.execute(
                    """
                    UPDATE templates
                    SET use_count = use_count + 1
                    WHERE id = ?;
                    """,
                    (tid,),
                )
                conn.commit()
                return tid
            else:
                cur = conn.execute(
                    """
                    INSERT INTO templates (
                        masked_intent, embedding, steps_json, slot_schema_json,
                        mutates_design, tool_sequence, use_count, failure_count
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, 0);
                    """,
                    (
                        template.masked_intent,
                        emb_json,
                        steps_json,
                        schema_json,
                        1 if template.mutates_design else 0,
                        template.tool_sequence,
                    ),
                )
                conn.commit()
                return cur.lastrowid

    def get_all_templates(self) -> list[tuple[int, PlanTemplate, list[float]]]:
        """Fetch all stored templates and their embeddings."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, masked_intent, embedding, steps_json, slot_schema_json,
                       mutates_design, tool_sequence, use_count, failure_count
                FROM templates;
                """
            ).fetchall()

        templates: list[tuple[int, PlanTemplate, list[float]]] = []
        for r in rows:
            emb = json.loads(r["embedding"])
            tpl = PlanTemplate(
                masked_intent=r["masked_intent"],
                steps=json.loads(r["steps_json"]),
                slot_schema=json.loads(r["slot_schema_json"]),
                mutates_design=bool(r["mutates_design"]),
                tool_sequence=r["tool_sequence"],
                use_count=r["use_count"],
                failure_count=r["failure_count"],
            )
            templates.append((r["id"], tpl, emb))
        return templates

    def record_success(self, template_id: int) -> None:
        """Increment use count on successful execution."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE templates SET use_count = use_count + 1 WHERE id = ?;",
                (template_id,),
            )
            conn.commit()

    def record_failure(self, template_id: int, max_failures_before_drop: int = 3) -> bool:
        """Increment failure count. If it reaches max_failures_before_drop, drop template.

        Returns True if dropped, False otherwise.
        """
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE templates SET failure_count = failure_count + 1 WHERE id = ?;",
                (template_id,),
            )
            row = conn.execute(
                "SELECT failure_count FROM templates WHERE id = ?;",
                (template_id,),
            ).fetchone()
            if row and row["failure_count"] >= max_failures_before_drop:
                conn.execute("DELETE FROM templates WHERE id = ?;", (template_id,))
                conn.commit()
                return True
            conn.commit()
            return False

    def clear(self) -> None:
        """Delete all templates."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM templates;")
            conn.commit()
