import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Standardized logger for reproducible caching experiments.
    
    Generates:
      - config.yaml
      - metadata.json
      - request_logs.jsonl
      - candidate_logs.jsonl
      - cache_events.jsonl
      - metrics.json
      - summary.csv
    """

    def __init__(self, run_id: str, output_dir: str, config: Dict[str, Any]):
        self.run_id = run_id
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.req_log_file = open(self.output_dir / "request_logs.jsonl", "w", encoding="utf-8")
        self.cand_log_file = open(self.output_dir / "candidate_logs.jsonl", "w", encoding="utf-8")
        self.event_log_file = open(self.output_dir / "cache_events.jsonl", "w", encoding="utf-8")
        self.summary_rows: List[Dict[str, Any]] = []

        self._save_config_and_metadata()

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    def _save_config_and_metadata(self) -> None:
        import yaml
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        metadata = {
            "run_id": self.run_id,
            "experiment_id": self.config.get("experiment_id", "unknown"),
            "strategy": self.config.get("strategy", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._get_git_commit(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def log_cache_event(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self.event_log_file.write(json.dumps(record) + "\n")
        self.event_log_file.flush()

    def log_candidates(self, request_id: str, candidates: List[Dict[str, Any]]) -> None:
        for cand in candidates:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                **cand,
            }
            self.cand_log_file.write(json.dumps(record) + "\n")
        self.cand_log_file.flush()

    def log_request(self, record: Dict[str, Any]) -> None:
        self.req_log_file.write(json.dumps(record) + "\n")
        self.req_log_file.flush()

        summary_item = {
            "request_id": record.get("request_id"),
            "testcase": record.get("testcase"),
            "prompt": record.get("prompt", "")[:60],
            "decision": record.get("decision_type"),
            "accepted": record.get("accepted"),
            "reason": record.get("reason"),
            "agreement": record.get("agreement"),
            "lat_saved_s": round(record.get("latency_saved_sec", 0.0), 3),
            "tokens_saved": record.get("total_tokens_saved", 0),
            "cost_saved_usd": round(record.get("cost_saved_usd", 0.0), 5),
            "cache_size": record.get("cache_size_after", 0),
        }
        self.summary_rows.append(summary_item)

    def finish(self, metrics: Dict[str, Any]) -> None:
        self.req_log_file.close()
        self.cand_log_file.close()
        self.event_log_file.close()

        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if self.summary_rows:
            fieldnames = list(self.summary_rows[0].keys())
            with open(self.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.summary_rows)
