#!/usr/bin/env python3
"""MLflow Forensic Cross-Reference Analysis.

Pulls all runs from the MLflow API, cross-references with git history and
per-commit source code, and produces a Markdown report that distinguishes
config changes from code-semantic drift to explain performance regressions.

Three-layer param trust model:
  Layer 1 (Logged)   -- raw MLflow params
  Layer 2 (Schema)   -- committed YAML config at each git commit
  Layer 3 (Semantic) -- .get() defaults in trainer/model code at each commit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEAK_COMMIT = "94940c5"
PEAK_COMMIT_FULL = "94940c52461f0a13c6e0879dcf9953b3335ebccc"

KNOWN_RUNS = {
    "94940c5": "cont-diff-best-one-ever-16-heads (THE PEAK)",
    "d6dddeb": "src2 refactor era",
    "dac04c2": "granular emb dim / regression era",
}

# Files to scan for .get() defaults at each commit.
# Keyed by era: pre-refactor (src/) vs post-refactor (src2/).
CODE_SCAN_PATHS_PRE_REFACTOR = [
    "src/hashi_puzzle_solver/diffusion_engine.py",
    "src/hashi_puzzle_solver/engine.py",
    "src/hashi_puzzle_solver/diffusion_utils.py",
]
CODE_SCAN_PATHS_POST_REFACTOR = [
    "src2/hashi_puzzle_solver/trainers/diffusion.py",
    "src2/hashi_puzzle_solver/trainers/base.py",
    "src2/hashi_puzzle_solver/trainers/one_shot.py",
    "src2/hashi_puzzle_solver/models/core.py",
    "src2/hashi_puzzle_solver/models/factory.py",
    "src2/hashi_puzzle_solver/models/encoders.py",
    "src2/hashi_puzzle_solver/models/heads.py",
    "src2/hashi_puzzle_solver/models/backbone.py",
]

CONFIG_PATH = "configs/diffusion_solver_continuous.yaml"

# Regex to locate the start of .get("key", ...) patterns.
# Balanced-paren extraction handles nested calls like .get("k", foo.get("x", 1)).
_GET_START = re.compile(r"""\.get\(\s*["']([^"']+)["']\s*,\s*""")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_log_all() -> list[dict[str, str]]:
    """Return chronological list of {hash, date, subject} for all branches."""
    result = subprocess.run(
        ["git", "log", "--all", "--format=%H %ai %s"],
        capture_output=True, text=True, check=True,
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(" ", 3)
        if len(parts) >= 3:
            commits.append({
                "hash": parts[0],
                "date": f"{parts[1]} {parts[2]}",
                "subject": parts[3] if len(parts) > 3 else "",
            })
    commits.reverse()
    return commits


def git_show_file(commit: str, path: str) -> str | None:
    """Return file contents at a specific commit, or None if absent."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def git_diff_stat(commit_a: str, commit_b: str, path_filter: str = "src2/") -> str:
    """Return diff --stat between two commits, filtered to a path."""
    result = subprocess.run(
        ["git", "diff", "--stat", f"{commit_a}..{commit_b}", "--", path_filter],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Try without path filter (pre-refactor era)
        result = subprocess.run(
            ["git", "diff", "--stat", f"{commit_a}..{commit_b}", "--", "src/"],
            capture_output=True, text=True,
        )
    return result.stdout.strip()


def git_diff_config(commit_a: str, commit_b: str) -> str:
    """Return full config YAML diff between two commits."""
    result = subprocess.run(
        ["git", "diff", f"{commit_a}..{commit_b}", "--", CONFIG_PATH],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# MLflow API helpers
# ---------------------------------------------------------------------------

def fetch_all_runs(mlflow_url: str, experiment_id: str) -> list[dict]:
    """Paginated fetch of all runs from MLflow REST API."""
    runs: list[dict] = []
    page_token: str | None = None
    while True:
        payload: dict[str, Any] = {
            "experiment_ids": [experiment_id],
            "max_results": 200,
        }
        if page_token:
            payload["page_token"] = page_token

        resp = requests.post(
            f"{mlflow_url}/api/2.0/mlflow/runs/search",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        runs.extend(data.get("runs", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return runs


def parse_run(raw: dict) -> dict[str, Any]:
    """Flatten an MLflow run dict into a convenient structure."""
    info = raw.get("info", {})
    data = raw.get("data", {})

    params = {}
    for p in data.get("params", []):
        params[p["key"]] = p["value"]

    metrics = {}
    for m in data.get("metrics", []):
        metrics[m["key"]] = m["value"]

    tags = {}
    for t in data.get("tags", []):
        tags[t["key"]] = t["value"]

    start_ms = info.get("start_time", 0)
    end_ms = info.get("end_time", 0)
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc) if start_ms else None

    return {
        "run_id": info.get("run_id", ""),
        "run_name": info.get("run_name", ""),
        "status": info.get("status", ""),
        "start_time": start_dt,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "git_commit": tags.get("mlflow.source.git.commit", ""),
        "source_name": tags.get("mlflow.source.name", ""),
        "params": params,
        "metrics": metrics,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Per-commit code analysis
# ---------------------------------------------------------------------------

def extract_get_defaults(source_code: str) -> dict[str, str]:
    """Extract all .get("key", default) patterns, handling nested parens."""
    defaults: dict[str, str] = {}
    for match in _GET_START.finditer(source_code):
        key = match.group(1).strip()
        start = match.end()
        depth = 1
        i = start
        while i < len(source_code) and depth > 0:
            ch = source_code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            value = source_code[start : i - 1].strip().rstrip(",")
            # Skip values that are themselves complex expressions with nested calls
            if ".get(" not in value and len(value) < 60:
                defaults[key] = value
    return defaults


def get_code_defaults_for_commit(commit: str) -> dict[str, str]:
    """Build a {param: default_value} map from trainer/model code at a commit."""
    all_defaults: dict[str, str] = {}

    # Try post-refactor paths first, then pre-refactor
    for path in CODE_SCAN_PATHS_POST_REFACTOR:
        content = git_show_file(commit, path)
        if content:
            all_defaults.update(extract_get_defaults(content))

    for path in CODE_SCAN_PATHS_PRE_REFACTOR:
        content = git_show_file(commit, path)
        if content:
            all_defaults.update(extract_get_defaults(content))

    return all_defaults


def get_config_at_commit(commit: str) -> dict[str, Any] | None:
    """Parse the YAML config at a specific commit."""
    content = git_show_file(commit, CONFIG_PATH)
    if content is None:
        return None
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError:
        return None


def flatten_config(cfg: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested config dict using dot notation."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, key))
        else:
            flat[key] = str(v) if v is not None else ""
    return flat


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def compute_trust_tier(
    commit_date_str: str, run_start: datetime | None
) -> tuple[float, str]:
    """Compute hours between commit and run start, assign trust tier."""
    if not run_start:
        return -1.0, "UNKNOWN"
    try:
        commit_dt = datetime.fromisoformat(commit_date_str.replace(" ", "T"))
        if commit_dt.tzinfo is None:
            commit_dt = commit_dt.replace(tzinfo=timezone.utc)
        delta = abs((run_start - commit_dt).total_seconds()) / 3600
    except (ValueError, TypeError):
        return -1.0, "UNKNOWN"
    if delta < 4:
        return delta, "HIGH"
    if delta < 24:
        return delta, "MEDIUM"
    return delta, "LOW"


def classify_transition(
    config_diff: str, code_stat: str
) -> tuple[str, str]:
    """Classify a commit transition as CONFIG_CHANGE, SEMANTIC_DRIFT, or ENTANGLED."""
    has_config = bool(config_diff.strip())
    has_code = bool(code_stat.strip())

    if has_config and has_code:
        return "ENTANGLED", "Both config and code changed -- cannot attribute from params alone"
    if has_config:
        return "CONFIG_CHANGE", "Same code, different params"
    if has_code:
        return "SEMANTIC_DRIFT", "Same params, different behavior -- code changed but config identical"
    return "IDENTICAL", "No config or code changes"


def diff_defaults(
    defaults_a: dict[str, str], defaults_b: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Diff two default maps. Returns {key: {old, new, change_type}}."""
    all_keys = sorted(set(defaults_a) | set(defaults_b))
    diffs: dict[str, dict[str, str]] = {}
    for k in all_keys:
        old = defaults_a.get(k)
        new = defaults_b.get(k)
        if old is None and new is not None:
            diffs[k] = {"old": "-", "new": new, "change": "ADDED"}
        elif old is not None and new is None:
            diffs[k] = {"old": old, "new": "-", "change": "REMOVED"}
        elif old != new:
            diffs[k] = {"old": str(old), "new": str(new), "change": "CHANGED"}
    return diffs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class ForensicReport:
    """Builds the Markdown report from collected data."""

    def __init__(self) -> None:
        self.sections: list[str] = []

    def add(self, text: str) -> None:
        self.sections.append(text)

    def render(self) -> str:
        return "\n".join(self.sections)

    # -- Section builders --

    def write_header(self, num_runs: int, num_commits: int) -> None:
        self.add("# MLflow Forensic Cross-Reference Report")
        self.add(f"\n*Generated {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
        self.add(f"- **Runs analyzed:** {num_runs}")
        self.add(f"- **Distinct commits:** {num_commits}")
        self.add(f"- **Peak commit:** `{PEAK_COMMIT}` (94940c5)")
        self.add("- **Generated by:** [`scripts/mlflow_forensic_analysis.py`](../scripts/mlflow_forensic_analysis.py)")
        self.add("- **Plan:** [`.cursor/plans/mlflow_forensic_cross-reference_622aa1b6.plan.md`](../.cursor/plans/mlflow_forensic_cross-reference_622aa1b6.plan.md)")
        self.add("")

    def write_config_schema_table(
        self, commit_hashes: list[str], config_maps: dict[str, dict[str, str]]
    ) -> None:
        """Section 1: Config Schema Evolution Table."""
        self.add("---\n## Section 1: Config Schema Evolution\n")
        self.add("Shows which config keys exist in the committed YAML at each commit.")
        self.add("Cells show the value, or `-` if the key was absent.\n")

        all_keys = sorted({k for m in config_maps.values() for k in m})
        if not all_keys or not commit_hashes:
            self.add("*No config data available.*\n")
            return

        short = [h[:7] for h in commit_hashes]
        header = "| Param | " + " | ".join(f"`{s}`" for s in short) + " |"
        sep = "|---|" + "|".join("---" for _ in short) + "|"
        self.add(header)
        self.add(sep)

        for key in all_keys:
            cells = []
            for h in commit_hashes:
                val = config_maps.get(h, {}).get(key, "-")
                if len(val) > 20:
                    val = val[:17] + "..."
                cells.append(val)
            self.add(f"| `{key}` | " + " | ".join(cells) + " |")
        self.add("")

    def write_code_defaults_table(
        self,
        commit_hashes: list[str],
        defaults_maps: dict[str, dict[str, str]],
    ) -> None:
        """Section 2: Code Default Drift Table."""
        self.add("---\n## Section 2: Code Default Drift (.get() Defaults)\n")
        self.add("Shows `.get()` default values extracted from trainer/model source code at each commit.")
        self.add("**Bold** cells indicate the value changed from the previous commit.\n")

        all_keys = sorted({k for m in defaults_maps.values() for k in m})
        if not all_keys or not commit_hashes:
            self.add("*No code default data available.*\n")
            return

        short = [h[:7] for h in commit_hashes]
        header = "| Param | " + " | ".join(f"`{s}`" for s in short) + " |"
        sep = "|---|" + "|".join("---" for _ in short) + "|"
        self.add(header)
        self.add(sep)

        for key in all_keys:
            cells = []
            prev_val = None
            for h in commit_hashes:
                val = defaults_maps.get(h, {}).get(key, "-")
                if prev_val is not None and val != prev_val:
                    cells.append(f"**{val}**")
                else:
                    cells.append(val)
                prev_val = val
            self.add(f"| `{key}` | " + " | ".join(cells) + " |")
        self.add("")

    def write_semantic_drift_section(
        self,
        commit_pairs: list[tuple[str, str]],
        config_diffs: dict[str, str],
        code_stats: dict[str, str],
        default_diffs: dict[str, dict[str, dict[str, str]]],
    ) -> None:
        """Section 3: Semantic Drift Detector."""
        self.add("---\n## Section 3: Semantic Drift Detector\n")
        self.add("For each pair of adjacent commits with MLflow runs, classifies the transition.\n")

        for commit_a, commit_b in commit_pairs:
            pair_key = f"{commit_a[:7]}..{commit_b[:7]}"
            cfg_diff = config_diffs.get(pair_key, "")
            code_stat = code_stats.get(pair_key, "")
            label, explanation = classify_transition(cfg_diff, code_stat)

            emoji = {
                "SEMANTIC_DRIFT": "🔴",
                "ENTANGLED": "🟡",
                "CONFIG_CHANGE": "🟢",
                "IDENTICAL": "⚪",
            }.get(label, "⚪")

            self.add(f"### `{commit_a[:7]}` → `{commit_b[:7]}` — {emoji} {label}\n")
            self.add(f"> {explanation}\n")

            if code_stat:
                self.add("**Code changes:**")
                self.add(f"```\n{code_stat}\n```\n")
            else:
                self.add("*No code file changes.*\n")

            if cfg_diff:
                self.add("**Config diff:**")
                lines = cfg_diff.splitlines()
                if len(lines) > 40:
                    self.add(f"```diff\n{chr(10).join(lines[:40])}\n... ({len(lines) - 40} more lines)\n```\n")
                else:
                    self.add(f"```diff\n{cfg_diff}\n```\n")
            else:
                self.add("*Config YAML is byte-for-byte identical.*\n")

            dd = default_diffs.get(pair_key, {})
            if dd:
                self.add("**Code default changes:**\n")
                self.add("| Param | Old Default | New Default | Change |")
                self.add("|---|---|---|---|")
                for k, v in sorted(dd.items()):
                    self.add(f"| `{k}` | `{v['old']}` | `{v['new']}` | {v['change']} |")
                self.add("")

    def write_run_timeline(
        self,
        runs: list[dict],
        commit_dates: dict[str, str],
    ) -> None:
        """Section 4: Run Timeline."""
        self.add("---\n## Section 4: Run Timeline\n")
        self.add("All finished runs with metrics, sorted by date and grouped by commit.\n")

        # Group runs by commit
        by_commit: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            if r["status"] != "FINISHED":
                continue
            if not r["metrics"]:
                continue
            by_commit[r["git_commit"]].append(r)

        # Sort commits by earliest run date
        sorted_commits = sorted(
            by_commit.keys(),
            key=lambda c: min(r["start_ms"] for r in by_commit[c]),
        )

        for commit in sorted_commits:
            commit_runs = sorted(by_commit[commit], key=lambda r: r["start_ms"])
            short = commit[:7]
            label = KNOWN_RUNS.get(short, "")
            date = commit_dates.get(commit, "?")
            self.add(f"### Commit `{short}` — {date}")
            if label:
                self.add(f"*{label}*\n")
            else:
                self.add("")

            self.add("| Run Name | VPA | RPA | RPA-k1 | RPA-k5 | RPA-k20 | Edge Acc | Entry Point | Trust |")
            self.add("|---|---|---|---|---|---|---|---|---|")

            for r in commit_runs:
                m = r["metrics"]
                vpa = m.get("val_perfect_acc", -1)
                rpa = m.get("rollout_perfect_accuracy", -1)
                rpa_k1 = m.get("rollout_perfect_acc_k1", -1)
                rpa_k5 = m.get("rollout_perfect_acc_k5", -1)
                rpa_k20 = m.get("rollout_perfect_acc_k20", -1)
                ea = m.get("val_acc", -1)
                entry = "src2/" if "src2/" in r.get("source_name", "") else "src/"
                _, tier = compute_trust_tier(date, r["start_time"])

                def fmt(v: float) -> str:
                    return f"{v:.3f}" if v >= 0 else "-"

                self.add(
                    f"| {r['run_name']} | {fmt(vpa)} | {fmt(rpa)} | {fmt(rpa_k1)} "
                    f"| {fmt(rpa_k5)} | {fmt(rpa_k20)} | {fmt(ea)} | {entry} | {tier} |"
                )
            self.add("")

    def write_peak_param_diff(
        self,
        peak_run: dict | None,
        regression_runs: list[dict],
        peak_config: dict[str, str],
        regression_configs: dict[str, dict[str, str]],
        peak_defaults: dict[str, str],
        regression_defaults: dict[str, dict[str, str]],
    ) -> None:
        """Section 5: Peak Param Diff (3-Layer)."""
        self.add("---\n## Section 5: Peak Param Diff (3-Layer)\n")
        self.add(f"Compares every regression-era commit against peak (`{PEAK_COMMIT}`).\n")

        if not peak_run:
            self.add("*Peak run not found in MLflow data.*\n")
            return

        peak_params = peak_run["params"]

        for r in regression_runs:
            short = r["git_commit"][:7]
            vpa = r["metrics"].get("val_perfect_acc", -1)
            rpa = r["metrics"].get("rollout_perfect_accuracy", -1)
            vpa_s = f"{vpa:.3f}" if vpa >= 0 else "-"
            rpa_s = f"{rpa:.3f}" if rpa >= 0 else "-"
            self.add(f"### `{short}` — {r['run_name']} (vpa={vpa_s}, rpa={rpa_s})\n")

            # Layer 1: Logged param diff
            self.add("#### Layer 1: Logged Param Diff (MLflow)\n")
            all_keys = sorted(set(peak_params) | set(r["params"]))
            diffs_1 = []
            for k in all_keys:
                pv = peak_params.get(k, "-")
                rv = r["params"].get(k, "-")
                if pv != rv:
                    diffs_1.append((k, pv, rv))

            if diffs_1:
                self.add("| Param | Peak | This Run | Note |")
                self.add("|---|---|---|---|")
                for k, pv, rv in diffs_1:
                    note = ""
                    if pv == "-":
                        note = "NEW (absent at peak)"
                    elif rv == "-":
                        note = "REMOVED"
                    self.add(f"| `{k}` | `{pv}` | `{rv}` | {note} |")
            else:
                self.add("*All logged params are identical to peak.*")
            self.add("")

            # Layer 2: Config schema diff
            reg_cfg = regression_configs.get(r["git_commit"], {})
            self.add("#### Layer 2: Config Schema Diff\n")
            all_cfg_keys = sorted(set(peak_config) | set(reg_cfg))
            diffs_2 = []
            for k in all_cfg_keys:
                pv = peak_config.get(k, "-")
                rv = reg_cfg.get(k, "-")
                if pv != rv:
                    reason = ""
                    if pv == "-":
                        reason = "NEW key (feature didn't exist at peak)"
                    elif rv == "-":
                        reason = "REMOVED key"
                    else:
                        reason = "VALUE CHANGED"
                    diffs_2.append((k, pv, rv, reason))

            if diffs_2:
                self.add("| Config Key | Peak | This Commit | Attribution |")
                self.add("|---|---|---|---|")
                for k, pv, rv, reason in diffs_2:
                    self.add(f"| `{k}` | `{pv}` | `{rv}` | {reason} |")
            else:
                self.add("*Config schema is identical to peak.*")
            self.add("")

            # Layer 3: Semantic diff (code defaults)
            reg_def = regression_defaults.get(r["git_commit"], {})
            self.add("#### Layer 3: Code Semantic Diff (.get() Defaults)\n")
            dd = diff_defaults(peak_defaults, reg_def)
            if dd:
                self.add("| Param | Peak Default | This Commit Default | Change |")
                self.add("|---|---|---|---|")
                for k, v in sorted(dd.items()):
                    self.add(f"| `{k}` | `{v['old']}` | `{v['new']}` | {v['change']} |")
            else:
                self.add("*All code defaults are identical to peak.*")
            self.add("")

    def write_trust_scoring(
        self, runs: list[dict], commit_dates: dict[str, str],
        defaults_maps: dict[str, dict[str, str]],
        peak_defaults: dict[str, str],
    ) -> None:
        """Section 6: Trust Scoring."""
        self.add("---\n## Section 6: Trust Scoring\n")
        self.add("Per-run trust assessment combining temporal proximity to commit, entry point, and semantic drift risk.\n")

        finished = [r for r in runs if r["status"] == "FINISHED" and r["metrics"]]
        finished.sort(key=lambda r: r["start_ms"])

        self.add("| Run Name | Commit | VPA | RPA | Drift Hours | Trust Tier | Entry Point | Semantic Drift Risk |")
        self.add("|---|---|---|---|---|---|---|---|")

        for r in finished:
            commit = r["git_commit"]
            short = commit[:7]
            date = commit_dates.get(commit, "")
            hours, tier = compute_trust_tier(date, r["start_time"])
            vpa = r["metrics"].get("val_perfect_acc", -1)
            rpa = r["metrics"].get("rollout_perfect_accuracy", -1)
            entry = "src2/" if "src2/" in r.get("source_name", "") else "src/"

            # Semantic drift risk: do the code defaults at this commit differ from peak?
            commit_defaults = defaults_maps.get(commit, {})
            dd = diff_defaults(peak_defaults, commit_defaults)
            n_semantic_diffs = len(dd)
            risk = "NONE"
            if n_semantic_diffs > 10:
                risk = "HIGH"
            elif n_semantic_diffs > 3:
                risk = "MEDIUM"
            elif n_semantic_diffs > 0:
                risk = "LOW"

            vpa_s = f"{vpa:.3f}" if vpa >= 0 else "-"
            rpa_s = f"{rpa:.3f}" if rpa >= 0 else "-"
            hours_s = f"{hours:.1f}h" if hours >= 0 else "?"

            self.add(f"| {r['run_name']} | `{short}` | {vpa_s} | {rpa_s} | {hours_s} | {tier} | {entry} | {risk} ({n_semantic_diffs} diffs) |")
        self.add("")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLflow Forensic Cross-Reference Analysis",
    )
    parser.add_argument(
        "--mlflow-url",
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URL (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--output-path",
        default="plans/MLFLOW_FORENSIC_REPORT.md",
        help="Output Markdown report path (default: plans/MLFLOW_FORENSIC_REPORT.md)",
    )
    parser.add_argument(
        "--experiment-id",
        default="1",
        help="MLflow experiment ID to analyze (default: 1)",
    )
    args = parser.parse_args()

    print(f"[forensic] Connecting to MLflow at {args.mlflow_url}...")

    # -----------------------------------------------------------------------
    # Source A: MLflow API
    # -----------------------------------------------------------------------
    print("[forensic] Fetching all runs...")
    raw_runs = fetch_all_runs(args.mlflow_url, args.experiment_id)
    runs = [parse_run(r) for r in raw_runs]
    print(f"[forensic] Fetched {len(runs)} runs")

    # -----------------------------------------------------------------------
    # Source B: Git history
    # -----------------------------------------------------------------------
    print("[forensic] Loading git history...")
    git_commits = git_log_all()
    commit_date_map: dict[str, str] = {c["hash"]: c["date"] for c in git_commits}
    commit_subject_map: dict[str, str] = {c["hash"]: c["subject"] for c in git_commits}

    # Distinct commits appearing in runs
    run_commits = sorted(
        {r["git_commit"] for r in runs if r["git_commit"]},
        key=lambda h: commit_date_map.get(h, ""),
    )
    print(f"[forensic] {len(run_commits)} distinct commits with MLflow runs")

    # -----------------------------------------------------------------------
    # Source B + C: Per-commit config + code defaults
    # -----------------------------------------------------------------------
    print("[forensic] Extracting per-commit configs and code defaults...")
    config_maps: dict[str, dict[str, str]] = {}
    defaults_maps: dict[str, dict[str, str]] = {}

    for commit in run_commits:
        # Config YAML
        cfg = get_config_at_commit(commit)
        if cfg:
            config_maps[commit] = flatten_config(cfg)
        else:
            config_maps[commit] = {}

        # Code .get() defaults
        defaults_maps[commit] = get_code_defaults_for_commit(commit)
        print(f"  {commit[:7]}: config={len(config_maps[commit])} keys, "
              f"defaults={len(defaults_maps[commit])} .get() entries")

    # -----------------------------------------------------------------------
    # Compute pairwise diffs for adjacent commits
    # -----------------------------------------------------------------------
    print("[forensic] Computing pairwise diffs...")
    commit_pairs = list(zip(run_commits[:-1], run_commits[1:]))
    config_diffs: dict[str, str] = {}
    code_stats: dict[str, str] = {}
    default_diffs_map: dict[str, dict[str, dict[str, str]]] = {}

    for a, b in commit_pairs:
        pair_key = f"{a[:7]}..{b[:7]}"
        config_diffs[pair_key] = git_diff_config(a, b)
        # Try src2/ first, fall back to src/
        stat = git_diff_stat(a, b, "src2/")
        if not stat.strip():
            stat = git_diff_stat(a, b, "src/")
        code_stats[pair_key] = stat
        default_diffs_map[pair_key] = diff_defaults(
            defaults_maps.get(a, {}), defaults_maps.get(b, {})
        )

    # -----------------------------------------------------------------------
    # Identify peak run and regression runs
    # -----------------------------------------------------------------------
    peak_run = None
    for r in runs:
        if r["git_commit"].startswith(PEAK_COMMIT):
            if peak_run is None:
                peak_run = r
            else:
                # Pick the one with higher VPA
                vpa_curr = peak_run["metrics"].get("rollout_perfect_accuracy",
                           peak_run["metrics"].get("val_perfect_acc", -1))
                vpa_new = r["metrics"].get("rollout_perfect_accuracy",
                           r["metrics"].get("val_perfect_acc", -1))
                if vpa_new > vpa_curr:
                    peak_run = r

    peak_config = config_maps.get(PEAK_COMMIT_FULL, config_maps.get(
        next((c for c in run_commits if c.startswith(PEAK_COMMIT)), ""), {}
    ))
    peak_defaults = defaults_maps.get(PEAK_COMMIT_FULL, defaults_maps.get(
        next((c for c in run_commits if c.startswith(PEAK_COMMIT)), ""), {}
    ))

    # Regression era: runs at commits after the peak
    regression_era_runs = []
    for r in runs:
        if r["status"] != "FINISHED" or not r["metrics"]:
            continue
        commit = r["git_commit"]
        if not commit or commit.startswith(PEAK_COMMIT):
            continue
        date = commit_date_map.get(commit, "")
        peak_date = commit_date_map.get(
            next((c for c in run_commits if c.startswith(PEAK_COMMIT)), ""), ""
        )
        if date > peak_date:
            regression_era_runs.append(r)
    regression_era_runs.sort(key=lambda r: r["start_ms"])

    # Configs/defaults for regression commits
    regression_configs: dict[str, dict[str, str]] = {}
    regression_defaults: dict[str, dict[str, str]] = {}
    for r in regression_era_runs:
        c = r["git_commit"]
        if c not in regression_configs:
            regression_configs[c] = config_maps.get(c, {})
            regression_defaults[c] = defaults_maps.get(c, {})

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    print("[forensic] Generating report...")
    report = ForensicReport()
    report.write_header(len(runs), len(run_commits))
    report.write_config_schema_table(run_commits, config_maps)
    report.write_code_defaults_table(run_commits, defaults_maps)
    report.write_semantic_drift_section(
        commit_pairs, config_diffs, code_stats, default_diffs_map,
    )
    report.write_run_timeline(runs, commit_date_map)
    report.write_peak_param_diff(
        peak_run,
        regression_era_runs,
        peak_config,
        regression_configs,
        peak_defaults,
        regression_defaults,
    )
    report.write_trust_scoring(runs, commit_date_map, defaults_maps, peak_defaults)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.render())
    print(f"[forensic] Report written to {output_path}")
    print(f"[forensic] Done. {len(runs)} runs across {len(run_commits)} commits analyzed.")


if __name__ == "__main__":
    main()
