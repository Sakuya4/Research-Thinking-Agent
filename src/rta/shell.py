from __future__ import annotations

import json
import os
import sys
import textwrap
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from .config import RTAConfig
from .schemas import InputPayload
from .pipeline import run_pipeline


# ---------- UI helpers (Codex/Gemini CLI-ish) ----------

def _supports_color() -> bool:
    return sys.stdout.isatty()

def _c(s: str, code: str) -> str:
    if not _supports_color():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"

def badge_ok(msg: str) -> str:
    return _c("[OK] ", "32;1") + msg

def badge_warn(msg: str) -> str:
    return _c("[WARN] ", "33;1") + msg

def badge_err(msg: str) -> str:
    return _c("[ERR] ", "31;1") + msg

def title(msg: str) -> str:
    return _c(msg, "36;1")

def dim(msg: str) -> str:
    return _c(msg, "2")

def hr() -> str:
    return dim("─" * 60)


# ---------- Shell ----------

class RTAShell:
    def __init__(self, cfg: Optional[RTAConfig] = None):
        self.cfg = cfg or RTAConfig()
        self.last_run_dir: Optional[Path] = None
        # extra knobs not in config (to avoid breaking pydantic)
        self.sources = os.getenv("RTA_SOURCES", "both").lower()  # both|arxiv|s2

    def run(self) -> None:
        print(title("Research Thinking Agent"))
        print(dim("Type /help for commands. Use /run <topic> to start.\n"))

        while True:
            try:
                line = input(_c("rta> ", "35;1")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                continue

            if line.startswith("/"):
                if self._handle_command(line):
                    return
                continue

            # convenience: treat plain text as /run
            self._cmd_run(line)

    def _handle_command(self, line: str) -> bool:
        parts = line[1:].strip().split()
        cmd = (parts[0].lower() if parts else "")
        args = parts[1:]

        if cmd in ("exit", "quit"):
            return True

        if cmd == "help":
            self._cmd_help()
            return False

        if cmd == "run":
            topic = " ".join(args).strip()
            if not topic:
                print(badge_err("Usage: /run <topic>"))
            else:
                self._cmd_run(topic)
            return False

        if cmd == "last":
            self._cmd_last()
            return False

        if cmd == "set":
            self._cmd_set(args)
            return False

        if cmd == "show":
            self._cmd_show(args)
            return False

        if cmd == "open":
            self._cmd_open(args)
            return False

        print(badge_err(f"Unknown command: /{cmd}. Try /help"))
        return False

    def _cmd_help(self) -> None:
        print(hr())
        print(title("Commands"))
        print("  /run <topic>                 Run once (plain text also works)")
        print("  /set <key> <value>           Set config for this session")
        print("     keys: max_papers, min_year, max_year, retrieval_mode, sources")
        print("     sources: both | arxiv | s2")
        print("  /show config|plan|retrieval|status   Print JSON (trimmed)")
        print("  /open report|plan|retrieval|status   Open file with default app")
        print("  /last                        Show last run directory")
        print("  /exit                        Quit")
        print(hr())

    def _cmd_last(self) -> None:
        if not self.last_run_dir:
            print(badge_warn("No runs yet. Use /run <topic>."))
            return
        print(badge_ok(f"Last run: {self.last_run_dir}"))

    def _cmd_set(self, args) -> None:
        if len(args) < 2:
            print(badge_err("Usage: /set <key> <value>"))
            return
        key = args[0].lower()
        value = " ".join(args[1:]).strip()

        try:
            if key in ("max_papers", "min_year", "max_year", "cache_ttl_hours"):
                setattr(self.cfg, key, int(value))
            elif key in ("retrieval_mode",):
                setattr(self.cfg, key, value)
            elif key == "sources":
                v = value.lower()
                if v not in ("both", "arxiv", "s2"):
                    raise ValueError("sources must be: both|arxiv|s2")
                self.sources = v
            else:
                raise ValueError(f"Unknown key: {key}")

            print(badge_ok(f"Set {key} = {value}"))
        except Exception as e:
            print(badge_err(str(e)))

    def _cmd_show(self, args) -> None:
        if not args:
            print(badge_err("Usage: /show config|plan|retrieval|status"))
            return
        what = args[0].lower()

        if what == "config":
            print(json.dumps(self.cfg.model_dump(), ensure_ascii=False, indent=2))
            print(badge_ok(f"sources = {self.sources}"))
            return

        if not self.last_run_dir:
            print(badge_warn("No runs yet. Use /run <topic>."))
            return

        mapping = {
            "plan": "query_plan.json",
            "retrieval": "retrieval.json",
            "status": "status.json",
        }
        fname = mapping.get(what)
        if not fname:
            print(badge_err("Usage: /show config|plan|retrieval|status"))
            return

        p = self.last_run_dir / fname
        if not p.exists():
            print(badge_warn(f"File not found: {p}"))
            return

        txt = p.read_text(encoding="utf-8", errors="replace")
        # trim huge outputs
        if len(txt) > 5000:
            txt = txt[:5000] + "\n...\n"
        print(txt)

    def _cmd_open(self, args) -> None:
        if not args:
            print(badge_err("Usage: /open report|plan|retrieval|status"))
            return
        what = args[0].lower()

        if not self.last_run_dir:
            print(badge_warn("No runs yet. Use /run <topic>."))
            return

        mapping = {
            "report": "report.md",
            "plan": "query_plan.json",
            "retrieval": "retrieval.json",
            "status": "status.json",
        }
        fname = mapping.get(what)
        if not fname:
            print(badge_err("Usage: /open report|plan|retrieval|status"))
            return

        p = self.last_run_dir / fname
        if not p.exists():
            print(badge_warn(f"File not found: {p}"))
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
            print(badge_ok(f"Opened: {p}"))
        except Exception as e:
            print(badge_err(f"Open failed: {e}"))

    def _cmd_run(self, topic: str) -> None:
        print(hr())
        print(title(f"Topic: {topic}"))
        print(dim(f"Config: max_papers={self.cfg.max_papers}, mode={self.cfg.retrieval_mode}, sources={self.sources}"))
        print(hr())

        # pass sources via env for retrieval_live to read (minimal change)
        os.environ["RTA_SOURCES"] = self.sources

        run_id, run_dir = run_pipeline(self.cfg, InputPayload(query=topic, context=""))
        self.last_run_dir = Path(run_dir)

        print(badge_ok(f"Saved outputs: {run_dir}"))
        print(dim(f"Try: /show retrieval  |  /open report  |  /last"))
        print(hr())
