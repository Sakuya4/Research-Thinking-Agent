from __future__ import annotations

import json
import os
import pydoc
import html
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style

from .config import RTAConfig
from .pipeline import run_pipeline
from .schemas import InputPayload


RTA_BANNER = r"""
██████╗ ████████╗ █████╗
██╔══██╗╚══██╔══╝██╔══██╗
██████╔╝   ██║   ███████║
██╔══██╗   ██║   ██╔══██║
██║  ██║   ██║   ██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
""".strip("\n")


def _print(style: Style, tag: str, msg: str) -> None:
    safe = html.escape(msg)
    print_formatted_text(HTML(f"<{tag}>{safe}</{tag}>"), style=style)

def _print_kv(style: Style, key: str, value: str) -> None:
    import html as _html
    k = _html.escape(key)
    v = _html.escape(value)
    print_formatted_text(HTML(f"<dim>{k}</dim> {v}"), style=style)

def _hr(style: Style) -> None:
    print_formatted_text(HTML("<dim>────────────────────────────────────────────────────────────</dim>"), style=style)


class RTAShell:
    def __init__(self, cfg: Optional[RTAConfig] = None):
        self.cfg = cfg or RTAConfig()
        self.last_run_dir: Optional[Path] = None
        self.sources = os.getenv("RTA_SOURCES", "both").lower()  # both|arxiv|s2

        self.style = Style.from_dict(
            {
                "banner": "bold #00afff",
                "title": "bold #5fd7ff",
                "hint": "#888888",
                "dim": "#888888",
                "prompt": "bold #00afff",
                "ok": "bold #5fff87",
                "warn": "bold #ffd75f",
                "err": "bold #ff5f5f",
            }
        )

        hist_path = Path(self.cfg.runs_dir) / ".rta_history"
        hist_path.parent.mkdir(parents=True, exist_ok=True)

        self._completer = WordCompleter(
            [
                "/help",
                "/run",
                "/set",
                "/show",
                "/open",
                "/last",
                "/exit",
                "max_papers",
                "min_year",
                "max_year",
                "retrieval_mode",
                "sources",
                "both",
                "arxiv",
                "s2",
                "config",
                "plan",
                "retrieval",
                "status",
                "report",
            ],
            ignore_case=True,
        )

        self._session = PromptSession(
            history=FileHistory(str(hist_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._completer,
        )

    # ----------------------
    # Main loop
    # ----------------------
    def run(self) -> None:
        print_formatted_text(HTML(f"<banner>{RTA_BANNER}</banner>"), style=self.style)
        _print(self.style, "title", "Research Thinking Agent")
        _print(self.style, "hint", "Type /help for commands. Use /run <topic> to start.\n")

        while True:
            try:
                line = self._session.prompt(HTML("<prompt>rta&gt; </prompt>"), style=self.style).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                _print(self.style, "dim", "Tip: /run <topic>  |  /set sources arxiv  |  /open report  |  /help")
                continue

            # Slash commands
            if line.startswith("/"):
                should_exit = self._handle_command(line)
                if should_exit:
                    return
                continue

            # Plain text => run
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
                _print(self.style, "err", "[ERR] Usage: /run <topic>")
            else:
                self._cmd_run(topic)
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

        if cmd == "last":
            self._cmd_last()
            return False

        _print(self.style, "err", f"[ERR] Unknown command: /{cmd}. Try /help")
        return False

    # ----------------------
    # Commands
    # ----------------------
    def _cmd_help(self) -> None:
        _hr(self.style)
        _print(self.style, "title", "Commands")
        _print_kv(self.style, "/run <topic>", "Run once (plain text also works)")
        _print_kv(self.style, "/set <key> <value>", "Set config for this session")
        _print_kv(self.style, "  keys", "max_papers, min_year, max_year, retrieval_mode, sources")
        _print_kv(self.style, "  sources", "both | arxiv | s2")
        _print_kv(self.style, "/show <what>", "Print JSON (trimmed): config|plan|retrieval|status")
        _print_kv(self.style, "/open <what>", "View file in CLI pager: report|plan|retrieval|status")
        _print_kv(self.style, "/last", "Show last run directory")
        _print_kv(self.style, "/exit", "Quit")
        _hr(self.style)

    def _cmd_last(self) -> None:
        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet. Use /run <topic>.")
            return
        _print(self.style, "ok", f"[OK] Last run: {self.last_run_dir}")

    def _cmd_set(self, args) -> None:
        if len(args) < 2:
            _print(self.style, "err", "[ERR] Usage: /set <key> <value>")
            return

        key = args[0].lower()
        value = " ".join(args[1:]).strip()

        try:
            if key in ("max_papers", "min_year", "max_year", "cache_ttl_hours"):
                setattr(self.cfg, key, int(value))
                _print(self.style, "ok", f"[OK] Set {key} = {value}")
                return

            if key == "retrieval_mode":
                setattr(self.cfg, key, value)
                _print(self.style, "ok", f"[OK] Set {key} = {value}")
                return

            if key == "sources":
                v = value.lower()
                if v not in ("both", "arxiv", "s2"):
                    raise ValueError("sources must be: both|arxiv|s2")
                self.sources = v
                _print(self.style, "ok", f"[OK] Set sources = {v}")
                return

            raise ValueError(f"Unknown key: {key}")

        except Exception as e:
            _print(self.style, "err", f"[ERR] {e}")

    def _cmd_show(self, args) -> None:
        if not args:
            _print(self.style, "err", "[ERR] Usage: /show config|plan|retrieval|status")
            return

        what = args[0].lower()

        if what == "config":
            cfg_obj = self.cfg.model_dump()
            cfg_obj["sources"] = self.sources
            print(json.dumps(cfg_obj, ensure_ascii=False, indent=2))
            return

        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet. Use /run <topic>.")
            return

        mapping = {
            "plan": "query_plan.json",
            "retrieval": "retrieval.json",
            "status": "status.json",
        }
        fname = mapping.get(what)
        if not fname:
            _print(self.style, "err", "[ERR] Usage: /show config|plan|retrieval|status")
            return

        p = self.last_run_dir / fname
        if not p.exists():
            _print(self.style, "warn", f"[WARN] File not found: {p}")
            return

        txt = p.read_text(encoding="utf-8", errors="replace")
        if len(txt) > 6000:
            txt = txt[:6000] + "\n...\n"
        print(txt)

    def _cmd_open(self, args) -> None:
        if not args:
            _print(self.style, "err", "[ERR] Usage: /open report|plan|retrieval|status")
            return

        what = args[0].lower()

        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet. Use /run <topic>.")
            return

        mapping = {
            "report": "report.md",
            "plan": "query_plan.json",
            "retrieval": "retrieval.json",
            "status": "status.json",
        }
        fname = mapping.get(what)
        if not fname:
            _print(self.style, "err", "[ERR] Usage: /open report|plan|retrieval|status")
            return

        p = self.last_run_dir / fname
        if not p.exists():
            _print(self.style, "warn", f"[WARN] File not found: {p}")
            return

        txt = p.read_text(encoding="utf-8", errors="replace")
        pydoc.pager(txt)

    def _cmd_run(self, topic: str) -> None:
        _hr(self.style)
        _print(self.style, "title", f"Topic: {topic}")
        _print(self.style, "dim", f"Config: max_papers={self.cfg.max_papers}, mode={self.cfg.retrieval_mode}, sources={self.sources}")
        _hr(self.style)

        os.environ["RTA_SOURCES"] = self.sources

        try:
            _, run_dir = run_pipeline(self.cfg, InputPayload(query=topic, context=""))
        except Exception as e:
            _print(self.style, "err", f"[ERR] {e}")
            _print(self.style, "hint", "Tip: set GEMINI_API_KEY in .env or PowerShell: $env:GEMINI_API_KEY='...'\n")
            return
        self.last_run_dir = Path(run_dir)

        _print(self.style, "ok", f"[OK] Saved outputs: {run_dir}")
        _print(self.style, "dim", "Try: /show retrieval  |  /open report  |  /last")
        _hr(self.style)
