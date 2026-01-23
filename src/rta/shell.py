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
""".strip(
    "\n"
)


def _print(style: Style, tag: str, msg: str) -> None:
    safe = html.escape(msg)
    print_formatted_text(HTML(f"<{tag}>{safe}</{tag}>"), style=style)


def _print_kv(style: Style, key: str, value: str) -> None:
    import html as _html

    k = _html.escape(key)
    v = _html.escape(value)
    print_formatted_text(HTML(f"<dim>{k}</dim> {v}"), style=style)


def _hr(style: Style) -> None:
    print_formatted_text(
        HTML("<dim>────────────────────────────────────────────────────────────</dim>"),
        style=style,
    )


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _find_latest_run_dir(runs_dir: str) -> Optional[Path]:
    p = Path(runs_dir)
    if not p.exists():
        return None
    dirs = [d for d in p.iterdir() if d.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[0]


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
                "reasoning",
            ],
            ignore_case=True,
        )

        self._session = PromptSession(
            history=FileHistory(str(hist_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._completer,
        )

        self._warned_missing_key = False

    def run(self) -> None:
        print_formatted_text(HTML(f"\n<banner>{RTA_BANNER}</banner>"), style=self.style)
        _print(self.style, "title", "Research Thinking Agent")
        _print(
            self.style, "hint", "Type /help for commands. Use /run <topic> to start.\n"
        )

        if not _has_gemini_key():
            self._warned_missing_key = True
            _print(
                self.style,
                "warn",
                "[WARN] GEMINI_API_KEY not set. Live LLM steps may be unavailable.",
            )
            _print(
                self.style,
                "hint",
                "       Create a .env file (recommended) or set $env:GEMINI_API_KEY='...'\n",
            )

        while True:
            try:
                line = self._session.prompt(
                    HTML("<prompt>rta&gt; </prompt>"), style=self.style
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                _print(
                    self.style,
                    "dim",
                    "Tip: /run <topic>  |  /set retrieval_mode mock  |  /open report  |  /help",
                )
                continue

            if line.startswith("/"):
                should_exit = self._handle_command(line)
                if should_exit:
                    return
                continue

            self._cmd_run(line)

    def _handle_command(self, line: str) -> bool:
        parts = line[1:].strip().split()
        cmd = parts[0].lower() if parts else ""
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

    def _cmd_help(self) -> None:
        _hr(self.style)
        _print(self.style, "title", "Commands")
        _print_kv(self.style, "/run <topic>", "Run once (plain text also works)")
        _print_kv(self.style, "/set <key> <value>", "Set config for this session")
        _print_kv(
            self.style,
            "  keys",
            "max_papers, min_year, max_year, retrieval_mode, sources",
        )
        _print_kv(self.style, "  retrieval_mode", "mock | live")
        _print_kv(self.style, "  sources", "both | arxiv | s2")
        _print_kv(
            self.style,
            "/show <what>",
            "Print JSON (trimmed): config|plan|retrieval|status|reasoning",
        )
        _print_kv(
            self.style,
            "/open <what>",
            "View file in CLI pager: report|plan|retrieval|status|reasoning",
        )
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
                v = value.lower()
                if v not in ("mock", "live"):
                    raise ValueError("retrieval_mode must be: mock|live")
                setattr(self.cfg, key, v)
                _print(self.style, "ok", f"[OK] Set {key} = {v}")
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
            _print(
                self.style,
                "err",
                "[ERR] Usage: /show config|plan|retrieval|status|reasoning",
            )
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
            "plan": "plan.json",  # [FIX] aligned filename
            "retrieval": "retrieval.json",
            "status": "structuring.json",  # [FIX] aligned filename
            "reasoning": "reasoning.json",
        }
        fname = mapping.get(what)
        if not fname:
            _print(
                self.style,
                "err",
                "[ERR] Usage: /show config|plan|retrieval|status|reasoning",
            )
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
            _print(
                self.style,
                "err",
                "[ERR] Usage: /open report|plan|retrieval|status|reasoning",
            )
            return

        what = args[0].lower()

        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet. Use /run <topic>.")
            return

        mapping = {
            "report": "report.md",  # [FIX] aligned filename
            "plan": "plan.json",  # [FIX] aligned filename
            "retrieval": "retrieval.json",
            "status": "structuring.json",  # [FIX] aligned filename
            "reasoning": "reasoning.json",
        }
        fname = mapping.get(what)
        if not fname:
            _print(
                self.style,
                "err",
                "[ERR] Usage: /open report|plan|retrieval|status|reasoning",
            )
            return

        p = self.last_run_dir / fname
        if not p.exists():
            _print(self.style, "warn", f"[WARN] File not found: {p}")
            return

        txt = p.read_text(encoding="utf-8", errors="replace")
        pydoc.pager(txt)

    # --------------------------------------------------------------------------
    # NEW: Chat Mode Method
    # --------------------------------------------------------------------------
    def _enter_chat_mode(self, topic, report_path):
        """Starts an interactive chat session about the research results."""
        from rich.console import Console

        # Use local import to avoid circular dependency
        try:
            from rta.utils.llm_client import get_default_client
        except ImportError:
            _print(self.style, "err", "[Chat] Failed to import LLM client.")
            return

        console = Console()
        client = get_default_client()

        console.print(
            "\n[bold green]RTA is ready to discuss findings with you (in English).[/bold green]"
        )
        console.print(f"[dim]Based on: {topic} and retrieved papers.[/dim]")
        console.print("[dim]Type 'exit' or 'quit' to end the chat.[/dim]\n")

        # System prompt to enforce the persona
        system_context = (
            f"You are a helpful research assistant. You have just finished analyzing the topic '{topic}'. "
            f"The user wants to discuss the findings. "
            f"Your goal is to help the user Brainstorm and Extend the research.\n"
            f"Rules:\n"
            f"1. Answer strictly in English.\n"
            f"2. Be concise but insightful.\n"
            f"3. Proactively suggest applications (e.g., if relevant, mention LVEF, clinical integration, etc.)."
        )

        # Chat Loop
        while True:
            try:
                # Use standard input for simplicity in this mode
                user_input = console.input("\n[bold blue](You) > [/bold blue]").strip()

                if user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Exiting chat mode.[/yellow]")
                    break

                if not user_input:
                    continue

                with console.status(
                    "[bold blue]RTA is thinking...[/bold blue]", spinner="dots"
                ):
                    # Construct prompt
                    prompt = f"{system_context}\n\n" f"User: {user_input}\n" f"RTA:"
                    # Call LLM
                    response = client.generate_text(prompt)

                # Print response nicely
                console.print(f"\n[bold cyan](RTA)[/bold cyan]: {response}")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Exiting chat mode.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Chat Error: {e}[/red]")
                break

    def _cmd_run(self, topic: str) -> None:
        _hr(self.style)
        _print(self.style, "title", f"Topic: {topic}")
        _print(
            self.style,
            "dim",
            f"Config: max_papers={self.cfg.max_papers}, mode={self.cfg.retrieval_mode}, sources={self.sources}",
        )
        _hr(self.style)

        os.environ["RTA_SOURCES"] = self.sources

        try:
            # [FIX] Correct function call matching the latest pipeline.py
            success, run_dir = run_pipeline(topic, output_dir=self.cfg.runs_dir)

            self.last_run_dir = Path(run_dir)

            if success:
                _print(self.style, "ok", f"[OK] Saved outputs: {run_dir}")
                _print(
                    self.style,
                    "dim",
                    "Try: /show retrieval  |  /show status  |  /open report  |  /last",
                )
                # [NEW] Enter chat mode automatically
                self._enter_chat_mode(topic, run_dir)
            else:
                _print(
                    self.style, "err", "[Fail] Pipeline did not complete successfully."
                )

            _hr(self.style)
            return

        except Exception as e:
            # Even on failure, try to set last_run_dir to the latest created run
            latest = _find_latest_run_dir(self.cfg.runs_dir)
            if latest is not None:
                self.last_run_dir = latest

            _print(self.style, "err", f"[ERR] {e}")
            if "GEMINI_API_KEY" in str(e):
                _print(
                    self.style,
                    "hint",
                    "Tip: create a .env file with GEMINI_API_KEY=... (recommended)\n",
                )
            else:
                _print(
                    self.style,
                    "hint",
                    "Tip: use /last then /open status to inspect failure.\n",
                )
            return
