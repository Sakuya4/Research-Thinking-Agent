from __future__ import annotations

import json
import os
import pydoc
import html
import re
from pathlib import Path
from typing import Optional, Dict, List

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
        self.sources = os.getenv("RTA_SOURCES", "both").lower()

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
                "/help", "/run", "/set", "/show", "/open", "/last", "/exit",
                "max_papers", "min_year", "max_year", "retrieval_mode",
                "sources", "both", "arxiv", "s2",
                "config", "plan", "retrieval", "status", "report", "reasoning",
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
        _print(self.style, "hint", "Type /help for commands. Use /run <topic> to start.\n")

        if not _has_gemini_key():
            self._warned_missing_key = True
            _print(self.style, "warn", "[WARN] GEMINI_API_KEY not set. Live LLM steps may be unavailable.")
            _print(self.style, "hint", "       Create a .env file (recommended) or set $env:GEMINI_API_KEY='...'\n")

        while True:
            try:
                line = self._session.prompt(HTML("<prompt>rta&gt; </prompt>"), style=self.style).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                _print(self.style, "dim", "Tip: /run <topic>  |  /set retrieval_mode mock  |  /open report  |  /help")
                continue

            if line.startswith("/"):
                should_exit = self._handle_command(line)
                if should_exit:
                    return
                continue

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

    def _cmd_help(self) -> None:
        _hr(self.style)
        _print(self.style, "title", "Commands")
        _print_kv(self.style, "/run <topic>", "Run once")
        _print_kv(self.style, "/set <key> <val>", "Set config (max_papers, retrieval_mode...)")
        _print_kv(self.style, "/show <what>", "Print JSON output")
        _print_kv(self.style, "/open <what>", "View result files")
        _print_kv(self.style, "/last", "Show last run info")
        _print_kv(self.style, "/exit", "Quit")
        _hr(self.style)

    def _cmd_last(self) -> None:
        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet.")
            return
        _print(self.style, "ok", f"[OK] Last run: {self.last_run_dir}")

    def _cmd_set(self, args) -> None:
        if len(args) < 2:
            _print(self.style, "err", "[ERR] Usage: /set <key> <value>")
            return
        key, value = args[0].lower(), " ".join(args[1:]).strip()
        try:
            if key in ("max_papers", "min_year", "max_year", "cache_ttl_hours"):
                setattr(self.cfg, key, int(value))
            elif key == "retrieval_mode":
                if value.lower() not in ("mock", "live"): raise ValueError("mock|live")
                setattr(self.cfg, key, value.lower())
            elif key == "sources":
                if value.lower() not in ("both", "arxiv", "s2"): raise ValueError("both|arxiv|s2")
                self.sources = value.lower()
            else:
                raise ValueError(f"Unknown key: {key}")
            _print(self.style, "ok", f"[OK] Set {key} = {value}")
        except Exception as e:
            _print(self.style, "err", f"[ERR] {e}")

    def _cmd_show(self, args) -> None:
        if not args:
            _print(self.style, "err", "[ERR] Usage: /show <what>")
            return
        what = args[0].lower()
        if what == "config":
            print(json.dumps(self.cfg.model_dump(), indent=2))
            return
        if not self.last_run_dir:
            _print(self.style, "warn", "[WARN] No runs yet.")
            return
        mapping = {"plan": "plan.json", "retrieval": "retrieval.json", "status": "structuring.json", "reasoning": "reasoning.json"}
        fname = mapping.get(what)
        if not fname: return
        p = self.last_run_dir / fname
        if p.exists():
            print(p.read_text(encoding="utf-8", errors="replace")[:6000])

    def _cmd_open(self, args) -> None:
        if not args: return
        what = args[0].lower()
        mapping = {"report": "report.md", "plan": "plan.json", "retrieval": "retrieval.json", "status": "structuring.json", "reasoning": "reasoning.json"}
        fname = mapping.get(what)
        if not fname or not self.last_run_dir: return
        p = self.last_run_dir / fname
        if p.exists():
            pydoc.pager(p.read_text(encoding="utf-8", errors="replace"))

    # --------------------------------------------------------------------------
    # CHAT MODE (Context-Isolated & Multi-Turn)
    # --------------------------------------------------------------------------
    def _enter_chat_mode(self, topic: str, run_dir: str):
        """Interactive chat with citation support and strict context isolation."""
        from rich.console import Console
        from rich.rule import Rule
        try:
            from rta.utils.llm_client import get_default_client
        except ImportError:
            return

        console = Console()
        client = get_default_client()
        run_path = Path(run_dir)

        # 1. Load Retrieved Papers for THIS session only
        papers_context = ""
        paper_map: Dict[str, str] = {}
        retrieval_file = run_path / "retrieval.json"
        
        paper_count = 0
        if retrieval_file.exists():
            try:
                data = json.loads(retrieval_file.read_text(encoding="utf-8"))
                if isinstance(data, list): papers = data
                else: papers = data.get('papers', [])
                
                context_lines = []
                for idx, p in enumerate(papers, 1):
                    title = p.get('title', 'Unknown Title')
                    authors = p.get('authors', [])
                    first_author = authors[0] if isinstance(authors, list) and authors else "Unknown"
                    year = p.get('year', 'n.d.')
                    
                    citation_key = f"[{idx}]"
                    citation_text = f"{title} ({first_author}, {year})"
                    paper_map[citation_key] = citation_text
                    context_lines.append(f"{citation_key} {citation_text}")
                
                papers_context = "\n".join(context_lines)
                paper_count = len(papers)
            except Exception:
                papers_context = "(Failed to load papers)"

        console.print(f"\n[bold green]RTA Chat Mode: '{topic}'[/bold green]")
        console.print(f"[dim]Loaded {paper_count} papers from this run only.[/dim]")
        console.print("[dim]Type 'exit' to quit. Type '/reset' to clear chat history.[/dim]\n")

        # 2. System Prompt (REMOVED specific examples like LVEF to prevent hallucination)
        system_prompt_base = (
            f"You are a research assistant expert in '{topic}'.\n"
            f"You have access to the following retrieved papers found in THIS specific session:\n"
            f"--------------------------------------------------\n"
            f"{papers_context}\n"
            f"--------------------------------------------------\n"
            f"RULES:\n"
            f"1. Answer strictly in English.\n"
            f"2. Use ONLY the provided papers and general knowledge. Do not infer from previous sessions.\n"
            f"3. Proactively suggest relevant applications or extensions based on the papers.\n"
            f"4. CITATION RULE: You MUST cite sources using [ID] when relevant.\n"
        )

        # 3. Conversation History (Short-term memory for THIS session)
        chat_history: List[str] = []

        while True:
            try:
                user_input = console.input("\n[bold blue](You) > [/bold blue]").strip()
                
                # --- Chat Commands ---
                if user_input.lower() in ['exit', 'quit']:
                    console.print("[yellow]Exiting chat mode.[/yellow]")
                    break
                
                if user_input.lower() in ['/clear', '/reset']:
                    chat_history = []
                    console.print("[yellow]Chat history cleared. Context reset.[/yellow]")
                    continue

                if user_input.lower() == '/context':
                    console.print(f"[dim]Current Context: {paper_count} papers loaded.[/dim]")
                    continue

                if not user_input: continue

                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    
                    # Build Prompt: System + History + New Input
                    conversation_text = "\n".join(chat_history)
                    final_prompt = (
                        f"{system_prompt_base}\n\n"
                        f"Conversation History:\n{conversation_text}\n\n"
                        f"User: {user_input}\n"
                        f"RTA:"
                    )

                    response = client.generate_text(final_prompt)

                console.print(f"\n[bold cyan](RTA)[/bold cyan]: {response}")

                # Update History
                chat_history.append(f"User: {user_input}")
                chat_history.append(f"RTA: {response}")

                # Display Citations
                found_citations = sorted(list(set(re.findall(r'\[\d+\]', response))))
                if found_citations:
                    console.print(Rule(style="dim"))
                    console.print("[bold dim]References mentioned:[/bold dim]")
                    for key in found_citations:
                        if key in paper_map:
                            console.print(f"[green]{key}[/green] {paper_map[key]}")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Exiting chat mode.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Chat Error: {e}[/red]")
                break

    def _cmd_run(self, topic: str) -> None:
        _hr(self.style)
        _print(self.style, "title", f"Topic: {topic}")
        _print(self.style, "dim", f"Config: {self.cfg.max_papers} papers, {self.cfg.retrieval_mode} mode")
        _hr(self.style)
        os.environ["RTA_SOURCES"] = self.sources
        try:
            success, run_dir = run_pipeline(topic, output_dir=self.cfg.runs_dir)
            self.last_run_dir = Path(run_dir)
            if success:
                _print(self.style, "ok", f"[OK] Saved: {run_dir}")
                self._enter_chat_mode(topic, run_dir)
            else:
                _print(self.style, "err", "[Fail] Pipeline failed.")
        except Exception as e:
            latest = _find_latest_run_dir(self.cfg.runs_dir)
            if latest: self.last_run_dir = latest
            _print(self.style, "err", f"[ERR] {e}")