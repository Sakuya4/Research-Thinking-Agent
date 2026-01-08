from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import DEFAULT_CONFIG
from .pipeline import run_pipeline
from .schemas import InputPayload, QueryPlan
from .logger import EventLogger
from .agent_reply import build_agent_reply, print_agent_reply


def interactive_loop() -> None:
    typer.echo("Research Thinking Agent (interactive mode)")
    typer.echo("Type a topic, or 'exit' to quit.\n")

    context: Optional[str] = None

    while True:
        try:
            user_text = typer.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n[exit]")
            break

        if user_text.lower() in {"exit", "quit"}:
            typer.echo("[exit]")
            break

        if not user_text:
            continue

        # Optional: allow prefix ":" like your example
        if user_text.startswith(":"):
            user_text = user_text[1:].strip()

        typer.echo(f"\nRTA: Got it. Topic = '{user_text}'.")
        typer.echo("RTA: Thinking and generating a research plan...\n")

        payload = InputPayload(query=user_text, context=context)

        # Run the pipeline (Stage1 -> query_plan.json)
        run_id, run_dir = run_pipeline(DEFAULT_CONFIG, payload)
        run_dir_path = Path(run_dir)

        # Load query_plan.json
        qp: Optional[QueryPlan] = None
        qp_path = run_dir_path / "query_plan.json"
        if qp_path.exists():
            with qp_path.open("r", encoding="utf-8") as f:
                qp = QueryPlan.model_validate(json.load(f))

        # Use the same run folder log file
        reply_logger = EventLogger(log_path=run_dir_path / "logs.jsonl")

        # Ask Gemini to produce an English, user-facing reply (summary/glossary/directions)
        reply = build_agent_reply(payload, reply_logger, qp, run_dir_path)
        print_agent_reply(reply, run_dir_path)

        # Simple rolling context
        context = f"Previous topic: {user_text}"
