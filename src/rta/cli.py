from __future__ import annotations

# ---- Environment bootstrap (MUST be first) ----
import os
from dotenv import load_dotenv

# Load .env once at process start
load_dotenv()

# ---- CLI / Pipeline imports ----
import typer
from .config import DEFAULT_CONFIG
from .schemas import InputPayload
from .pipeline import run_pipeline
from .interactive import interactive_loop


app = typer.Typer(add_completion=False, help="Research Thinking Agent (RTA)")


@app.callback(invoke_without_command=True)
def main(
    query: str = typer.Argument(None),
    context: str = typer.Option("", help="Optional context"),
):
    """
    Research Thinking Agent (RTA)
    """
    if query is None:
        interactive_loop()
        return

    user_input = InputPayload(query=query, context=context or None)
    run_id, run_dir = run_pipeline(DEFAULT_CONFIG, user_input)
    typer.echo(f"[OK] run_id={run_id}")
    typer.echo(f"[OK] outputs at: {run_dir}")


if __name__ == "__main__":
    app()
