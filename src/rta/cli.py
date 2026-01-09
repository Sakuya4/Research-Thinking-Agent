from __future__ import annotations

import typer

from .config import RTAConfig
from .shell import RTAShell

app = typer.Typer(add_completion=False)

@app.command()
def main() -> None:
    """Launch RTA interactive shell."""
    RTAShell(RTAConfig()).run()

if __name__ == "__main__":
    main()
