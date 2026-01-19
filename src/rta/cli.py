"""
RTA Command Line Interface.
File: src/rta/cli.py
"""
from __future__ import annotations

import os

# [CRITICAL] Set these BEFORE importing any other heavy libraries
# "3" suppresses ERROR logs (like ALTS creds ignored). "2" was not enough.
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import warnings

# Filter Python level warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import typer

from .config import RTAConfig
from .shell import RTAShell

app = typer.Typer(add_completion=False)

@app.command()
def main() -> None:
    """Launch RTA interactive shell."""
    try:
        # Initialize and run the shell
        RTAShell(RTAConfig()).run()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"[Fatal Error] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()