"""
User Interface Utilities.
Provides rich console outputs, spinners, and status indicators.
File: src/rta/utils/ui.py
"""

from contextlib import contextmanager
from typing import Generator
import logging

try:
    from rich.console import Console
    from rich.logging import RichHandler
    # Initialize a global console instance
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

@contextmanager
def spinner(text: str = "Processing...") -> Generator[None, None, None]:
    """
    Context manager that displays a spinning loading animation.
    
    Usage:
        with spinner("Searching..."):
            do_heavy_work()
    """
    if HAS_RICH and console:
        # 'dots' is the classic spinner. 'aesthetic' or 'earth' are also cool options.
        with console.status(f"[bold green]{text}", spinner="dots"):
            yield
    else:
        # Fallback for environments without rich or dumb terminals
        logging.info(f"[*] {text}")
        yield
        logging.info("[*] Done.")

def print_header(title: str, subtitle: str = ""):
    """Prints a styled header."""
    if HAS_RICH and console:
        console.rule(f"[bold blue]{title}")
        if subtitle:
            console.print(f"[dim]{subtitle}[/dim]", justify="center")
        console.print()
    else:
        print(f"=== {title} ===")
        print(subtitle)
        print("-" * 40)