"""Main CLI application entry point"""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from cli.services import services_app
from cli.ai_assistant import ai_app
from cli.data import data_app
from cli.ml import ml_app
from cli.repl import start_repl
from cli.config import config_app, get_config

console = Console()
app = typer.Typer(
    name="kgirl",
    help="AI-Powered IDE for Orwell's Egg ML/Data Platform",
    add_completion=True,
)

# Register sub-commands
app.add_typer(services_app, name="service", help="Manage services (start/stop/status)")
app.add_typer(ai_app, name="ai", help="AI-powered assistance and code generation")
app.add_typer(data_app, name="data", help="Data pipeline and packet processing")
app.add_typer(ml_app, name="ml", help="Machine learning training and models")
app.add_typer(config_app, name="config", help="Configuration management")


@app.command()
def status():
    """Show overall system status"""
    from cli.client import KGirlClient

    console.print("[bold cyan]kgirl System Status[/bold cyan]\n")

    config = get_config()
    client = KGirlClient(config)

    # Create status table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("URL", style="dim")

    # Check main API
    main_status = client.check_health()
    table.add_row(
        "Main API",
        "✓ Online" if main_status else "✗ Offline",
        config.get("main_api_url", "http://localhost:8000")
    )

    # Check skin-OS API
    skin_status = client.check_skin_os_health()
    table.add_row(
        "skin-OS API",
        "✓ Online" if skin_status else "✗ Offline",
        config.get("skin_os_url", "http://localhost:8001")
    )

    # Check database
    db_status = client.check_database()
    table.add_row(
        "Database",
        "✓ Connected" if db_status else "✗ Disconnected",
        config.get("database_url", "postgresql://localhost:5432/limps")
    )

    console.print(table)
    console.print()


@app.command()
def repl():
    """Start interactive REPL mode"""
    console.print("[bold green]Starting kgirl Interactive Mode[/bold green]")
    console.print("[dim]Type 'help' for available commands, 'exit' to quit[/dim]\n")
    start_repl()


@app.command()
def version():
    """Show version information"""
    from cli import __version__
    console.print(f"[bold]kgirl[/bold] version [cyan]{__version__}[/cyan]")
    console.print("[dim]AI-Powered IDE for Orwell's Egg ML/Data Platform[/dim]")


@app.callback()
def main():
    """
    kgirl - AI-Powered IDE Command Line Interface

    Comprehensive CLI for managing the Orwell's Egg ML/data orchestration platform.
    Use sub-commands for specific functionality or run 'kgirl repl' for interactive mode.
    """
    pass


if __name__ == "__main__":
    app()
