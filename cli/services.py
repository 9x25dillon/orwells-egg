"""Service management commands"""

import subprocess
import signal
import os
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from cli.config import get_config
from cli.client import KGirlClient

console = Console()
services_app = typer.Typer()

# Service process tracking
SERVICE_PIDS = {}


def is_process_running(pid: int) -> bool:
    """Check if a process is running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@services_app.command("start")
def start_service(
    service: str = typer.Argument(..., help="Service name (api, skin-os, all)"),
    port: Optional[int] = typer.Option(None, help="Override default port"),
    background: bool = typer.Option(True, help="Run in background"),
):
    """Start a service"""
    config = get_config()

    if service == "all":
        start_service("api", background=background)
        time.sleep(2)  # Give main API time to start
        start_service("skin-os", background=background)
        return

    if service == "api":
        port = port or 8000
        console.print(f"[cyan]Starting Main API on port {port}...[/cyan]")

        cmd = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)]
        if not background:
            cmd.append("--reload")

        if background:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/home/user/orwells-egg"
            )
            SERVICE_PIDS["api"] = process.pid
            console.print(f"[green]✓ Main API started (PID: {process.pid})[/green]")
        else:
            try:
                subprocess.run(cmd, cwd="/home/user/orwells-egg")
            except KeyboardInterrupt:
                console.print("\n[yellow]Service stopped[/yellow]")

    elif service == "skin-os":
        port = port or 8001
        console.print(f"[cyan]Starting skin-OS API on port {port}...[/cyan]")

        cmd = [
            "uvicorn",
            "apps.api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port)
        ]

        if background:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/home/user/orwells-egg/skin-os"
            )
            SERVICE_PIDS["skin-os"] = process.pid
            console.print(f"[green]✓ skin-OS API started (PID: {process.pid})[/green]")
        else:
            try:
                subprocess.run(cmd, cwd="/home/user/orwells-egg/skin-os")
            except KeyboardInterrupt:
                console.print("\n[yellow]Service stopped[/yellow]")

    else:
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print("Available services: api, skin-os, all")


@services_app.command("stop")
def stop_service(
    service: str = typer.Argument(..., help="Service name (api, skin-os, all)"),
):
    """Stop a service"""
    if service == "all":
        stop_service("api")
        stop_service("skin-os")
        return

    if service in SERVICE_PIDS:
        pid = SERVICE_PIDS[service]
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]✓ {service} stopped[/green]")
            del SERVICE_PIDS[service]
        except ProcessLookupError:
            console.print(f"[yellow]Process not found (PID: {pid})[/yellow]")
            del SERVICE_PIDS[service]
    else:
        # Try to find and kill by port
        console.print(f"[yellow]No tracked PID for {service}, searching by port...[/yellow]")

        port = 8000 if service == "api" else 8001
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pid = int(result.stdout.strip().split()[0])
                os.kill(pid, signal.SIGTERM)
                console.print(f"[green]✓ {service} stopped (PID: {pid})[/green]")
            else:
                console.print(f"[yellow]No process found on port {port}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error stopping service: {e}[/red]")


@services_app.command("status")
def service_status():
    """Check status of all services"""
    config = get_config()
    client = KGirlClient(config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Check Main API
    api_status = client.check_health()
    table.add_row(
        "Main API",
        "✓ Running" if api_status else "✗ Stopped",
        config.get("main_api_url", "http://localhost:8000")
    )

    # Check skin-OS
    skin_status = client.check_skin_os_health()
    table.add_row(
        "skin-OS API",
        "✓ Running" if skin_status else "✗ Stopped",
        config.get("skin_os_url", "http://localhost:8001")
    )

    # Check Database
    db_status = client.check_database()
    table.add_row(
        "Database",
        "✓ Connected" if db_status else "✗ Disconnected",
        "PostgreSQL"
    )

    console.print(table)


@services_app.command("restart")
def restart_service(
    service: str = typer.Argument(..., help="Service name (api, skin-os, all)"),
    port: Optional[int] = typer.Option(None, help="Override default port"),
):
    """Restart a service"""
    console.print(f"[cyan]Restarting {service}...[/cyan]")
    stop_service(service)
    time.sleep(1)
    start_service(service, port=port, background=True)


@services_app.command("logs")
def show_logs(
    service: str = typer.Argument(..., help="Service name (api, skin-os)"),
    follow: bool = typer.Option(False, "--follow", help="Follow log output"),
    lines: int = typer.Option(50, help="Number of lines to show"),
):
    """Show service logs"""
    log_files = {
        "api": "/tmp/kgirl-api.log",
        "skin-os": "/tmp/kgirl-skin-os.log",
    }

    if service not in log_files:
        console.print(f"[red]Unknown service: {service}[/red]")
        return

    log_file = log_files[service]

    if not Path(log_file).exists():
        console.print(f"[yellow]No log file found: {log_file}[/yellow]")
        console.print("[dim]Logs may be written to stdout if service is running in foreground[/dim]")
        return

    try:
        if follow:
            subprocess.run(["tail", "-f", log_file])
        else:
            subprocess.run(["tail", f"-{lines}", log_file])
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped following logs[/yellow]")
