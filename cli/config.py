"""Configuration management for kgirl CLI"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import typer
from rich.console import Console
from rich.table import Table

console = Console()
config_app = typer.Typer()

CONFIG_DIR = Path.home() / ".kgirl"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "main_api_url": "http://localhost:8000",
    "skin_os_url": "http://localhost:8001",
    "database_url": os.getenv("DATABASE_URL", "postgresql+psycopg://limps:limps@localhost:5432/limps"),
    "julia_base": os.getenv("JULIA_BASE", "http://localhost:9000"),
    "choppy_base": os.getenv("CHOPPY_BASE", "http://localhost:9100"),
    "api_key": os.getenv("API_KEY", ""),
    "ai_model": "claude-3-5-sonnet-20241022",
    "editor": os.getenv("EDITOR", "vim"),
    "theme": "monokai",
}


def ensure_config_dir():
    """Ensure config directory exists"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    ensure_config_dir()

    if not CONFIG_FILE.exists():
        # Create default config
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    ensure_config_dir()

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")


def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return load_config()


@config_app.command("show")
def show_config():
    """Show current configuration"""
    config = load_config()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        # Mask API key
        if key == "api_key" and value:
            value = value[:8] + "..." if len(value) > 8 else "***"
        table.add_row(key, str(value))

    console.print(table)
    console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")


@config_app.command("set")
def set_config(key: str, value: str):
    """Set a configuration value"""
    config = load_config()

    # Type conversion for known settings
    if key in ["main_api_url", "skin_os_url", "database_url", "julia_base", "choppy_base", "api_key", "ai_model", "editor", "theme"]:
        config[key] = value
        save_config(config)
        console.print(f"[green]✓[/green] Set {key} = {value}")
    else:
        console.print(f"[yellow]Warning: Unknown config key '{key}'[/yellow]")
        console.print("Known keys: " + ", ".join(DEFAULT_CONFIG.keys()))


@config_app.command("reset")
def reset_config():
    """Reset configuration to defaults"""
    if typer.confirm("Are you sure you want to reset all configuration to defaults?"):
        save_config(DEFAULT_CONFIG)
        console.print("[green]✓ Configuration reset to defaults[/green]")
    else:
        console.print("Cancelled")


@config_app.command("path")
def show_config_path():
    """Show configuration file path"""
    console.print(f"Config file: [cyan]{CONFIG_FILE}[/cyan]")
