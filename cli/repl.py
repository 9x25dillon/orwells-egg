"""Interactive REPL mode for kgirl"""

import cmd
import sys
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from cli.config import get_config
from cli.client import KGirlClient

console = Console()


class KGirlREPL(cmd.Cmd):
    """Interactive REPL for kgirl IDE"""

    intro = """
╔══════════════════════════════════════════════════════════════╗
║              kgirl Interactive Mode                          ║
║      AI-Powered IDE for Orwell's Egg Platform                ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' or '?' for available commands
Type 'exit' or press Ctrl-D to quit
"""

    prompt = "kgirl> "

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.client = KGirlClient(self.config)
        self.history = []

    def do_status(self, arg):
        """Show system status"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")

        # Check services
        api_status = self.client.check_health()
        skin_status = self.client.check_skin_os_health()
        db_status = self.client.check_database()

        table.add_row("Main API", "✓ Online" if api_status else "✗ Offline")
        table.add_row("skin-OS", "✓ Online" if skin_status else "✗ Offline")
        table.add_row("Database", "✓ Connected" if db_status else "✗ Disconnected")

        console.print(table)

    def do_process(self, arg):
        """Process a data packet
        Usage: process <content>"""
        if not arg:
            console.print("[yellow]Usage: process <content>[/yellow]")
            return

        console.print(f"[cyan]Processing: {arg}[/cyan]")

        packet_data = {
            "content": arg,
            "hydration": 0,
            "pigment": "repl",
            "metadata": {}
        }

        result = self.client.process_packet(packet_data)

        if result:
            console.print("[green]✓ Processed successfully[/green]")
            import json
            from rich.json import JSON
            console.print(JSON(json.dumps(result, indent=2)))
        else:
            console.print("[red]✗ Processing failed[/red]")

    def do_query(self, arg):
        """Execute a data query
        Usage: query <table> [limit]"""
        parts = arg.split()
        if not parts:
            console.print("[yellow]Usage: query <table> [limit][/yellow]")
            return

        table = parts[0]
        limit = int(parts[1]) if len(parts) > 1 else 100

        query_params = {
            "table": table,
            "limit": limit
        }

        result = self.client.select_data(query_params)

        if result:
            console.print("[green]✓ Query executed[/green]")
            if "sql" in result:
                syntax = Syntax(result["sql"], "sql", theme="monokai")
                console.print(Panel(syntax, title="Generated SQL"))
        else:
            console.print("[red]✗ Query failed[/red]")

    def do_train(self, arg):
        """Start model training
        Usage: train <model_name>"""
        if not arg:
            console.print("[yellow]Usage: train <model_name>[/yellow]")
            return

        console.print(f"[cyan]Starting training for {arg}...[/cyan]")
        console.print("[yellow]Training job queued. Use 'jobs' to check status.[/yellow]")

    def do_jobs(self, arg):
        """List training jobs"""
        console.print("[cyan]Training jobs:[/cyan]\n")

        job = self.client.lease_job("ml_training")

        if job:
            import json
            from rich.json import JSON
            console.print(JSON(json.dumps(job, indent=2)))
        else:
            console.print("[dim]No jobs in queue[/dim]")

    def do_config(self, arg):
        """Show or set configuration
        Usage: config [key] [value]"""
        parts = arg.split(maxsplit=1)

        if not parts:
            # Show all config
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            for key, value in self.config.items():
                if key == "api_key" and value:
                    value = "***"
                table.add_row(key, str(value))

            console.print(table)
        elif len(parts) == 1:
            # Show specific config
            key = parts[0]
            if key in self.config:
                console.print(f"{key} = {self.config[key]}")
            else:
                console.print(f"[yellow]Unknown config key: {key}[/yellow]")
        else:
            # Set config
            key, value = parts
            from cli.config import load_config, save_config
            config = load_config()
            config[key] = value
            save_config(config)
            self.config = config
            console.print(f"[green]✓ Set {key} = {value}[/green]")

    def do_ai(self, arg):
        """Ask the AI assistant
        Usage: ai <question>"""
        if not arg:
            console.print("[yellow]Usage: ai <question>[/yellow]")
            return

        console.print(f"[bold cyan]Question:[/bold cyan] {arg}\n")

        # Placeholder AI response
        response = f"""I'm an AI assistant for kgirl. You asked: "{arg}"

To enable full AI capabilities, configure your AI provider:
  config ai_api_key YOUR_API_KEY

I can help with:
- Code analysis and review
- Bug detection
- Refactoring suggestions
- Documentation generation
- Test generation
"""

        console.print(response)

    def do_metrics(self, arg):
        """Show pipeline metrics"""
        metrics = self.client.get_metrics()

        if metrics:
            console.print("[bold]Pipeline Metrics:[/bold]\n")
            console.print(metrics)
        else:
            console.print("[red]✗ Failed to fetch metrics[/red]")

    def do_history(self, arg):
        """Show command history"""
        if not self.history:
            console.print("[dim]No commands in history[/dim]")
            return

        for i, cmd in enumerate(self.history, 1):
            console.print(f"  {i}. {cmd}")

    def do_clear(self, arg):
        """Clear the screen"""
        console.clear()

    def do_exit(self, arg):
        """Exit the REPL"""
        console.print("[cyan]Goodbye![/cyan]")
        return True

    def do_quit(self, arg):
        """Exit the REPL"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Handle Ctrl-D"""
        console.print()
        return self.do_exit(arg)

    def precmd(self, line):
        """Store command in history"""
        if line.strip() and not line.startswith('history'):
            self.history.append(line)
        return line

    def emptyline(self):
        """Do nothing on empty line"""
        pass

    def default(self, line):
        """Handle unknown commands"""
        console.print(f"[yellow]Unknown command: {line}[/yellow]")
        console.print("[dim]Type 'help' for available commands[/dim]")


def start_repl():
    """Start the interactive REPL"""
    try:
        repl = KGirlREPL()
        repl.cmdloop()
    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    start_repl()
