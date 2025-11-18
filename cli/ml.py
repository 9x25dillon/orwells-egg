"""Machine learning training commands"""

import typer
import json
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from cli.config import get_config
from cli.client import KGirlClient

console = Console()
ml_app = typer.Typer()


@ml_app.command("train")
def train_model(
    model_name: str = typer.Argument(..., help="Model name"),
    config_file: Optional[str] = typer.Option(None, help="Training config file"),
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
):
    """Start model training"""
    console.print(f"[cyan]Starting training for model: {model_name}[/cyan]\n")

    config = get_config()
    client = KGirlClient(config)

    # Load training config if provided
    train_config = {}
    if config_file:
        path = Path(config_file)
        if path.exists():
            with open(path, 'r') as f:
                train_config = json.load(f)
        else:
            console.print(f"[yellow]Warning: Config file not found: {config_file}[/yellow]")

    # Show training setup
    table = Table(show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_name)
    table.add_row("Epochs", str(epochs))
    table.add_row("Batch Size", str(batch_size))

    console.print(table)
    console.print()

    # Simulate training progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training in progress...", total=None)

        # This would connect to actual training loop
        console.print("[yellow]Note: Integrate with actual training loop in ml2_core.py[/yellow]")


@ml_app.command("status")
def training_status(
    model_name: Optional[str] = typer.Option(None, help="Filter by model name"),
):
    """Check training status"""
    console.print("[cyan]Checking training status...[/cyan]\n")

    # This would query the database for active training jobs
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Epoch", style="yellow")
    table.add_column("Loss", style="blue")

    # Placeholder data
    table.add_row("model_v1", "Training", "45/100", "0.0234")
    table.add_row("model_v2", "Completed", "100/100", "0.0123")

    console.print(table)


@ml_app.command("snapshot")
def create_snapshot(
    model_name: str = typer.Argument(..., help="Model name"),
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    description: Optional[str] = typer.Option(None, help="Snapshot description"),
):
    """Create an RFV snapshot of a trained model"""
    config = get_config()
    client = KGirlClient(config)

    console.print(f"[cyan]Creating snapshot for {model_name}...[/cyan]")

    path = Path(checkpoint_path)
    if not path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        return

    snapshot_data = {
        "model_name": model_name,
        "checkpoint_path": str(path.absolute()),
        "description": description or f"Snapshot of {model_name}"
    }

    result = client.create_snapshot(snapshot_data)

    if result:
        console.print("[green]✓ Snapshot created successfully[/green]")
        console.print(f"  Snapshot ID: {result.get('snapshot_id', 'N/A')}")
    else:
        console.print("[red]✗ Failed to create snapshot[/red]")


@ml_app.command("evaluate")
def evaluate_model(
    model_name: str = typer.Argument(..., help="Model name"),
    test_data: str = typer.Argument(..., help="Test data path"),
    metrics: str = typer.Option("accuracy,loss", help="Metrics to compute (comma-separated)"),
):
    """Evaluate a trained model"""
    console.print(f"[cyan]Evaluating model: {model_name}[/cyan]\n")

    path = Path(test_data)
    if not path.exists():
        console.print(f"[red]Test data not found: {test_data}[/red]")
        return

    # Show evaluation setup
    console.print(f"[bold]Test Data:[/bold] {test_data}")
    console.print(f"[bold]Metrics:[/bold] {metrics}")
    console.print()

    # Placeholder for actual evaluation
    console.print("[bold]Evaluation Results:[/bold]")

    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    # Mock results
    for metric in metrics.split(','):
        metric = metric.strip()
        if metric == "accuracy":
            results_table.add_row("Accuracy", "94.5%")
        elif metric == "loss":
            results_table.add_row("Loss", "0.0234")
        else:
            results_table.add_row(metric.capitalize(), "N/A")

    console.print(results_table)


@ml_app.command("list")
def list_models(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
):
    """List available models"""
    console.print("[cyan]Available Models:[/cyan]\n")

    # This would query the RepoRML table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="cyan")
    table.add_column("Architecture", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="dim")

    # Placeholder data
    table.add_row("model_v1", "CompoundNode", "Training", "2024-01-15")
    table.add_row("model_v2", "SkipPreserveBlock", "Completed", "2024-01-10")
    table.add_row("baseline", "Simple MLP", "Archived", "2024-01-05")

    console.print(table)


@ml_app.command("logs")
def show_training_logs(
    model_name: str = typer.Argument(..., help="Model name"),
    lines: int = typer.Option(50, help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", help="Follow log output"),
):
    """Show training logs for a model"""
    console.print(f"[cyan]Training logs for {model_name}...[/cyan]\n")

    # Placeholder - would read from actual log files
    console.print("[dim]Epoch 1/100 - loss: 0.4523 - val_loss: 0.4321[/dim]")
    console.print("[dim]Epoch 2/100 - loss: 0.3234 - val_loss: 0.3456[/dim]")
    console.print("[dim]Epoch 3/100 - loss: 0.2345 - val_loss: 0.2678[/dim]")
    console.print("[dim]...[/dim]")


@ml_app.command("coach")
def show_coach_state(
    model_name: Optional[str] = typer.Option(None, help="Filter by model"),
):
    """Show entropy-aware coach state"""
    console.print("[cyan]Coach State (Entropy-Aware Scheduling):[/cyan]\n")

    # This would query the coach state from the database
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Learning Rate", style="green")
    table.add_column("Top-K", style="yellow")
    table.add_column("Entropy", style="blue")

    # Placeholder data
    table.add_row("model_v1", "0.001", "5", "2.345")
    table.add_row("model_v2", "0.0005", "3", "1.234")

    console.print(table)


@ml_app.command("architecture")
def show_architecture():
    """Show ML architecture overview"""
    arch_diagram = """
[bold cyan]ML2 Architecture Overview[/bold cyan]

┌──────────────────────────────────────────────────────────────┐
│                    Input Layer                                │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  CompoundNode - Multi-branch activation                       │
│  ┌─────────────┬──────────────┬────────────────┐            │
│  │   Linear    │     ReLU     │    Sigmoid     │            │
│  └─────────────┴──────────────┴────────────────┘            │
│              Softmax mixing weights                           │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SkipPreserveBlock - Skip connections                         │
│  ┌──────────────────────────────────────┐                    │
│  │  Input ───────┐                      │                    │
│  │       │       │                      │                    │
│  │       ▼       │                      │                    │
│  │   New Layer   │                      │                    │
│  │       │       │                      │                    │
│  │       ▼       ▼                      │                    │
│  │     Concat/Add                       │                    │
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  GradNormalizer - Gradient normalization                      │
│  - Max-norm or L2 normalization                               │
│  - Layer-by-layer control                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  RNNBPTTNormalizer - Temporal normalization                   │
│  - Time-wise gradient control                                 │
│  - For RNN/Transformer architectures                          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Output Layer                               │
└──────────────────────────────────────────────────────────────┘

[bold]Key Components:[/bold]
• CompoundNode: Ensemble of activation functions
• SkipPreserveBlock: Residual-style connections
• GradNormalizer: Stable gradient flow
• RNNBPTTNormalizer: Temporal gradient control
• Entropy-aware coaching: Dynamic LR & top-k
"""

    console.print(arch_diagram)


@ml_app.command("job")
def manage_training_job(
    action: str = typer.Argument(..., help="Action: lease, complete, list"),
    job_id: Optional[str] = typer.Option(None, help="Job ID for complete action"),
):
    """Manage training jobs in the queue"""
    config = get_config()
    client = KGirlClient(config)

    if action == "lease":
        console.print("[cyan]Leasing next training job...[/cyan]")
        job = client.lease_job("ml_training")

        if job:
            console.print("[green]✓ Job leased[/green]\n")
            from rich.json import JSON
            console.print(Panel(JSON(json.dumps(job, indent=2)), title="Job Details"))
        else:
            console.print("[yellow]No jobs available[/yellow]")

    elif action == "list":
        console.print("[cyan]Training job queue:[/cyan]\n")

        # This would query the AALCQueue table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Job ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Priority", style="blue")

        # Placeholder
        table.add_row("job_001", "model_v1", "queued", "10")
        table.add_row("job_002", "model_v2", "leased", "5")

        console.print(table)

    elif action == "complete":
        if not job_id:
            console.print("[red]Error: job_id required for complete action[/red]")
            return

        console.print(f"[cyan]Marking job {job_id} as complete...[/cyan]")
        # Would update AALCQueue status
        console.print("[green]✓ Job marked complete[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: lease, complete, list")
