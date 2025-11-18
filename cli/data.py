"""Data pipeline interaction commands"""

import typer
import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from rich.panel import Panel

from cli.config import get_config
from cli.client import KGirlClient

console = Console()
data_app = typer.Typer()


@data_app.command("process")
def process_packet(
    content: str = typer.Argument(..., help="Content to process"),
    hydration: int = typer.Option(0, help="Hydration level (0-3)"),
    pigment: str = typer.Option("default", help="Pigment (provenance marker)"),
):
    """Process a data packet through the skin-OS pipeline"""
    config = get_config()
    client = KGirlClient(config)

    console.print("[cyan]Processing packet through skin-OS layers...[/cyan]\n")

    packet_data = {
        "content": content,
        "hydration": hydration,
        "pigment": pigment,
        "metadata": {}
    }

    result = client.process_packet(packet_data)

    if result:
        console.print("[bold green]✓ Packet processed successfully[/bold green]\n")

        # Show results
        json_data = JSON(json.dumps(result, indent=2))
        console.print(Panel(json_data, title="Processing Results", expand=False))

        # Show layer processing details if available
        if "layers" in result:
            console.print("\n[bold]Layer Processing:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Layer", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim")

            for layer_name, layer_result in result["layers"].items():
                status = "✓ Passed" if layer_result.get("passed", True) else "✗ Failed"
                details = layer_result.get("message", "")
                table.add_row(layer_name, status, details)

            console.print(table)
    else:
        console.print("[red]✗ Failed to process packet[/red]")


@data_app.command("enrich")
def enrich_packet(
    packet_id: str = typer.Argument(..., help="Packet ID to enrich"),
    data: str = typer.Argument(..., help="Enrichment data (JSON string)"),
):
    """Enrich a packet with additional data"""
    config = get_config()
    client = KGirlClient(config)

    try:
        enrichment_data = json.loads(data)
    except json.JSONDecodeError:
        console.print("[red]Error: Invalid JSON data[/red]")
        return

    console.print(f"[cyan]Enriching packet {packet_id}...[/cyan]")

    success = client.enrich_packet(packet_id, enrichment_data)

    if success:
        console.print("[green]✓ Packet enriched successfully[/green]")
    else:
        console.print("[red]✗ Failed to enrich packet[/red]")


@data_app.command("query")
def query_data(
    table: str = typer.Argument(..., help="Table to query"),
    filters: Optional[str] = typer.Option(None, help="Filter conditions (JSON)"),
    limit: int = typer.Option(100, help="Result limit"),
):
    """Generate and execute a data selection query"""
    config = get_config()
    client = KGirlClient(config)

    console.print(f"[cyan]Querying table: {table}...[/cyan]\n")

    query_params = {
        "table": table,
        "limit": limit
    }

    if filters:
        try:
            query_params["filters"] = json.loads(filters)
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid filter JSON[/red]")
            return

    result = client.select_data(query_params)

    if result:
        console.print("[bold green]✓ Query generated[/bold green]\n")

        # Show generated SQL
        if "sql" in result:
            from rich.syntax import Syntax
            syntax = Syntax(result["sql"], "sql", theme="monokai")
            console.print(Panel(syntax, title="Generated SQL", expand=False))

        # Show entropy if available
        if "entropy" in result:
            console.print(f"\n[bold]Query Entropy:[/bold] {result['entropy']:.4f}")

        # Show results if available
        if "results" in result:
            console.print(f"\n[bold]Results:[/bold] {len(result['results'])} rows")
    else:
        console.print("[red]✗ Query failed[/red]")


@data_app.command("metrics")
def show_metrics():
    """Show skin-OS pipeline metrics"""
    config = get_config()
    client = KGirlClient(config)

    console.print("[cyan]Fetching pipeline metrics...[/cyan]\n")

    metrics = client.get_metrics()

    if metrics:
        # Parse Prometheus metrics
        console.print("[bold]Pipeline Metrics:[/bold]\n")

        lines = metrics.split('\n')
        for line in lines:
            if line and not line.startswith('#'):
                console.print(f"  {line}")
    else:
        console.print("[red]✗ Failed to fetch metrics[/red]")


@data_app.command("pipeline")
def show_pipeline():
    """Show data pipeline architecture"""
    pipeline_diagram = """
[bold cyan]skin-OS Data Pipeline Architecture[/bold cyan]

┌─────────────────────────────────────────────────────────────┐
│                     Data Packet Input                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: stratum_corneum (Surface Security)                │
│  - SQL injection detection                                   │
│  - XSS filtering                                             │
│  - Input validation                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: stratum_granulosum (Feature Extraction)           │
│  - Content hashing                                           │
│  - Length metrics                                            │
│  - Feature computation                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: stratum_spinosum (Provenance)                     │
│  - Origin tracking                                           │
│  - Pigment verification                                      │
│  - Metadata enrichment                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: stratum_basale (Job Spawning)                     │
│  - Enrichment job creation                                   │
│  - Queue insertion                                           │
│  - Async processing                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               ViscoElasticQueue (visco layer)                │
│  - Dynamic capacity adjustment                               │
│  - Load-responsive queueing                                  │
│  - Latency-aware rate limiting                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Enrichment Workers                        │
│  - Async processing                                          │
│  - External service calls                                    │
│  - Result aggregation                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Enriched Data Output                       │
└─────────────────────────────────────────────────────────────┘

[bold]Key Features:[/bold]
• Biology-inspired layered architecture
• Entropy measurement at each stage
• Visco-elastic queue dynamics
• Prometheus metrics integration
"""

    console.print(pipeline_diagram)


@data_app.command("inspect")
def inspect_packet(
    packet_id: str = typer.Argument(..., help="Packet ID to inspect"),
):
    """Inspect a packet's processing history"""
    # This would query the database for packet history
    console.print(f"[cyan]Inspecting packet: {packet_id}[/cyan]\n")

    # Placeholder - would query actual database
    console.print("[yellow]Feature not yet implemented[/yellow]")
    console.print("[dim]This will show:[/dim]")
    console.print("  - Packet content and metadata")
    console.print("  - Processing timeline")
    console.print("  - Layer-by-layer results")
    console.print("  - Enrichment history")
    console.print("  - Entropy measurements")


@data_app.command("batch")
def batch_process(
    input_file: str = typer.Argument(..., help="Input file (JSON lines)"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
):
    """Batch process multiple packets"""
    from pathlib import Path
    import json

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        return

    config = get_config()
    client = KGirlClient(config)

    console.print(f"[cyan]Batch processing from {input_file}...[/cyan]\n")

    results = []
    successful = 0
    failed = 0

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                packet_data = json.loads(line)
                result = client.process_packet(packet_data)

                if result:
                    results.append(result)
                    successful += 1
                    console.print(f"  ✓ Line {line_num} processed")
                else:
                    failed += 1
                    console.print(f"  ✗ Line {line_num} failed")
            except json.JSONDecodeError:
                console.print(f"  ⚠️  Line {line_num} - invalid JSON, skipping")
                failed += 1

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Successful: {successful}")
    console.print(f"  Failed: {failed}")

    if output_file and results:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]✓ Results written to {output_file}[/green]")
