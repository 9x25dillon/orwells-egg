# kgirl - AI-Powered IDE Command Line Interface

kgirl is a comprehensive command-line interface for managing and interacting with the Orwell's Egg ML/data orchestration platform. It provides an AI-powered development environment with intuitive commands for service management, data processing, machine learning, and more.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Command Reference](#command-reference)
- [Interactive Mode](#interactive-mode)
- [Examples](#examples)
- [Architecture](#architecture)

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Make kgirl executable:**

```bash
chmod +x kgirl.py
```

3. **Optional: Create a symlink for easy access:**

```bash
sudo ln -s $(pwd)/kgirl.py /usr/local/bin/kgirl
```

## Quick Start

### Check System Status

```bash
python kgirl.py status
```

### Start Services

```bash
# Start all services
python kgirl.py service start all

# Start specific service
python kgirl.py service start api
python kgirl.py service start skin-os
```

### Interactive Mode

```bash
python kgirl.py repl
```

### Process Data

```bash
python kgirl.py data process "Hello, World!"
```

### AI Assistance

```bash
python kgirl.py ai ask "How do I optimize this query?"
python kgirl.py ai analyze myfile.py
```

## Configuration

kgirl stores configuration in `~/.kgirl/config.json`. Manage settings with:

### View Configuration

```bash
python kgirl.py config show
```

### Set Configuration Values

```bash
# API endpoints
python kgirl.py config set main_api_url http://localhost:8000
python kgirl.py config set skin_os_url http://localhost:8001

# Database
python kgirl.py config set database_url postgresql://user:pass@localhost:5432/db

# External services
python kgirl.py config set julia_base http://localhost:9000
python kgirl.py config set choppy_base http://localhost:9100

# AI settings
python kgirl.py config set ai_model claude-3-5-sonnet-20241022
python kgirl.py config set ai_api_key YOUR_API_KEY

# Editor preference
python kgirl.py config set editor vim
```

### Reset Configuration

```bash
python kgirl.py config reset
```

## Command Reference

### Global Commands

#### `status`
Show overall system status including all services and database connectivity.

```bash
python kgirl.py status
```

#### `version`
Display kgirl version information.

```bash
python kgirl.py version
```

#### `repl`
Start interactive REPL mode.

```bash
python kgirl.py repl
```

---

### Service Management (`service`)

Manage the lifecycle of Orwell's Egg services.

#### `service start`
Start a service or all services.

```bash
# Start all services
python kgirl.py service start all

# Start main API
python kgirl.py service start api

# Start skin-OS
python kgirl.py service start skin-os

# Start with custom port
python kgirl.py service start api --port 9000

# Start in foreground (with live reload)
python kgirl.py service start api --no-background
```

#### `service stop`
Stop a running service.

```bash
python kgirl.py service stop api
python kgirl.py service stop skin-os
python kgirl.py service stop all
```

#### `service status`
Check the status of all services.

```bash
python kgirl.py service status
```

#### `service restart`
Restart a service.

```bash
python kgirl.py service restart api
python kgirl.py service restart skin-os --port 8002
```

#### `service logs`
View service logs.

```bash
# Show last 50 lines (default)
python kgirl.py service logs api

# Follow logs in real-time
python kgirl.py service logs api -f

# Show specific number of lines
python kgirl.py service logs skin-os -n 100
```

---

### Data Pipeline (`data`)

Interact with the skin-OS data processing pipeline.

#### `data process`
Process a data packet through the pipeline layers.

```bash
# Basic processing
python kgirl.py data process "Hello, World!"

# With hydration level (0-3)
python kgirl.py data process "Data content" --hydration 2

# With pigment (provenance marker)
python kgirl.py data process "Sensitive data" --pigment "high-security"
```

#### `data enrich`
Enrich an existing packet with additional data.

```bash
python kgirl.py data enrich <packet_id> '{"key": "value", "metadata": "info"}'
```

#### `data query`
Generate and execute a data selection query.

```bash
# Basic query
python kgirl.py data query users

# With limit
python kgirl.py data query users --limit 50

# With filters (JSON)
python kgirl.py data query users --filters '{"age": {"gt": 18}}'
```

#### `data metrics`
Display skin-OS pipeline metrics (Prometheus format).

```bash
python kgirl.py data metrics
```

#### `data pipeline`
Show the data pipeline architecture diagram.

```bash
python kgirl.py data pipeline
```

#### `data inspect`
Inspect a packet's processing history.

```bash
python kgirl.py data inspect <packet_id>
```

#### `data batch`
Batch process multiple packets from a file.

```bash
# Process JSONL file
python kgirl.py data batch input.jsonl

# With output file
python kgirl.py data batch input.jsonl --output-file results.json
```

**Input file format (JSONL):**
```json
{"content": "First packet", "hydration": 1}
{"content": "Second packet", "hydration": 2}
{"content": "Third packet", "pigment": "test"}
```

---

### Machine Learning (`ml`)

Manage model training, evaluation, and deployment.

#### `ml train`
Start model training.

```bash
# Basic training
python kgirl.py ml train my_model

# With configuration
python kgirl.py ml train my_model --config-file train_config.json

# With parameters
python kgirl.py ml train my_model --epochs 100 --batch-size 64
```

#### `ml status`
Check training status.

```bash
# All models
python kgirl.py ml status

# Specific model
python kgirl.py ml status --model-name my_model
```

#### `ml snapshot`
Create an RFV snapshot of a trained model.

```bash
python kgirl.py ml snapshot my_model /path/to/checkpoint.pt

# With description
python kgirl.py ml snapshot my_model checkpoint.pt --description "Best model v1"
```

#### `ml evaluate`
Evaluate a trained model.

```bash
# Basic evaluation
python kgirl.py ml evaluate my_model test_data.json

# Custom metrics
python kgirl.py ml evaluate my_model test_data.json --metrics "accuracy,precision,recall,f1"
```

#### `ml list`
List available models.

```bash
# All models
python kgirl.py ml list

# Filter by status
python kgirl.py ml list --status training
```

#### `ml logs`
Show training logs for a model.

```bash
# Last 50 lines
python kgirl.py ml logs my_model

# Follow logs
python kgirl.py ml logs my_model -f

# Custom line count
python kgirl.py ml logs my_model --lines 200
```

#### `ml coach`
Show entropy-aware coach state (learning rate scheduler).

```bash
# All models
python kgirl.py ml coach

# Specific model
python kgirl.py ml coach --model-name my_model
```

#### `ml architecture`
Display the ML2 architecture diagram.

```bash
python kgirl.py ml architecture
```

#### `ml job`
Manage training jobs in the queue.

```bash
# Lease next job
python kgirl.py ml job lease

# List all jobs
python kgirl.py ml job list

# Mark job complete
python kgirl.py ml job complete --job-id job_123
```

---

### AI Assistant (`ai`)

AI-powered code assistance and analysis.

#### `ai ask`
Ask the AI assistant a question.

```bash
# Simple question
python kgirl.py ai ask "How do I optimize database queries?"

# With context from file
python kgirl.py ai ask "Explain this code" --context myfile.py
```

#### `ai analyze`
Analyze code quality and patterns.

```bash
# Basic analysis
python kgirl.py ai analyze myfile.py

# With detailed metrics
python kgirl.py ai analyze myfile.py --metrics
```

The analyzer checks for:
- Code metrics (LOC, functions, classes)
- Complexity issues
- Security vulnerabilities (eval, exec, shell injection)
- TODO/FIXME comments

#### `ai generate`
Generate code from a description.

```bash
# Generate to stdout
python kgirl.py ai generate "A function to validate email addresses"

# Specify language
python kgirl.py ai generate "Quick sort algorithm" --language python

# Save to file
python kgirl.py ai generate "REST API client" --output client.py
```

#### `ai explain`
Explain complex code.

```bash
# Explain entire file
python kgirl.py ai explain complex_module.py

# Explain specific function
python kgirl.py ai explain myfile.py --function calculate_entropy
```

#### `ai refactor`
Get refactoring suggestions.

```bash
# Get suggestions
python kgirl.py ai refactor legacy_code.py

# Apply suggestions automatically
python kgirl.py ai refactor legacy_code.py --apply
```

#### `ai chat`
Start an interactive AI chat session.

```bash
python kgirl.py ai chat
```

In chat mode:
- `exit`, `quit`, `q` - Exit chat
- `help` - Show commands
- `clear` - Clear conversation history
- `save <file>` - Save conversation to file

---

### Configuration (`config`)

Manage kgirl configuration.

#### `config show`
Display all configuration settings.

```bash
python kgirl.py config show
```

#### `config set`
Set a configuration value.

```bash
python kgirl.py config set <key> <value>
```

#### `config reset`
Reset all configuration to defaults.

```bash
python kgirl.py config reset
```

#### `config path`
Show configuration file path.

```bash
python kgirl.py config path
```

## Interactive Mode

The REPL (Read-Eval-Print Loop) provides an interactive shell for kgirl commands.

### Starting REPL

```bash
python kgirl.py repl
```

### REPL Commands

All commands can be used without the `kgirl.py` prefix:

```
kgirl> status                    # System status
kgirl> process Hello, World!     # Process data
kgirl> query users 10            # Query data
kgirl> train my_model            # Start training
kgirl> jobs                      # List jobs
kgirl> config api_url            # View config
kgirl> ai What is entropy?       # Ask AI
kgirl> metrics                   # Show metrics
kgirl> history                   # Command history
kgirl> clear                     # Clear screen
kgirl> exit                      # Exit REPL
```

## Examples

### Example 1: Complete Workflow

```bash
# 1. Configure system
python kgirl.py config set database_url postgresql://localhost:5432/mydb
python kgirl.py config set julia_base http://localhost:9000

# 2. Start services
python kgirl.py service start all

# 3. Check status
python kgirl.py status

# 4. Process some data
python kgirl.py data process "Training data example" --hydration 2

# 5. Start training
python kgirl.py ml train my_model --epochs 50 --batch-size 32

# 6. Monitor progress
python kgirl.py ml status
python kgirl.py ml logs my_model -f
```

### Example 2: Batch Data Processing

```bash
# Create input file
cat > packets.jsonl << EOF
{"content": "First data point", "hydration": 1}
{"content": "Second data point", "hydration": 2}
{"content": "Third data point", "hydration": 1}
EOF

# Process batch
python kgirl.py data batch packets.jsonl --output-file results.json

# Analyze results
python kgirl.py ai analyze results.json
```

### Example 3: AI-Assisted Development

```bash
# Analyze existing code
python kgirl.py ai analyze app.py --metrics

# Get refactoring suggestions
python kgirl.py ai refactor app.py

# Generate new functionality
python kgirl.py ai generate "A REST endpoint for user authentication" --output auth.py

# Ask questions
python kgirl.py ai ask "How can I improve the performance of this entropy calculation?" --context entropy_adapter.py
```

### Example 4: Interactive Development Session

```bash
python kgirl.py repl

kgirl> status
kgirl> process Testing the pipeline
kgirl> ai How can I optimize this?
kgirl> query aalc_queue 10
kgirl> train experiment_v1
kgirl> history
kgirl> exit
```

## Architecture

### CLI Structure

```
kgirl.py                 # Main entry point
cli/
├── __init__.py         # Package initialization
├── main.py             # Main CLI application
├── config.py           # Configuration management
├── client.py           # API client
├── services.py         # Service management commands
├── ai_assistant.py     # AI-powered features
├── data.py             # Data pipeline commands
├── ml.py               # ML training commands
└── repl.py             # Interactive REPL
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    kgirl CLI                             │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ Services │   Data   │    ML    │    AI    │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
    ┌─────────────┐  ┌──────────┐  ┌─────────┐
    │  Main API   │  │ skin-OS  │  │Database │
    │  :8000      │  │ :8001    │  │ :5432   │
    └─────────────┘  └──────────┘  └─────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
         ┌──────────┐ ┌─────────┐ ┌────────┐
         │  Julia   │ │ Choppy  │ │Workers │
         │ :9000    │ │ :9100   │ │        │
         └──────────┘ └─────────┘ └────────┘
```

### Data Flow

1. **User Input** → kgirl CLI
2. **CLI** → API Client (httpx)
3. **API Client** → Services (Main API / skin-OS)
4. **Services** → Backend (Database, Julia, Workers)
5. **Response** → CLI → Rich formatting → User

## Troubleshooting

### Services won't start

```bash
# Check if ports are in use
lsof -i :8000
lsof -i :8001

# Check service logs
python kgirl.py service logs api
python kgirl.py service logs skin-os
```

### Database connection issues

```bash
# Verify database URL
python kgirl.py config show

# Test connection
python kgirl.py status

# Update database URL
python kgirl.py config set database_url postgresql://user:pass@host:port/db
```

### AI features not working

```bash
# Configure AI provider
python kgirl.py config set ai_api_key YOUR_API_KEY
python kgirl.py config set ai_model claude-3-5-sonnet-20241022

# Test AI
python kgirl.py ai ask "Hello, can you help me?"
```

## Contributing

To extend kgirl with new commands:

1. Create command module in `cli/` directory
2. Define typer app: `my_app = typer.Typer()`
3. Add commands with `@my_app.command()`
4. Register in `cli/main.py`: `app.add_typer(my_app, name="mycommand")`

## License

See project LICENSE file.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/orwells-egg/issues)
- Documentation: See `README.md` and `skin-os/README.md`
