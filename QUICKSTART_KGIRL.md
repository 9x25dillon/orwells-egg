# kgirl CLI - Quick Start Guide

Welcome to kgirl, the AI-powered IDE command-line interface for Orwell's Egg!

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Make kgirl Executable

```bash
chmod +x kgirl.py
```

### Step 3: Test Installation

```bash
./kgirl.py --version
```

You should see:
```
kgirl version 0.1.0
AI-Powered IDE for Orwell's Egg ML/Data Platform
```

## First Steps

### 1. Check System Status

```bash
./kgirl.py status
```

This shows the status of all services (Main API, skin-OS, Database).

### 2. Configure Your Environment

```bash
# View current configuration
./kgirl.py config show

# Set custom values if needed
./kgirl.py config set main_api_url http://localhost:8000
./kgirl.py config set skin_os_url http://localhost:8001
```

### 3. Start Services

```bash
# Start all services in the background
./kgirl.py service start all

# Or start individually
./kgirl.py service start api
./kgirl.py service start skin-os
```

### 4. Verify Services Are Running

```bash
./kgirl.py service status
```

You should see all services marked as "Running" ✓

## Try It Out!

### Process Your First Data Packet

```bash
./kgirl.py data process "Hello, kgirl!"
```

This sends your data through the skin-OS pipeline layers:
- stratum_corneum (security checks)
- stratum_granulosum (feature extraction)
- stratum_spinosum (provenance)
- stratum_basale (job spawning)

### Ask the AI Assistant

```bash
./kgirl.py ai ask "What is entropy in machine learning?"
```

### Analyze Code

```bash
./kgirl.py ai analyze app.py
```

This performs:
- Code metrics analysis
- Security vulnerability checks
- Complexity assessment
- Best practices review

### Interactive Mode

For a more interactive experience:

```bash
./kgirl.py repl
```

Then try:
```
kgirl> status
kgirl> process Testing the system
kgirl> ai What can you help me with?
kgirl> help
kgirl> exit
```

## Common Commands Cheat Sheet

### Service Management
```bash
./kgirl.py service start all      # Start all services
./kgirl.py service stop all       # Stop all services
./kgirl.py service restart api    # Restart main API
./kgirl.py service logs api -f    # Follow API logs
./kgirl.py service status         # Check service status
```

### Data Pipeline
```bash
./kgirl.py data process "content"              # Process data
./kgirl.py data query users                    # Query database
./kgirl.py data metrics                        # View metrics
./kgirl.py data pipeline                       # Show architecture
./kgirl.py data batch input.jsonl             # Batch process
```

### Machine Learning
```bash
./kgirl.py ml train my_model                   # Start training
./kgirl.py ml status                           # Check training
./kgirl.py ml logs my_model -f                 # Follow logs
./kgirl.py ml list                             # List models
./kgirl.py ml snapshot my_model checkpoint.pt  # Create snapshot
./kgirl.py ml architecture                     # Show ML architecture
```

### AI Assistant
```bash
./kgirl.py ai ask "question"      # Ask AI
./kgirl.py ai analyze file.py     # Analyze code
./kgirl.py ai generate "desc"     # Generate code
./kgirl.py ai explain file.py     # Explain code
./kgirl.py ai refactor file.py    # Get refactoring suggestions
./kgirl.py ai chat                # Interactive chat
```

### Configuration
```bash
./kgirl.py config show            # Show all settings
./kgirl.py config set key value   # Set a value
./kgirl.py config reset           # Reset to defaults
```

## Optional: Global Installation

To use `kgirl` from anywhere:

```bash
# Option 1: Create symlink
sudo ln -s $(pwd)/kgirl.py /usr/local/bin/kgirl

# Option 2: Add to PATH
echo "export PATH=\$PATH:$(pwd)" >> ~/.bashrc
source ~/.bashrc

# Now use it anywhere
kgirl status
```

## What's Next?

### Learn More
- Read the full documentation: `docs/KGIRL_CLI.md`
- Explore the main README: `README.md`
- Check skin-OS docs: `skin-os/README.md`

### Example Workflows

#### 1. Data Processing Pipeline
```bash
# Start services
./kgirl.py service start all

# Process data with different hydration levels
./kgirl.py data process "basic data" --hydration 0
./kgirl.py data process "enriched data" --hydration 2

# View metrics
./kgirl.py data metrics
```

#### 2. Machine Learning Workflow
```bash
# List available models
./kgirl.py ml list

# Start training
./kgirl.py ml train my_experiment --epochs 50 --batch-size 32

# Monitor in another terminal
./kgirl.py ml logs my_experiment -f

# Check coach state (entropy-aware scheduler)
./kgirl.py ml coach
```

#### 3. AI-Assisted Development
```bash
# Analyze existing code
./kgirl.py ai analyze complex_module.py --metrics

# Get refactoring suggestions
./kgirl.py ai refactor legacy_code.py

# Generate new code
./kgirl.py ai generate "REST API endpoint for users" --output api.py

# Interactive help
./kgirl.py ai chat
```

#### 4. Debugging and Monitoring
```bash
# Check system status
./kgirl.py status

# View service logs
./kgirl.py service logs api -f

# Inspect data processing
./kgirl.py data metrics

# Check training jobs
./kgirl.py ml job list
```

## Troubleshooting

### "Connection refused" errors
- Make sure services are running: `./kgirl.py service status`
- Start services if needed: `./kgirl.py service start all`
- Check if ports are available: `lsof -i :8000`

### Database connection issues
- Verify database is running: `psql -d limps -U limps`
- Check database URL: `./kgirl.py config show`
- Update if needed: `./kgirl.py config set database_url postgresql://...`

### AI features not responding
- Configure AI provider: `./kgirl.py config set ai_api_key YOUR_KEY`
- Currently uses placeholder responses; integrate with actual AI API

### Permission denied
- Make sure kgirl.py is executable: `chmod +x kgirl.py`
- Check Python is in PATH: `which python3`

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         kgirl CLI Interface              │
│  ┌───────────────────────────────────┐  │
│  │  Commands:                        │  │
│  │  • service  (manage services)     │  │
│  │  • data     (pipeline ops)        │  │
│  │  • ml       (training & models)   │  │
│  │  • ai       (AI assistance)       │  │
│  │  • config   (settings)            │  │
│  │  • repl     (interactive mode)    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│     API Client (httpx)                   │
└─────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐ ┌─────────┐ ┌──────────┐
│Main API│ │skin-OS  │ │Database  │
│:8000   │ │:8001    │ │:5432     │
└────────┘ └─────────┘ └──────────┘
```

## Support & Contributing

- **Documentation**: See `docs/KGIRL_CLI.md` for comprehensive guide
- **Issues**: Report bugs or request features via GitHub Issues
- **Contributing**: Add new commands by creating modules in `cli/` directory

## Next Steps

1. ✓ Install dependencies
2. ✓ Test with `./kgirl.py --version`
3. ✓ Configure with `./kgirl.py config show`
4. ✓ Start services with `./kgirl.py service start all`
5. ✓ Try commands above
6. → Read full docs in `docs/KGIRL_CLI.md`
7. → Build your workflows!

---

**Happy coding with kgirl! 🚀**
