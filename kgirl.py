#!/usr/bin/env python3
"""
kgirl - AI-Powered IDE Command Line Interface
A comprehensive CLI for managing the Orwell's Egg ML/data orchestration platform
"""

import sys
from cli.main import app

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
