#!/usr/bin/env python3
"""Convenience alias pointing to tooling.texture_adjust."""
import sys
from pathlib import Path

# Support running both directly and as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tooling.texture_adjust import main

if __name__ == "__main__":
    sys.exit(main())
