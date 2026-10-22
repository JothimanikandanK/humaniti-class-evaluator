"""Check (and optionally download) prerequisites. Usage: python scripts/check_setup.py [--download]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Settings
from src.preflight import run_preflight

if __name__ == "__main__":
    checks = run_preflight(Settings.from_env(), download="--download" in sys.argv)
    for c in checks:
        print(f"[{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    sys.exit(0 if all(c.ok for c in checks if c.blocking) else 1)
