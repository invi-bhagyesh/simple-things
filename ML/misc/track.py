#!/usr/bin/env python3
"""
Auto-update README.md when running exercises
Usage: python track.py 01_mlp.py
"""
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path

def update_readme(filename: str, status: str = "DONE"):
    readme = Path(__file__).parent.parent / "README.md"
    content = readme.read_text()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Match the row with this filename and update status + date
    pattern = rf"(\| \d+ \|[^|]+\| \[{re.escape(filename)}\][^|]+\|)\s*[^|]*\s*\|\s*[^|]*\s*\|"
    replacement = rf"\1 {status} | {today} |"
    
    new_content = re.sub(pattern, replacement, content)
    readme.write_text(new_content)
    print(f"Updated README: {filename} -> {status} ({today})")

def run_exercise(filename: str):
    """Run exercise and update README if tests pass"""
    result = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0 and "passed" in result.stdout.lower():
        update_readme(filename, "DONE")
    elif result.returncode != 0:
        update_readme(filename, "WIP")
        print("Tests failed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track.py <exercise.py>")
        print("Example: python track.py 01_mlp.py")
        sys.exit(1)
    
    run_exercise(sys.argv[1])
