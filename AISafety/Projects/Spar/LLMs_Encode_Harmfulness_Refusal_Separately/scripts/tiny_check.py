#!/usr/bin/env python3
"""
Tiny Check Script

Quick sanity check for refusal labeling in inference outputs.
Prints counts of refused vs accepted and shows example entries.

Usage:
    python scripts/tiny_check.py RESULTS_FILE
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/tiny_check.py <results_file.jsonl>")
        sys.exit(1)
    
    fn = sys.argv[1]
    
    # Load items
    items = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    # Count refused vs accepted
    counts = {"refused": 0, "accepted": 0}
    for item in items:
        if item.get('refused', False):
            counts['refused'] += 1
        else:
            counts['accepted'] += 1
    
    print(f"\n{'='*60}")
    print(f"File: {fn}")
    print(f"Total items: {len(items)}")
    print(f"Refused: {counts['refused']} ({100*counts['refused']/len(items):.1f}%)")
    print(f"Accepted: {counts['accepted']} ({100*counts['accepted']/len(items):.1f}%)")
    print(f"{'='*60}\n")
    
    # Print example entries
    print("Sample entries:")
    print("-" * 60)
    for i, item in enumerate(items[:5]):
        prompt = item.get('prompt', '')[:100]
        if len(item.get('prompt', '')) > 100:
            prompt += "..."
        refused_str = "REFUSED" if item.get('refused', False) else "ACCEPTED"
        print(f"[{i+1}] {refused_str}: {prompt}")
        
        # Show beginning of generation if available
        gen = item.get('generation', '')[:80]
        if gen:
            if len(item.get('generation', '')) > 80:
                gen += "..."
            print(f"    Response: {gen}")
        print()
    
    print("-" * 60)


if __name__ == "__main__":
    main()
