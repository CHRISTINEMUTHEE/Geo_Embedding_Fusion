#!/usr/bin/env python
"""Example artifact script template."""
import argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def main(light: bool = False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if light:
        data = list(range(10))
    else:
        data = list(range(100))
    
    output_file = OUTPUT_DIR / "example.txt"
    output_file.write_text(f"Generated {len(data)} items: {data[:5]}...")
    print(f"Artifact produced: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--light", action="store_true", help="Fast mode for prototyping")
    args = parser.parse_args()
    main(light=args.light)
