#!/usr/bin/env python3
"""
Convert JSONL file to JSON array format for MiniGPT-4 training.
"""
import json
import sys

def convert_jsonl_to_json(jsonl_file, json_file):
    """Convert JSONL file to JSON array format."""
    data = []

    print(f"Reading from {jsonl_file}...")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)

                if i % 100 == 0:
                    print(f"Processed {i} lines...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue

    print(f"Writing {len(data)} items to {json_file}...")

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Conversion completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl_to_json.py <input_jsonl> <output_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_jsonl_to_json(input_file, output_file)

