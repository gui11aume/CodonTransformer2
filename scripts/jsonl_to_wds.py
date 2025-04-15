#!/usr/bin/env python3
import argparse
import json
import os

import webdataset as wds


def jsonl_to_wds(input_file, output_dir, shard_size=2048):
    """
    Convert a JSONL file to WebDataset format.

    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save the WebDataset shards
        shard_size: Number of samples per shard
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create pattern for output shards.
    output_pattern = os.path.join(output_dir, "shard-%06d.tar")

    # Open the sink for writing WebDataset shards
    with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
        with open(input_file) as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    sample = {"__key__": f"sample{i:08d}", "json": json.dumps(data)}
                    sink.write(sample)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON at line {i + 1}, skipping")
                except Exception as e:
                    print(f"Error processing line {i + 1}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL file to WebDataset format")
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("output_dir", type=str, help="Directory to save WebDataset shards")
    parser.add_argument(
        "--shard-size", type=int, default=32768, help="Number of samples per shard (default: 2048)"
    )

    args = parser.parse_args()

    jsonl_to_wds(args.input_file, args.output_dir, args.shard_size)

    print(f"Conversion complete. WebDataset shards saved to {args.output_dir}")
