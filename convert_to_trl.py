#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

from datasets import load_dataset, Dataset
from tqdm import tqdm


def to_trl_example(row: Dict) -> Dict:
    """
    Map a raw row to TRL SFT-style chat columns:
      - prompt:   list[{"role": "user", "content": <question>}]
      - completion: list[{"role": "assistant", "content": <no_think_response>}]
    """
    question = (row.get("question") or "").strip()
    answer = (row.get("no_think_response") or "").strip()

    # minimal cleanup: strip surrounding whitespace; (keep LaTeX/boxed content as-is)
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ReasoningTransferability/math_sft_40K",
                    help="Source dataset path on HF Hub")
    ap.add_argument("--split", default="train", help="Split to load (default: train)")
    ap.add_argument("--repo-id", required=True,
                    help="Destination HF dataset repo, e.g. yourname/math_sft_40K_trl")
    ap.add_argument("--private", action="store_true", help="Push as a private dataset")
    ap.add_argument("--revision", default=None, help="Target branch (default: main)")
    ap.add_argument("--push-max-shard-size", default="500MB",
                    help="Max shard size when pushing parquet (HF will shard if bigger)")
    ap.add_argument("--num-proc", type=int, default=8, help="Parallelism for map()")
    args = ap.parse_args()

    print(f"Loading {args.source} split={args.split} ...")
    ds = load_dataset(args.source, split=args.split)

    # Validate expected columns exist
    expected = {"question", "no_think_response"}
    missing = expected - set(ds.column_names)
    if missing:
        raise ValueError(f"Missing columns {missing} in source dataset.")

    print("Converting to TRL chat format (prompt/completion) ...")
    ds_trl: Dataset = ds.map(
        to_trl_example,
        remove_columns=[c for c in ds.column_names if c not in expected],
        num_proc=args.num_proc,
        desc="Mapping rows",
    )

    # Optional sanity checks
    print(ds_trl[0]["prompt"])
    print(ds_trl[0]["completion"])

    print(f"Pushing to hub: {args.repo_id}")
    # You can set HF_TOKEN env var or be logged in via `huggingface-cli login`
    ds_trl.push_to_hub(
        args.repo_id,
        private=args.private,
        revision=args.revision,
        max_shard_size=args.push_max_shard_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
