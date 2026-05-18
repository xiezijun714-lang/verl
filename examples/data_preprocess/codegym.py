# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert CodeGym JSONL training instances to verl parquet format."""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_row(row: dict, idx: int, split: str, codegym_server_host: str | None) -> dict:
    ability = row["ability"]
    extra_info = dict(row.get("extra_info") or {})
    extra_info.update(
        {
            "split": split,
            "index": extra_info.get("index", idx),
            "code_id": row.get("code_id"),
            "env_str": ability,
            "need_tools_kwargs": False,
        }
    )
    if codegym_server_host:
        extra_info["codegym_server_host"] = codegym_server_host

    return {
        "data_source": row.get("data_source", "codegym_gym_v1"),
        "agent_name": "codegym_agent",
        "prompt": row["prompt"],
        "ability": ability,
        "reward_model": row.get("reward_model", {"style": "agent_env", "ground_truth": None}),
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Path to CodeGym training_instance.jsonl.")
    parser.add_argument("--local_dir", default=None, help="Deprecated alias of --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/codegym", help="Output directory.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codegym_server_host", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    random.Random(args.seed).shuffle(rows)
    val_size = int(len(rows) * args.val_ratio)
    if args.val_ratio > 0 and rows:
        val_size = max(1, val_size)

    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    local_save_dir = args.local_dir or args.local_save_dir
    output_dir = Path(os.path.expanduser(local_save_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        [convert_row(row, idx, "train", args.codegym_server_host) for idx, row in enumerate(train_rows)]
    )
    val_df = pd.DataFrame(
        [convert_row(row, idx, "validation", args.codegym_server_host) for idx, row in enumerate(val_rows)]
    )

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "validation.parquet", index=False)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(output_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
