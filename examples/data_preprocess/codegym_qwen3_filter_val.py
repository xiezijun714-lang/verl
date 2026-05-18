#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("/root/paddlejob/workspace/env_run/xzj")
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "dataset/CodeGym"
DEFAULT_TASK_DIR = DEFAULT_DATA_ROOT / "task_en_instruction_en_env"
DEFAULT_TRAIN_FILE = DEFAULT_ROOT / "echo/data/codegym_filtered_grpo/train_12800.parquet"
DEFAULT_CANDIDATES = DEFAULT_DATA_ROOT / "val/val_candidates_qwen3_filter.parquet"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "val/val_128_qwen3_pass25.parquet"


def env_id(ability: str) -> str:
    value = str(ability)
    if value.startswith("codegym_v1@"):
        value = value[len("codegym_v1@") :]
    return value.split("@", 1)[0]


def parse_extra(extra_info):
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str) and extra_info:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(extra_info)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


def solve_round(extra_info):
    parsed = parse_extra(extra_info)
    if "solve_fc_round" in parsed:
        try:
            return int(parsed["solve_fc_round"])
        except Exception:
            pass
    std_traj = parsed.get("std_traj")
    if std_traj:
        try:
            return len(ast.literal_eval(std_traj))
        except Exception:
            pass
    return None


def function_count(prompt):
    try:
        messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
        text = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
    except Exception:
        text = str(prompt)
    return len(set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, re.M)))


def load_train_ids(train_file: Path):
    train_df = pd.read_parquet(train_file, columns=["code_id", "ability"])
    return (
        set(train_df["code_id"].astype(str)),
        set(train_df["ability"].astype(str)),
        set(train_df["ability"].map(env_id)),
    )


def make_candidates(args):
    train_code_ids, train_abilities, train_env_ids = load_train_ids(args.train_file)
    df = pd.read_parquet(args.source_file)
    df["_env_id"] = df["ability"].map(env_id)
    df["_solve_round"] = df["extra_info"].map(solve_round)
    df["_function_count"] = df["prompt"].map(function_count)

    filtered = df[
        ~df["code_id"].astype(str).isin(train_code_ids)
        & ~df["ability"].astype(str).isin(train_abilities)
        & ~df["_env_id"].isin(train_env_ids)
        & df["_solve_round"].between(args.min_round, args.max_round, inclusive="both")
        & (df["_function_count"] >= args.min_functions)
    ].copy()
    filtered = filtered.sample(frac=1, random_state=args.seed).drop_duplicates("code_id", keep="first")
    if args.num_candidates > 0:
        if len(filtered) < args.num_candidates:
            raise RuntimeError(f"not enough candidates: need {args.num_candidates}, got {len(filtered)}")
        filtered = filtered.sample(n=args.num_candidates, random_state=args.seed + 1)
    filtered = filtered.reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.drop(columns=["_env_id", "_solve_round", "_function_count"]).to_parquet(args.output, index=False)
    print(
        f"wrote {args.output}: rows={len(filtered)}, unique_env_id={filtered['_env_id'].nunique()}, "
        f"unique_code_id={filtered['code_id'].nunique()}, "
        f"solve_round_mean={filtered['_solve_round'].mean():.2f}, "
        f"solve_round_p50={filtered['_solve_round'].median():.2f}, "
        f"solve_round_min={filtered['_solve_round'].min()}, solve_round_max={filtered['_solve_round'].max()}"
    )


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def select_val(args):
    candidates = pd.read_parquet(args.candidates)
    rows = list(read_jsonl(args.dump_jsonl))
    if not rows:
        raise RuntimeError(f"empty dump jsonl: {args.dump_jsonl}")

    key = "ability" if "ability" in rows[0] else "code_id"
    if key not in rows[0]:
        raise RuntimeError("validation dump must contain ability or code_id; rerun after ray_trainer dump patch")

    eval_df = pd.DataFrame(
        {
            key: [str(row.get(key)) for row in rows],
            "acc": [float(row.get("acc", row.get("score", 0.0))) for row in rows],
        }
    )
    pass_rate = eval_df.groupby(key)["acc"].agg(["mean", "count"]).rename(columns={"mean": "pass_rate"})
    merged = candidates.merge(pass_rate, left_on=key, right_index=True, how="inner")
    if args.expected_repeats > 0:
        merged = merged[merged["count"] >= args.expected_repeats]

    selected = merged[merged["pass_rate"] <= args.threshold].copy()
    selected = selected.sample(frac=1, random_state=args.seed).drop_duplicates("code_id", keep="first")
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"not enough Qwen3-hard samples: need {args.num_samples}, got {len(selected)} "
            f"(threshold={args.threshold}, evaluated={len(merged)})"
        )

    selected["_env_id"] = selected["ability"].map(env_id)
    selected["_solve_round"] = selected["extra_info"].map(solve_round)
    selected = selected.sample(frac=1, random_state=args.seed)

    if args.target_envs > 0:
        if args.max_configs_per_env != 2:
            raise ValueError("target_envs mode currently expects --max-configs-per-env 2")
        if args.num_samples < args.target_envs or args.num_samples > args.target_envs * args.max_configs_per_env:
            raise ValueError("num_samples must be between target_envs and target_envs * max_configs_per_env")

        groups = {eid: group for eid, group in selected.groupby("_env_id", sort=False)}
        double_envs = [eid for eid, group in groups.items() if len(group) >= 2]
        single_envs = [eid for eid, group in groups.items() if len(group) >= 1]
        need_double = args.num_samples - args.target_envs
        need_single = args.target_envs - need_double
        if len(double_envs) < need_double:
            raise RuntimeError(f"not enough envs with >=2 configs: need {need_double}, got {len(double_envs)}")

        rng = pd.Series(double_envs).sample(frac=1, random_state=args.seed)
        chosen_double = rng.head(need_double).tolist()
        remaining_single = [eid for eid in single_envs if eid not in set(chosen_double)]
        if len(remaining_single) < need_single:
            raise RuntimeError(f"not enough remaining single-config envs: need {need_single}, got {len(remaining_single)}")
        chosen_single = pd.Series(remaining_single).sample(n=need_single, random_state=args.seed + 1).tolist()

        parts = [groups[eid].head(2) for eid in chosen_double] + [groups[eid].head(1) for eid in chosen_single]
        selected = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed + 2).reset_index(drop=True)
    else:
        selected = selected.groupby("_env_id", sort=False).head(args.max_configs_per_env)
        if len(selected) < args.num_samples:
            raise RuntimeError(f"not enough samples after per-env cap: need {args.num_samples}, got {len(selected)}")
        selected = selected.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)

    output = selected.drop(columns=["pass_rate", "count", "_env_id", "_solve_round"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        f"wrote {args.output}: rows={len(output)}, unique_env_id={selected['_env_id'].nunique()}, "
        f"max_configs_per_env={selected.groupby('_env_id').size().max()}, "
        f"pass_rate_mean={selected['pass_rate'].mean():.3f}, "
        f"pass_rate_max={selected['pass_rate'].max():.3f}, solve_round_mean={selected['_solve_round'].mean():.2f}, "
        f"solve_round_p50={selected['_solve_round'].median():.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    make = sub.add_parser("make-candidates")
    make.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    make.add_argument("--source-file", type=Path, default=DEFAULT_TASK_DIR / "train-00003-of-00004.parquet")
    make.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES)
    make.add_argument("--num-candidates", type=int, default=1024)
    make.add_argument("--min-round", type=int, default=10)
    make.add_argument("--max-round", type=int, default=256)
    make.add_argument("--min-functions", type=int, default=4)
    make.add_argument("--seed", type=int, default=42)
    make.set_defaults(func=make_candidates)

    select = sub.add_parser("select")
    select.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    select.add_argument("--dump-jsonl", type=Path, required=True)
    select.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    select.add_argument("--num-samples", type=int, default=128)
    select.add_argument("--threshold", type=float, default=0.25)
    select.add_argument("--expected-repeats", type=int, default=4)
    select.add_argument("--target-envs", type=int, default=0)
    select.add_argument("--max-configs-per-env", type=int, default=2)
    select.add_argument("--seed", type=int, default=42)
    select.set_defaults(func=select_val)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
