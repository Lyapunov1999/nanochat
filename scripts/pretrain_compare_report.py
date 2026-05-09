"""
Summarize base-model experiment reports across optimizer runs.

Expected directory layout:
  <root>/<optimizer>/report/
"""

import argparse
import csv
import os
from typing import Dict


def parse_markdown_kv(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- "):
                continue
            key, sep, value = line[2:].partition(":")
            if sep:
                data[key.strip()] = value.strip()
    return data


def collect_rows(root: str):
    rows = []
    for entry in sorted(os.listdir(root)):
        optimizer_dir = os.path.join(root, entry)
        if not os.path.isdir(optimizer_dir):
            continue
        report_dir = os.path.join(optimizer_dir, "report")
        training = parse_markdown_kv(os.path.join(report_dir, "base-model-training.md"))
        evaluation = parse_markdown_kv(os.path.join(report_dir, "base-model-evaluation.md"))
        if not training and not evaluation:
            continue
        optimizer = training.get("optimizer", entry)
        rows.append({
            "optimizer": optimizer,
            "model_tag": training.get("model_tag", "-"),
            "core": evaluation.get("CORE metric", "-"),
            "train_bpb": evaluation.get("train bpb", "-"),
            "val_bpb": evaluation.get("val bpb", training.get("Final validation bpb", "-")),
            "min_val_bpb": training.get("Minimum validation bpb", "-"),
            "train_time": training.get("Total training time", "-"),
            "train_flops": training.get("Total training flops", "-"),
            "mfu": training.get("MFU %", "-"),
            "peak_memory": training.get("Peak memory usage", "-"),
            "depth": training.get("depth", "-"),
            "ratio": training.get("target_param_data_ratio", "-"),
            "device_batch_size": training.get("device_batch_size", "-"),
            "total_batch_size": training.get("total_batch_size", "-"),
            "fp8": training.get("fp8", "-"),
            "ddp_world_size": training.get("DDP world size", "-"),
            "report_dir": report_dir,
        })
    return rows


def write_csv(root: str, rows):
    path = os.path.join(root, "compare.csv")
    fields = [
        "optimizer", "model_tag", "core", "train_bpb", "val_bpb", "min_val_bpb",
        "train_time", "train_flops", "mfu", "peak_memory", "depth", "ratio",
        "device_batch_size", "total_batch_size", "fp8", "ddp_world_size", "report_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(root: str, rows):
    path = os.path.join(root, "compare.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Base Optimizer Comparison\n\n")
        if not rows:
            f.write("No reports found.\n")
            return path
        f.write("| Optimizer | Model Tag | CORE | Val BPB | Min Val BPB | Time | MFU | Peak Memory |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['optimizer']} | {row['model_tag']} | {row['core']} | "
                f"{row['val_bpb']} | {row['min_val_bpb']} | {row['train_time']} | "
                f"{row['mfu']} | {row['peak_memory']} |\n"
            )
        f.write("\n## Run Details\n\n")
        for row in rows:
            f.write(
                f"- `{row['optimizer']}`: report_dir={row['report_dir']}, depth={row['depth']}, "
                f"ratio={row['ratio']}, device_batch_size={row['device_batch_size']}, "
                f"total_batch_size={row['total_batch_size']}, fp8={row['fp8']}, "
                f"world_size={row['ddp_world_size']}\n"
            )
    return path


def main():
    parser = argparse.ArgumentParser(description="Summarize base-model optimizer comparison reports.")
    parser.add_argument("--root", required=True, help="Comparison root directory")
    args = parser.parse_args()

    rows = collect_rows(args.root)
    write_csv(args.root, rows)
    markdown_path = write_markdown(args.root, rows)
    print(f"Wrote comparison summary to {markdown_path}")


if __name__ == "__main__":
    main()
