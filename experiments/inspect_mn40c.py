"""Inspect a ModelNet40-C data root and emit an index JSON.

Usage:
  python -m experiments.inspect_mn40c --root /path/to/ModelNet40-C --out experiments/data/mn40c_index.json

Layout heuristics: every .npy/.npz file whose name contains a known
corruption keyword is registered; severity parsed from a trailing integer
in the file name, defaulting to a per-file stack (severity = 0..K-1).
The script prints the discovered structure so the caller can verify it.
"""
import argparse
import json
import os

from experiments.robustness.mn40c import CORRUPTIONS, taxonomy_of


def build_index(root: str):
    index = {}
    unknown = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith((".npy", ".npz")):
                continue
            stem = os.path.splitext(fname)[0].lower().replace("-", "_").replace(" ", "_")
            match = None
            for corr in sorted(CORRUPTIONS, key=len, reverse=True):
                if corr in stem:
                    match = corr
                    break
            if match is None:
                unknown.append(os.path.join(dirpath, fname))
                continue
            sev = 0
            for tok in stem.replace(match, " ").split("_"):
                if tok.isdigit():
                    sev = int(tok)
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            index[rel] = {"corruption": match, "severity": sev,
                          "taxonomy": taxonomy_of(match)}
    return index, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    index, unknown = build_index(args.root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"root": args.root, "files": index}, f, indent=2, sort_keys=True)
    print(f"indexed {len(index)} files -> {args.out}")
    print("corruptions found:", sorted({m['corruption'] for m in index.values()}))
    if unknown:
        print(f"WARNING: {len(unknown)} unclassified files, e.g. {unknown[:3]}")


if __name__ == "__main__":
    main()
