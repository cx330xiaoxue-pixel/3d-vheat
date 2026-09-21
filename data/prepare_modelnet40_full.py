#!/usr/bin/env python3
"""Prepare full ModelNet40 npy arrays for the server runs.

Two sources, in order of preference:

A. Official benchmark HDF5 (modelnet40_ply_hdf5_2048.zip) -- the standard
   2048-point data used by PointNet/DGCNN/etc. 1024 points are subsampled
   following the paper protocol. 40 classes, 9843/2468 split.

B. Local OFF files (data/modelnet40/ModelNet40) -- full 9843/2468 raw meshes.
   NOTE: 7 classes (bathtub/desk/dresser/monitor/night_stand/sofa/table) use
   a glued OFF header (``OFF3514 3546 0`` on one line); the legacy converter
   silently dropped them, which is why the old npy files only had 33 classes.

Usage:
  python prepare_modelnet40_full.py --source hdf5 --hdf5-zip /path/to/modelnet40_ply_hdf5_2048.zip --out-dir data/full
  python prepare_modelnet40_full.py --source off  --off-root data/modelnet40/ModelNet40 --out-dir data/full

Outputs (in --out-dir):
  modelnet40_train_points.npy  (9843, 1024, 3) float32, unit-sphere normalized
  modelnet40_train_labels.npy  (9843,)         int64
  modelnet40_test_points.npy   (2468, 1024, 3)
  modelnet40_test_labels.npy   (2468,)
  prepare_report.json          counts + checksums
"""
import argparse
import hashlib
import json
import os
import random
import sys
import zipfile

import numpy as np

SEED = 20260919


def normalize(points: np.ndarray) -> np.ndarray:
    """Center + unit-sphere normalization (matches MN40.__getitem__ at train time)."""
    pts = np.asarray(points, dtype=np.float32)
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    d = np.linalg.norm(pts, axis=2, keepdims=True).max(axis=1, keepdims=True)
    d = np.maximum(d, 1e-8)
    return (pts / d).astype(np.float32)


def subsample(points: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    m = points.shape[0]
    if m >= n:
        idx = rng.sample(range(m), n)
    else:
        idx = [rng.randrange(m) for _ in range(n)]
    return points[idx]


def prepare_hdf5(zip_path: str, out_dir: str, n_points: int, rng: random.Random):
    import h5py

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".h5")]
        if not members:
            raise SystemExit(f"no .h5 files inside {zip_path}")

        def load_split(split: str):
            pts_all, lbl_all = [], []
            for m in sorted(members):
                if split not in os.path.basename(m):
                    continue
                with zf.open(m) as fh:
                    with h5py.File(fh, "r") as h:
                        data = h["data"][:]      # (N, 2048, 3)
                        label = h["label"][:]    # (N, 1) or (N,)
                label = np.asarray(label).reshape(-1)
                for i in range(data.shape[0]):
                    pts_all.append(subsample(np.asarray(data[i], dtype=np.float32),
                                             n_points, rng))
                    lbl_all.append(int(label[i]))
            return np.stack(pts_all), np.asarray(lbl_all, dtype=np.int64)

        tr_pts, tr_lbl = load_split("train")
        te_pts, te_lbl = load_split("test")
    return tr_pts, tr_lbl, te_pts, te_lbl


def read_off_vertices(path: str) -> np.ndarray:
    """OFF reader tolerant of glued headers (``OFF3514 3546 0``)."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("empty OFF")
    first = lines[0]
    if first.startswith("OFF") and len(first) > 3:
        n_vert = int(first[3:].split()[0])
        rest = lines[1:]
    elif first == "OFF":
        n_vert = int(lines[1].split()[0])
        rest = lines[2:]
    else:
        parts = first.split()
        if parts[0] == "OFF":
            n_vert = int(parts[1])
            rest = lines[1:]
        else:
            n_vert = int(parts[0])
            rest = lines[1:]
    verts = []
    for l in rest:
        if len(verts) >= n_vert:
            break
        try:
            verts.append([float(x) for x in l.split()[:3]])
        except ValueError:
            continue
    if not verts:
        raise ValueError("no vertices")
    return np.asarray(verts, dtype=np.float32)


def prepare_off(off_root: str, out_dir: str, n_points: int, rng: random.Random):
    cats = sorted(d for d in os.listdir(off_root)
                  if os.path.isdir(os.path.join(off_root, d)))
    if len(cats) != 40:
        raise SystemExit(f"expected 40 categories under {off_root}, found {len(cats)}")

    def load_split(split: str):
        pts_all, lbl_all, skipped = [], [], []
        for label, cat in enumerate(cats):
            d = os.path.join(off_root, cat, split)
            for fname in sorted(os.listdir(d)):
                if not fname.endswith(".off"):
                    continue
                try:
                    verts = read_off_vertices(os.path.join(d, fname))
                    pts_all.append(subsample(verts, n_points, rng))
                    lbl_all.append(label)
                except Exception as exc:  # noqa: BLE001
                    skipped.append((cat, fname, str(exc)))
        return (np.stack(pts_all), np.asarray(lbl_all, dtype=np.int64), skipped)

    tr_pts, tr_lbl, tr_skip = load_split("train")
    te_pts, te_lbl, te_skip = load_split("test")
    print(f"[off] skipped train={len(tr_skip)} test={len(te_skip)}")
    for row in (tr_skip + te_skip)[:10]:
        print("  skipped:", row)
    return tr_pts, tr_lbl, te_pts, te_lbl


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hdf5", "off"], required=True)
    ap.add_argument("--hdf5-zip", default=None)
    ap.add_argument("--off-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.source == "hdf5":
        if not args.hdf5_zip:
            raise SystemExit("--hdf5-zip required for --source hdf5")
        tr_pts, tr_lbl, te_pts, te_lbl = prepare_hdf5(
            args.hdf5_zip, args.out_dir, args.n_points, rng)
    else:
        if not args.off_root:
            raise SystemExit("--off-root required for --source off")
        tr_pts, tr_lbl, te_pts, te_lbl = prepare_off(
            args.off_root, args.out_dir, args.n_points, rng)

    tr_pts = normalize(tr_pts)
    te_pts = normalize(te_pts)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "modelnet40_train_points.npy": tr_pts,
        "modelnet40_train_labels.npy": tr_lbl,
        "modelnet40_test_points.npy": te_pts,
        "modelnet40_test_labels.npy": te_lbl,
    }
    for name, arr in out.items():
        np.save(os.path.join(args.out_dir, name), arr)

    report = {
        "source": args.source,
        "seed": args.seed,
        "n_points": args.n_points,
        "train": {"n": int(len(tr_lbl)), "classes": int(len(np.unique(tr_lbl)))},
        "test": {"n": int(len(te_lbl)), "classes": int(len(np.unique(te_lbl)))},
        "shapes": {k: list(v.shape) for k, v in out.items()},
        "sha1": {k: sha1_of(os.path.join(args.out_dir, k)) for k in out},
    }
    with open(os.path.join(args.out_dir, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2))
    expected = (9843, 2468)
    ok = report["train"]["n"] == expected[0] and report["test"]["n"] == expected[1]
    print(f"[{'OK' if ok else 'MISMATCH'}] expected {expected}, "
          f"got ({report['train']['n']}, {report['test']['n']})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
