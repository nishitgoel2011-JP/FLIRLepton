"""
Convert a CVAT "YOLO 1.1" export into the folder structure and dataset.yaml
required by ultralytics YOLOv8 / YOLO11.

Usage:
    python prepare_dataset.py --input  path/to/cvat_export \
                              --output path/to/dataset \
                              --val-split 0.2

The annotation .txt files are format-compatible; only the folder layout changes.
"""

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="CVAT YOLO 1.1 → ultralytics dataset converter")
    p.add_argument("--input",     required=True, help="Path to CVAT YOLO 1.1 export folder")
    p.add_argument("--output",    required=True, help="Destination dataset folder")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data to use for validation (default 0.2)")
    p.add_argument("--seed",      type=int, default=42, help="Random seed for split")
    return p.parse_args()


def read_class_names(cvat_root: Path) -> list[str]:
    """Read class names from obj.names."""
    names_file = cvat_root / "obj.names"
    if not names_file.exists():
        raise FileNotFoundError(f"obj.names not found in {cvat_root}")
    names = [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]
    return names


def collect_samples(cvat_root: Path) -> list[Path]:
    """
    Find all image files inside the export.  CVAT puts them in obj_train_data/
    but also handles flat exports gracefully.
    """
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(cvat_root.rglob(f"*{ext}"))
        candidates.extend(cvat_root.rglob(f"*{ext.upper()}"))
    # Only keep images that have a matching label file
    paired = [p for p in candidates if p.with_suffix(".txt").exists()]
    if not paired:
        raise RuntimeError(
            f"No image+label pairs found under {cvat_root}.\n"
            "Make sure the export contains both images and .txt annotation files."
        )
    return sorted(paired)


def split(samples: list, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


def copy_split(samples: list[Path], out_img_dir: Path, out_lbl_dir: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in samples:
        lbl_path = img_path.with_suffix(".txt")
        shutil.copy2(img_path, out_img_dir / img_path.name)
        shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)


def write_yaml(out_root: Path, class_names: list[str]):
    yaml_path = out_root / "dataset.yaml"
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def main():
    args = parse_args()
    cvat_root = Path(args.input)
    out_root  = Path(args.output)

    if not cvat_root.exists():
        raise SystemExit(f"Input folder not found: {cvat_root}")

    print(f"Reading CVAT export from : {cvat_root}")
    class_names = read_class_names(cvat_root)
    print(f"Classes ({len(class_names)}): {class_names}")

    samples = collect_samples(cvat_root)
    print(f"Found {len(samples)} annotated image(s)")

    train_samples, val_samples = split(samples, args.val_split, args.seed)
    print(f"Split → train: {len(train_samples)}  val: {len(val_samples)}")

    copy_split(train_samples, out_root / "images/train", out_root / "labels/train")
    copy_split(val_samples,   out_root / "images/val",   out_root / "labels/val")

    yaml_path = write_yaml(out_root, class_names)
    print(f"\nDataset written to : {out_root}")
    print(f"dataset.yaml       : {yaml_path}")
    print("\nTo train YOLOv8:")
    print(f"  yolo detect train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=320")
    print("\nTo train YOLO11:")
    print(f"  yolo detect train model=yolo11n.pt data={yaml_path} epochs=50 imgsz=320")


if __name__ == "__main__":
    main()
