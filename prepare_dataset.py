"""
Convert a CVAT "YOLO 1.1" export into the folder structure and dataset.yaml
required by ultralytics YOLOv8 / YOLO11.

Two supported layouts:

  A) Images and labels together (original CVAT full export):
       python prepare_dataset.py --input cvat_export/ --output dataset/

  B) Labels-only export + separate image folder (your case):
       python prepare_dataset.py --input  cvat_export/ \
                                 --images path/to/ir_images/ \
                                 --output dataset/

     The script matches each .txt file to an image by filename stem,
     e.g. frame_001.txt  →  frame_001.png  (any supported extension).

The annotation .txt files are format-compatible with ultralytics;
only the folder layout and dataset.yaml need to be created.
"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="CVAT YOLO 1.1 → ultralytics dataset converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Path to CVAT YOLO 1.1 export folder "
                        "(contains .txt labels, obj.names, obj.data)")
    p.add_argument("--images", default=None,
                   help="Folder containing the raw IR images. "
                        "Required when the CVAT export does NOT include images. "
                        "Images are matched to labels by filename stem.")
    p.add_argument("--output", required=True,
                   help="Destination dataset folder (will be created)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data reserved for validation (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split (default: 42)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_class_names(cvat_root: Path) -> list[str]:
    """Read class names from obj.names."""
    names_file = cvat_root / "obj.names"
    if not names_file.exists():
        raise FileNotFoundError(
            f"obj.names not found in {cvat_root}.\n"
            "Make sure --input points to the CVAT export root."
        )
    return [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]


def build_image_index(image_root: Path) -> dict[str, Path]:
    """
    Walk image_root and return {stem: path} for every image found.
    Handles sub-folders and mixed extensions.
    """
    index: dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for p in image_root.rglob(f"*{ext}"):
            index[p.stem] = p
        for p in image_root.rglob(f"*{ext.upper()}"):
            index[p.stem] = p
    return index


def collect_label_files(cvat_root: Path) -> list[Path]:
    """Return all .txt files that are annotation labels (exclude obj.data etc.)."""
    labels = [
        p for p in cvat_root.rglob("*.txt")
        # CVAT sometimes includes train.txt / test.txt listing image paths — skip those
        if p.stem not in {"train", "test", "valid", "val"}
        and not p.read_text().startswith("obj_")   # skip path-list files
    ]
    if not labels:
        raise RuntimeError(f"No .txt label files found under {cvat_root}.")
    return sorted(labels)


def collect_samples_labels_only(cvat_root: Path,
                                image_root: Path) -> list[tuple[Path, Path]]:
    """
    Match label .txt files from cvat_root to images from image_root by stem.
    Returns [(img_path, lbl_path), ...].
    """
    image_index = build_image_index(image_root)
    if not image_index:
        raise RuntimeError(
            f"No images found under {image_root}.\n"
            f"Supported extensions: {IMAGE_EXTENSIONS}"
        )

    label_files = collect_label_files(cvat_root)

    matched, unmatched = [], []
    for lbl in label_files:
        img = image_index.get(lbl.stem)
        if img:
            matched.append((img, lbl))
        else:
            unmatched.append(lbl.name)

    if unmatched:
        print(f"  WARNING: {len(unmatched)} label file(s) had no matching image "
              f"and were skipped:")
        for name in unmatched[:10]:
            print(f"    {name}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    if not matched:
        raise RuntimeError(
            "No label files could be matched to images.\n"
            "Check that image filenames match label filenames "
            "(e.g. frame_001.png ↔ frame_001.txt)."
        )
    return sorted(matched)


def collect_samples_collocated(cvat_root: Path) -> list[tuple[Path, Path]]:
    """
    Original mode: find images that sit next to their .txt label files.
    """
    pairs = []
    for ext in IMAGE_EXTENSIONS:
        for img in cvat_root.rglob(f"*{ext}"):
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
        for img in cvat_root.rglob(f"*{ext.upper()}"):
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    if not pairs:
        raise RuntimeError(
            f"No image+label pairs found under {cvat_root}.\n"
            "If your CVAT export does not include images, pass --images <folder>."
        )
    return sorted(pairs)


def split(samples: list, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


def copy_split(samples: list[tuple[Path, Path]],
               out_img_dir: Path, out_lbl_dir: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in samples:
        shutil.copy2(img_path, out_img_dir / img_path.name)
        # Label must share the same stem as the image
        shutil.copy2(lbl_path, out_lbl_dir / (img_path.stem + ".txt"))


def write_yaml(out_root: Path, class_names: list[str]) -> Path:
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cvat_root = Path(args.input)
    out_root  = Path(args.output)

    if not cvat_root.exists():
        raise SystemExit(f"Input folder not found: {cvat_root}")

    print(f"\nReading CVAT export from : {cvat_root}")
    class_names = read_class_names(cvat_root)
    print(f"Classes ({len(class_names)}): {class_names}")

    if args.images:
        image_root = Path(args.images)
        if not image_root.exists():
            raise SystemExit(f"Images folder not found: {image_root}")
        print(f"Image folder            : {image_root}")
        print("Mode                    : labels-only export + separate images")
        samples = collect_samples_labels_only(cvat_root, image_root)
    else:
        print("Mode                    : collocated images + labels")
        samples = collect_samples_collocated(cvat_root)

    print(f"Matched pairs           : {len(samples)}")

    train_samples, val_samples = split(samples, args.val_split, args.seed)
    print(f"Split → train: {len(train_samples)}   val: {len(val_samples)}")

    copy_split(train_samples, out_root / "images/train", out_root / "labels/train")
    copy_split(val_samples,   out_root / "images/val",   out_root / "labels/val")

    yaml_path = write_yaml(out_root, class_names)

    print(f"\nDataset written to  : {out_root}")
    print(f"dataset.yaml        : {yaml_path}")
    print("\n── Train YOLOv8 ──────────────────────────────────────────────")
    print(f"  yolo detect train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=320")
    print("\n── Train YOLO11 ──────────────────────────────────────────────")
    print(f"  yolo detect train model=yolo11n.pt data={yaml_path} epochs=50 imgsz=320")
    print()


if __name__ == "__main__":
    main()
