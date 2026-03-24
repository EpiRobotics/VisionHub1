"""Convert LabelMe JSON annotations to binary mask PNGs for panel segmentation training.

Usage:
    python labelme_to_masks.py <labelme_dir> [--output <output_dir>] [--label panel]

Example:
    python labelme_to_masks.py D:\\tst\\整板半板
    python labelme_to_masks.py D:\\tst\\整板半板 --output D:\\datasets\\panel_seg

This script:
1. Reads all .json files in <labelme_dir> (LabelMe annotation format)
2. Draws polygons with the specified label as white (255) on black (0) background
3. Saves binary mask PNGs to <output_dir>/masks/
4. Copies (or symlinks) the corresponding original images to <output_dir>/images/

The output structure matches the panel_seg_v1 plugin's expected training dataset:
    <output_dir>/
        images/     <- original images
        masks/      <- binary mask PNGs (white=panel, black=background)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite replacement that handles non-ASCII (e.g. Chinese) paths on Windows.

    OpenCV's imwrite silently fails with unicode paths on Windows.
    This uses imencode + Python file I/O which handles unicode correctly.
    """
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def find_image_for_json(json_path: Path, labelme_dir: Path) -> Path | None:
    """Find the original image file corresponding to a LabelMe JSON file.

    LabelMe stores imagePath in the JSON. We try:
    1. The imagePath field in the JSON (absolute or relative)
    2. Same stem with common image extensions in the same directory
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try imagePath from JSON
    image_path_str = data.get("imagePath", "")
    if image_path_str:
        # Could be absolute or relative
        candidate = Path(image_path_str)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        # Relative to JSON file's directory
        candidate = json_path.parent / candidate
        if candidate.exists():
            return candidate
        # Relative to labelme_dir
        candidate = labelme_dir / image_path_str
        if candidate.exists():
            return candidate

    # Fallback: same stem with common extensions
    stem = json_path.stem
    for ext in IMG_EXTS:
        candidate = json_path.parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Also check uppercase
        candidate = json_path.parent / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate

    return None


def json_to_mask(json_path: Path, label: str) -> tuple[np.ndarray | None, dict]:
    """Convert a LabelMe JSON to a binary mask.

    Args:
        json_path: Path to LabelMe JSON file.
        label: Label name to extract (e.g., "panel").

    Returns:
        (mask, data) where mask is H x W uint8 (0 or 255), data is the parsed JSON.
        Returns (None, data) if no matching shapes found.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight", 0)
    w = data.get("imageWidth", 0)

    if h == 0 or w == 0:
        return None, data

    mask = np.zeros((h, w), dtype=np.uint8)

    found = False
    for shape in data.get("shapes", []):
        shape_label = shape.get("label", "").strip()
        # Case-insensitive match
        if shape_label.lower() != label.lower():
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        # Convert points to numpy array of int32 for cv2.fillPoly
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        found = True

    if not found:
        return None, data

    return mask, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to binary masks for panel segmentation training."
    )
    parser.add_argument(
        "labelme_dir",
        type=str,
        help="Directory containing LabelMe JSON files and images.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output directory. Defaults to <labelme_dir>/dataset.",
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        default="panel",
        help="Label name to extract from annotations (default: 'panel').",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy original images to output/images/ (default: True).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of samples to use for validation (0.0 = no split, e.g. 0.15 = 15%%).",
    )

    args = parser.parse_args()

    labelme_dir = Path(args.labelme_dir)
    if not labelme_dir.exists():
        print(f"ERROR: Directory not found: {labelme_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else labelme_dir / "dataset"
    label = args.label

    # Create output directories
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if args.val_split > 0:
        val_images_dir = output_dir / "val_images"
        val_masks_dir = output_dir / "val_masks"
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_masks_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No .json files found in: {labelme_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {labelme_dir}")
    print(f"Extracting label: '{label}'")
    print(f"Output directory: {output_dir}")
    print()

    # Process each JSON
    successful = []
    skipped = 0
    errors = 0

    for json_path in json_files:
        stem = json_path.stem

        # Convert to mask
        mask, data = json_to_mask(json_path, label)

        if mask is None:
            print(f"  SKIP: {json_path.name} (no '{label}' shapes found or invalid dimensions)")
            skipped += 1
            continue

        # Find original image
        img_path = find_image_for_json(json_path, labelme_dir)
        if img_path is None:
            print(f"  WARN: {json_path.name} - original image not found, saving mask only")

        # Save mask as PNG (use imwrite_unicode for non-ASCII path support on Windows)
        mask_out = masks_dir / f"{stem}.png"
        imwrite_unicode(str(mask_out), mask)

        # Copy original image
        if img_path is not None and args.copy_images:
            img_out = images_dir / f"{stem}{img_path.suffix}"
            if not img_out.exists() or img_out.resolve() != img_path.resolve():
                shutil.copy2(str(img_path), str(img_out))

        panel_pct = (mask > 0).sum() / mask.size * 100
        print(f"  OK: {stem} -> mask {mask.shape[1]}x{mask.shape[0]}, panel={panel_pct:.1f}%")
        successful.append(stem)

    print()
    print(f"Done: {len(successful)} masks created, {skipped} skipped, {errors} errors")

    # Validation split
    if args.val_split > 0 and len(successful) > 1:
        import random
        random.seed(42)
        random.shuffle(successful)
        n_val = max(1, int(len(successful) * args.val_split))
        val_stems = set(successful[:n_val])

        moved_count = 0
        for stem in val_stems:
            # Move mask
            mask_src = masks_dir / f"{stem}.png"
            mask_dst = val_masks_dir / f"{stem}.png"
            if mask_src.exists():
                shutil.move(str(mask_src), str(mask_dst))

            # Move image (find by stem)
            for img_file in images_dir.glob(f"{stem}.*"):
                img_dst = val_images_dir / img_file.name
                shutil.move(str(img_file), str(img_dst))
                moved_count += 1
                break

        print(f"Validation split: {len(val_stems)} samples moved to val_images/ and val_masks/")
        print(f"Training: {len(successful) - len(val_stems)}, Validation: {len(val_stems)}")

    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print(f"    images/   ({len(list(images_dir.glob('*')))} files)")
    print(f"    masks/    ({len(list(masks_dir.glob('*')))} files)")
    if args.val_split > 0:
        print(f"    val_images/ ({len(list(val_images_dir.glob('*')))} files)")
        print(f"    val_masks/  ({len(list(val_masks_dir.glob('*')))} files)")
    print()
    print("Ready for panel_seg_v1 training!")


if __name__ == "__main__":
    main()
