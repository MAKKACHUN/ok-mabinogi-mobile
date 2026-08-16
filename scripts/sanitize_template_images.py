"""Remove private screen content while preserving COCO template pixels exactly."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _load_regions(coco_path: Path) -> dict[Path, list[tuple[int, int, int, int]]]:
    with coco_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    image_names = {image["id"]: image["file_name"] for image in data["images"]}
    regions: dict[Path, list[tuple[int, int, int, int]]] = defaultdict(list)
    for annotation in data["annotations"]:
        image_name = image_names[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        left = round(x)
        top = round(y)
        right = round(x + width)
        bottom = round(y + height)
        regions[coco_path.parent / Path(image_name)].append(
            (left, top, right, bottom)
        )

    for image_name, safe_regions in data.get("privacy_keep_regions", {}).items():
        image_path = coco_path.parent / Path(image_name)
        for left, top, right, bottom in safe_regions:
            regions[image_path].append((left, top, right, bottom))
    return regions


def sanitize_coco_images(coco_path: Path, *, dry_run: bool = False) -> list[Path]:
    """White out everything except annotated regions and return changed paths."""
    changed: list[Path] = []
    for image_path, regions in _load_regions(coco_path).items():
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read COCO image: {image_path}")

        height, width = image.shape[:2]
        sanitized = np.full_like(image, 255)
        for left, top, right, bottom in regions:
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(
                    f"Invalid bbox for {image_path}: "
                    f"{left},{top},{right},{bottom} within {width}x{height}"
                )
            sanitized[top:bottom, left:right] = image[top:bottom, left:right]

        if np.array_equal(image, sanitized):
            continue
        changed.append(image_path)
        if dry_run:
            continue

        temporary_path = image_path.with_name(f".{image_path.name}.privacy-tmp.png")
        if not cv2.imwrite(str(temporary_path), sanitized):
            raise OSError(f"Could not write sanitized image: {temporary_path}")
        os.replace(temporary_path, image_path)

    return changed


def find_unsanitized_images(coco_path: Path) -> list[Path]:
    """Return images containing any non-white pixel outside annotated regions."""
    unsafe: list[Path] = []
    for image_path, regions in _load_regions(coco_path).items():
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read COCO image: {image_path}")

        keep = np.zeros(image.shape[:2], dtype=bool)
        height, width = image.shape[:2]
        for left, top, right, bottom in regions:
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(f"Invalid bbox for {image_path}")
            keep[top:bottom, left:right] = True

        outside = image[~keep]
        if outside.size and not np.all(outside == 255):
            unsafe.append(image_path)
    return unsafe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coco_path",
        nargs="?",
        type=Path,
        default=Path("assets/coco_annotations.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    coco_path = args.coco_path.resolve()
    if args.check:
        unsafe = find_unsanitized_images(coco_path)
        for path in unsafe:
            print(path)
        return 1 if unsafe else 0

    changed = sanitize_coco_images(coco_path)
    for path in changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
