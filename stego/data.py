"""Finds dataset images and checks their count against the paper's Table III."""

from __future__ import annotations

import json
import os
from pathlib import Path

PUBLISHED_COUNTS = {"voc2007": 9963, "voc2012": 11540, "coco2014": 164062}
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class DataError(RuntimeError):
    """Raised when images cannot be found or the count is wrong."""


def _walk(root: Path):
    for current, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        yield Path(current), sorted(files)


def _images_in(folder: Path, files: list[str]) -> dict[str, Path]:
    return {name: folder / name for name in files if name.lower().endswith(_IMAGE_SUFFIXES)}


def _discover_voc(root: Path, year: str, use_trainval_list: bool) -> dict[str, Path]:
    images: dict[str, Path] = {}
    allowed: set[str] = set()
    for folder, files in _walk(root):
        parts = folder.parts
        if len(parts) >= 2 and parts[-1] == "JPEGImages" and parts[-2] == f"VOC{year}":
            images.update(_images_in(folder, files))
        if (
            use_trainval_list
            and len(parts) >= 3
            and parts[-3:] == ("VOC" + year, "ImageSets", "Main")
            and "trainval.txt" in files
        ):
            lines = (folder / "trainval.txt").read_text(encoding="utf-8").split()
            allowed.update(f"{line}.jpg" for line in lines)
    if use_trainval_list:
        if not allowed:
            raise DataError(f"Could not find VOC{year}/ImageSets/Main/trainval.txt under {root}")
        images = {name: path for name, path in images.items() if name in allowed}
    return images


def _discover_coco(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for folder, files in _walk(root):
        if folder.name in ("train2014", "val2014", "test2014"):
            images.update(_images_in(folder, files))
    return images


def _discover_synthetic(root: Path) -> dict[str, Path]:
    folder = root / "images"
    return _images_in(folder, sorted(os.listdir(folder))) if folder.is_dir() else {}


def expected_count(config, root: Path) -> int:
    if config.expected_images:
        return config.expected_images
    if config.dataset == "synthetic":
        meta = root / "meta.json"
        if not meta.is_file():
            raise DataError(f"Missing synthetic dataset metadata: {meta}")
        return int(json.loads(meta.read_text(encoding="utf-8"))["images"])
    return PUBLISHED_COUNTS[config.dataset]


def discover_images(config, root: str | Path) -> tuple[list[str], dict[str, Path]]:
    """Returns sorted image IDs and their paths. Stops if the count is wrong."""
    root = Path(root)
    if not root.is_dir():
        raise DataError(f"Data folder does not exist: {root}")
    if config.dataset == "voc2007":
        images = _discover_voc(root, "2007", use_trainval_list=False)
    elif config.dataset == "voc2012":
        images = _discover_voc(root, "2012", use_trainval_list=True)
    elif config.dataset == "coco2014":
        images = _discover_coco(root)
    else:
        images = _discover_synthetic(root)

    expected = expected_count(config, root)
    if len(images) != expected:
        raise DataError(
            f"Found {len(images)} images for {config.dataset} under {root}, expected {expected}. "
            "Attach the complete dataset, or set expected_images if you are knowingly using another "
            "version. A changed count must be reported as a deviation from the paper."
        )
    ids = sorted(images)
    return ids, {image_id: images[image_id] for image_id in ids}
