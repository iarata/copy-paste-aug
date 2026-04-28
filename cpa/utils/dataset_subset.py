from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def validate_subset_percent(percent: float) -> float:
    percent = float(percent)
    if not 0.0 < percent <= 100.0:
        raise ValueError(f"Subset percentage must be in (0, 100], got {percent}.")
    return percent


def subset_size(total: int, percent: float) -> int:
    if total <= 0:
        return 0

    percent = validate_subset_percent(percent)
    if percent >= 100.0:
        return total
    return max(1, min(total, math.ceil(total * percent / 100.0)))


def subset_indices(total: int, percent: float, seed: int) -> list[int]:
    size = subset_size(total, percent)
    if size == total:
        return list(range(total))

    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(total, size=size, replace=False).tolist())


def subset_sequence(items: list[Any], percent: float, seed: int) -> list[Any]:
    return [items[index] for index in subset_indices(len(items), percent, seed)]


def subset_coco(coco: dict[str, Any], percent: float, seed: int) -> dict[str, Any]:
    images = list(coco.get("images", []))
    selected_images = subset_sequence(images, percent, seed)
    selected_image_ids = {image["id"] for image in selected_images}
    annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if annotation.get("image_id") in selected_image_ids
    ]

    return {
        **coco,
        "images": selected_images,
        "annotations": annotations,
    }


def write_coco_subset_json(
    source_json: str | Path,
    output_json: str | Path,
    *,
    percent: float,
    seed: int,
) -> Path:
    source_path = Path(source_json)
    output_path = Path(output_json)

    with source_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(subset_coco(coco, percent, seed), separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path
