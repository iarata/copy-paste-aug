"""Build premade COCO2017 copy-paste datasets.

The generator writes a new COCO-style dataset root instead of modifying the
source COCO2017 tree.  The train split combines selected original images with
new augmented images.  Validation images and annotations are copied/linked
without augmentation.

The default ``simple`` method follows the core recipe from Ghiasi et al.,
"Simple Copy-Paste is a Strong Data Augmentation Method for Instance
Segmentation" (CVPR 2021): two images are independently scale-jittered and
flipped, a random subset of source objects is pasted onto the base image, and
the annotations are recomputed from the resulting masks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Protocol

import cv2
from loguru import logger
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
import yaml

from cpa.augs.copy_paste import image_copy_paste
from cpa.utils.dataset_subset import subset_indices, validate_subset_percent

logger.disable(__name__)


@dataclass(frozen=True)
class InstanceMask:
    """One object instance represented as a binary mask."""

    category_id: int
    mask: np.ndarray
    source_annotation_id: int | None = None


@dataclass(frozen=True)
class LoadedSample:
    """Image and instance masks after loading or geometric transforms."""

    image_id: int
    file_name: str
    image: np.ndarray
    instances: list[InstanceMask]


@dataclass(frozen=True)
class GeneratedImageRecord:
    """Manifest entry for one generated augmented image."""

    image_id: int
    file_name: str
    base_image_id: int
    paste_image_id: int
    paste_annotation_ids: list[int]


@dataclass(frozen=True)
class AugmentationTask:
    """Self-contained unit of offline augmentation work."""

    task_index: int
    image_id: int
    file_name: str
    base_image: dict[str, Any]
    paste_image: dict[str, Any]
    base_annotations: list[dict[str, Any]]
    paste_annotations: list[dict[str, Any]]
    rng_seed: int


@dataclass(frozen=True)
class AugmentationResult:
    """Result returned by one worker after writing an augmented image."""

    task_index: int
    image_record: dict[str, Any]
    annotations: list[dict[str, Any]]
    generated_record: GeneratedImageRecord
    file_name: str


@dataclass(frozen=True)
class PremadeCocoConfig:
    """Configuration for the premade COCO2017 dataset builder."""

    source_root: Path
    output_root: Path
    method: str = "simple"
    seed: int = 42
    train_subset_percent: float = 50.0
    val_subset_percent: float | None = None
    copy_paste_percent: float = 100.0
    augmented_per_image: int = 1
    objects_paste_percent: float = 50.0
    max_paste_objects: int | None = 10
    scale_min: float = 0.1
    scale_max: float = 2.0
    flip_prob: float = 0.5
    blend: bool = True
    sigma: float = 3.0
    link_mode: str = "symlink"
    workers: int = 1
    parallel_backend: str = "thread"
    show_progress: bool = True
    log_level: str = "INFO"
    overwrite: bool = False
    train_images: str = "train2017"
    val_images: str = "val2017"
    train_json: str = "annotations/instances_train2017.json"
    val_json: str = "annotations/instances_val2017.json"


class CopyPasteMethod(Protocol):
    """Interface for expandable offline copy-paste methods."""

    name: str

    def generate(
        self,
        *,
        base: LoadedSample,
        paste: LoadedSample,
        rng: np.random.Generator,
        config: PremadeCocoConfig,
    ) -> tuple[np.ndarray, list[InstanceMask], list[int]]:
        """Return ``(image, instances, selected_source_annotation_ids)``."""


class SimpleCopyPasteMethod:
    """Offline implementation of the simple random copy-paste method."""

    name = "simple"

    def generate(
        self,
        *,
        base: LoadedSample,
        paste: LoadedSample,
        rng: np.random.Generator,
        config: PremadeCocoConfig,
    ) -> tuple[np.ndarray, list[InstanceMask], list[int]]:
        target_h, target_w = base.image.shape[:2]
        base_aug = _scale_jitter_flip_to_canvas(
            base,
            target_h=target_h,
            target_w=target_w,
            rng=rng,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            flip_prob=config.flip_prob,
        )
        paste_aug = _scale_jitter_flip_to_canvas(
            paste,
            target_h=target_h,
            target_w=target_w,
            rng=rng,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            flip_prob=config.flip_prob,
        )

        if not paste_aug.instances:
            return base_aug.image, base_aug.instances, []

        selected_indices = _select_object_indices(
            len(paste_aug.instances),
            percent=config.objects_paste_percent,
            max_objects=config.max_paste_objects,
            rng=rng,
        )
        selected_paste_instances = [paste_aug.instances[index] for index in selected_indices]
        selected_ids = [
            instance.source_annotation_id
            for instance in selected_paste_instances
            if instance.source_annotation_id is not None
        ]

        alpha = np.any([instance.mask > 0 for instance in selected_paste_instances], axis=0).astype(np.uint8)
        image = image_copy_paste(
            base_aug.image,
            paste_aug.image,
            alpha,
            blend=config.blend,
            sigma=config.sigma,
        )

        output_instances: list[InstanceMask] = []
        alpha_bool = alpha.astype(bool)
        for instance in base_aug.instances:
            visible_mask = instance.mask.copy()
            visible_mask[alpha_bool] = 0
            if visible_mask.any():
                output_instances.append(
                    InstanceMask(
                        category_id=instance.category_id,
                        mask=visible_mask,
                        source_annotation_id=instance.source_annotation_id,
                    )
                )

        output_instances.extend(selected_paste_instances)
        return image, output_instances, selected_ids


METHOD_REGISTRY: dict[str, type[CopyPasteMethod]] = {
    SimpleCopyPasteMethod.name: SimpleCopyPasteMethod,
}


def build_premade_coco2017(config: PremadeCocoConfig) -> Path:
    """Create a separate COCO-style dataset root and return its path."""

    _validate_config(config)
    source_root = config.source_root
    output_root = config.output_root
    train_image_src = source_root / config.train_images
    val_image_src = source_root / config.val_images
    train_json_src = source_root / config.train_json
    val_json_src = source_root / config.val_json

    logger.info("Building premade COCO2017 dataset")
    logger.info("source_root={} output_root={}", source_root, output_root)

    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    train_image_out = output_root / config.train_images
    val_image_out = output_root / config.val_images
    annotations_out = output_root / "annotations"
    lists_out = output_root / "lists"
    for directory in (train_image_out, val_image_out, annotations_out, lists_out):
        directory.mkdir(parents=True, exist_ok=True)

    train_coco = _read_json(train_json_src)
    val_coco = _read_json(val_json_src)
    val_subset_percent = _effective_val_subset_percent(config)
    selected_train_images = select_category_balanced_images(
        train_coco,
        percent=config.train_subset_percent,
        seed=config.seed,
    )
    selected_val_images = select_category_balanced_images(
        val_coco,
        percent=val_subset_percent,
        seed=config.seed + 10_000,
    )
    logger.info(
        "Selected {} train images and {} val images",
        len(selected_train_images),
        len(selected_val_images),
    )

    train_ann_by_image = _annotations_by_image(train_coco)

    train_original_files = _materialize_original_images(
        images=selected_train_images,
        source_dir=train_image_src,
        output_dir=train_image_out,
        link_mode=config.link_mode,
        show_progress=config.show_progress,
        desc="train originals",
    )
    val_files = _materialize_original_images(
        images=selected_val_images,
        source_dir=val_image_src,
        output_dir=val_image_out,
        link_mode=config.link_mode,
        show_progress=config.show_progress,
        desc="val originals",
    )

    train_images_out = [dict(image) for image in selected_train_images]
    train_annotations_out, next_ann_id = _copy_annotations_for_images(
        train_coco,
        selected_train_images,
        start_ann_id=1,
    )
    next_image_id = _next_numeric_id(train_coco.get("images", []))

    augmentation_bases = _select_copy_paste_base_images(
        selected_train_images,
        percent=config.copy_paste_percent,
        seed=config.seed + 20_000,
    )
    generation_tasks = _build_generation_tasks(
        config=config,
        augmentation_bases=augmentation_bases,
        selected_train_images=selected_train_images,
        annotations_by_image=train_ann_by_image,
        first_image_id=next_image_id,
    )
    logger.info(
        "Generating {} augmented images with {} backend and {} worker(s)",
        len(generation_tasks),
        config.parallel_backend,
        config.workers,
    )

    generated_records: list[GeneratedImageRecord] = []
    generated_files: list[str] = []
    generation_results = _run_generation_tasks(
        tasks=generation_tasks,
        source_image_dir=train_image_src,
        output_image_dir=train_image_out,
        config=config,
    )
    for result in generation_results:
        train_images_out.append(result.image_record)
        generated_files.append(result.file_name)
        generated_records.append(result.generated_record)
        for annotation in result.annotations:
            annotation["id"] = next_ann_id
            train_annotations_out.append(annotation)
            next_ann_id += 1

    val_images_out = [dict(image) for image in selected_val_images]
    val_annotations_out, _ = _copy_annotations_for_images(
        val_coco,
        selected_val_images,
        start_ann_id=1,
    )

    _write_json(
        annotations_out / "instances_train2017.json",
        _coco_with_split(train_coco, train_images_out, train_annotations_out),
    )
    _write_json(
        annotations_out / "instances_val2017.json",
        _coco_with_split(val_coco, val_images_out, val_annotations_out),
    )
    _write_lists(
        lists_out,
        train_original=train_original_files,
        train_augmented=generated_files,
        val=val_files,
    )
    _write_dataset_yaml(output_root, train_coco["categories"])
    _write_manifest(
        output_root,
        config=config,
        selected_train_images=selected_train_images,
        selected_val_images=selected_val_images,
        generated_records=generated_records,
        effective_val_subset_percent=val_subset_percent,
    )
    logger.info("Wrote premade COCO dataset to {}", output_root)
    return output_root


def select_category_balanced_images(coco: dict[str, Any], percent: float, seed: int) -> list[dict[str, Any]]:
    """Select images so each category keeps at least ``percent`` image coverage.

    COCO images often contain multiple categories.  The selector therefore uses
    per-category image presence rather than annotation counts: for every class,
    it deterministically shuffles the images containing that class and adds
    images until the requested class target is met.  Multi-class images count
    toward all of their categories.
    """

    percent = validate_subset_percent(percent)
    images = list(coco.get("images", []))
    if percent >= 100.0:
        return images

    image_ids_in_order = [image["id"] for image in images]
    image_by_id = {image["id"]: image for image in images}
    class_to_images: dict[int, set[int]] = defaultdict(set)
    image_to_classes: dict[int, set[int]] = defaultdict(set)

    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in image_by_id or category_id is None:
            continue
        class_to_images[int(category_id)].add(image_id)
        image_to_classes[image_id].add(int(category_id))

    rng = np.random.default_rng(int(seed))
    selected: set[int] = set()
    selected_counts: dict[int, int] = defaultdict(int)

    for category in sorted(class_to_images):
        category_images = sorted(class_to_images[category])
        target = max(1, math.ceil(len(category_images) * percent / 100.0))
        current = selected_counts[category]
        if current >= target:
            continue

        shuffled = [category_images[index] for index in rng.permutation(len(category_images)).tolist()]
        for image_id in shuffled:
            if image_id in selected:
                continue
            selected.add(image_id)
            for image_category in image_to_classes[image_id]:
                selected_counts[image_category] += 1
            current = selected_counts[category]
            if current >= target:
                break

    return [image_by_id[image_id] for image_id in image_ids_in_order if image_id in selected]


def _validate_config(config: PremadeCocoConfig) -> None:
    validate_subset_percent(config.train_subset_percent)
    if config.val_subset_percent is not None:
        validate_subset_percent(config.val_subset_percent)
    validate_subset_percent(config.copy_paste_percent)
    validate_subset_percent(config.objects_paste_percent)
    if config.method not in METHOD_REGISTRY:
        supported = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unsupported copy-paste method {config.method!r}. Supported: {supported}")
    if config.augmented_per_image < 1:
        raise ValueError("augmented_per_image must be >= 1.")
    if config.max_paste_objects is not None and config.max_paste_objects < 1:
        raise ValueError("max_paste_objects must be >= 1 when set.")
    if config.scale_min <= 0 or config.scale_max <= 0 or config.scale_min > config.scale_max:
        raise ValueError("scale_min and scale_max must be positive and scale_min <= scale_max.")
    if not 0.0 <= config.flip_prob <= 1.0:
        raise ValueError("flip_prob must be in [0, 1].")
    if config.link_mode not in {"symlink", "copy", "hardlink"}:
        raise ValueError("link_mode must be one of: symlink, copy, hardlink.")
    if config.workers < 1:
        raise ValueError("workers must be >= 1.")
    if config.parallel_backend not in {"thread", "process"}:
        raise ValueError("parallel_backend must be one of: thread, process.")


def _effective_val_subset_percent(config: PremadeCocoConfig) -> float:
    """Use train subset size for val unless the caller overrides it."""

    if config.val_subset_percent is None:
        return config.train_subset_percent
    return config.val_subset_percent


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        by_image[int(annotation["image_id"])].append(annotation)
    return by_image


def _select_copy_paste_base_images(
    images: list[dict[str, Any]],
    *,
    percent: float,
    seed: int,
) -> list[dict[str, Any]]:
    indices = subset_indices(len(images), percent, seed)
    return [images[index] for index in indices]


def _build_generation_tasks(
    *,
    config: PremadeCocoConfig,
    augmentation_bases: list[dict[str, Any]],
    selected_train_images: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    first_image_id: int,
) -> list[AugmentationTask]:
    image_by_id = {int(image["id"]): image for image in selected_train_images}
    image_ids = [int(image["id"]) for image in selected_train_images]
    tasks: list[AugmentationTask] = []

    for base_image in augmentation_bases:
        base_image_id = int(base_image["id"])
        for aug_index in range(config.augmented_per_image):
            task_index = len(tasks)
            rng_seed = _stable_task_seed(config.seed, config.method, base_image_id, aug_index)
            paste_image = _choose_paste_image(
                image_by_id=image_by_id,
                image_ids=image_ids,
                base_image_id=base_image_id,
                rng=np.random.default_rng(rng_seed),
            )
            tasks.append(
                AugmentationTask(
                    task_index=task_index,
                    image_id=first_image_id + task_index,
                    file_name=_generated_file_name(config.method, config.seed, base_image_id, aug_index),
                    base_image=base_image,
                    paste_image=paste_image,
                    base_annotations=annotations_by_image.get(base_image_id, []),
                    paste_annotations=annotations_by_image.get(int(paste_image["id"]), []),
                    rng_seed=rng_seed,
                )
            )

    return tasks


def _run_generation_tasks(
    *,
    tasks: list[AugmentationTask],
    source_image_dir: Path,
    output_image_dir: Path,
    config: PremadeCocoConfig,
) -> list[AugmentationResult]:
    if not tasks:
        return []

    if config.workers == 1:
        results = [
            _generate_augmented_image_task(task, source_image_dir, output_image_dir, config)
            for task in tqdm(
                tasks,
                desc="copy-paste images",
                disable=not config.show_progress,
            )
        ]
        return sorted(results, key=lambda result: result.task_index)

    executor_cls = ThreadPoolExecutor if config.parallel_backend == "thread" else ProcessPoolExecutor
    results: list[AugmentationResult] = []
    with executor_cls(max_workers=config.workers) as executor:
        futures = [
            executor.submit(_generate_augmented_image_task, task, source_image_dir, output_image_dir, config)
            for task in tasks
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="copy-paste images",
            disable=not config.show_progress,
        ):
            results.append(future.result())

    return sorted(results, key=lambda result: result.task_index)


def _generate_augmented_image_task(
    task: AugmentationTask,
    source_image_dir: Path,
    output_image_dir: Path,
    config: PremadeCocoConfig,
) -> AugmentationResult:
    method = METHOD_REGISTRY[config.method]()
    rng = np.random.default_rng(task.rng_seed)
    base_sample = _load_sample_from_annotations(source_image_dir, task.base_image, task.base_annotations)
    paste_sample = _load_sample_from_annotations(source_image_dir, task.paste_image, task.paste_annotations)

    generated_image, generated_instances, paste_annotation_ids = method.generate(
        base=base_sample,
        paste=paste_sample,
        rng=rng,
        config=config,
    )

    Image.fromarray(generated_image).save(output_image_dir / task.file_name, quality=95)
    height, width = generated_image.shape[:2]
    annotations: list[dict[str, Any]] = []
    for instance in generated_instances:
        annotation = _annotation_from_mask(
            annotation_id=0,
            image_id=task.image_id,
            category_id=instance.category_id,
            mask=instance.mask,
        )
        if annotation is not None:
            annotations.append(annotation)

    image_record = {
        "id": task.image_id,
        "file_name": task.file_name,
        "height": height,
        "width": width,
    }
    generated_record = GeneratedImageRecord(
        image_id=task.image_id,
        file_name=task.file_name,
        base_image_id=int(task.base_image["id"]),
        paste_image_id=int(task.paste_image["id"]),
        paste_annotation_ids=paste_annotation_ids,
    )
    return AugmentationResult(
        task_index=task.task_index,
        image_record=image_record,
        annotations=annotations,
        generated_record=generated_record,
        file_name=task.file_name,
    )


def _load_sample_from_annotations(
    image_dir: Path,
    image_info: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> LoadedSample:
    image_path = image_dir / image_info["file_name"]
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    instances: list[InstanceMask] = []
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        mask = _segmentation_to_mask(annotation.get("segmentation"), height, width)
        if mask is None or not mask.any():
            continue
        instances.append(
            InstanceMask(
                category_id=int(annotation["category_id"]),
                mask=mask,
                source_annotation_id=int(annotation["id"]) if annotation.get("id") is not None else None,
            )
        )

    return LoadedSample(
        image_id=int(image_info["id"]),
        file_name=image_info["file_name"],
        image=image,
        instances=instances,
    )


def _stable_task_seed(global_seed: int, method: str, base_image_id: int, aug_index: int) -> int:
    payload = f"{global_seed}:{method}:{base_image_id}:{aug_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _materialize_original_images(
    *,
    images: list[dict[str, Any]],
    source_dir: Path,
    output_dir: Path,
    link_mode: str,
    show_progress: bool,
    desc: str,
) -> list[str]:
    files: list[str] = []
    for image in tqdm(images, desc=desc, disable=not show_progress):
        file_name = image["file_name"]
        source = source_dir / file_name
        destination = output_dir / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "copy":
            shutil.copy2(source, destination)
        elif link_mode == "hardlink":
            destination.hardlink_to(source)
        else:
            destination.symlink_to(source.resolve())
        files.append(file_name)
    return files


def _copy_annotations_for_images(
    coco: dict[str, Any],
    images: list[dict[str, Any]],
    *,
    start_ann_id: int,
) -> tuple[list[dict[str, Any]], int]:
    selected_ids = {image["id"] for image in images}
    annotations: list[dict[str, Any]] = []
    next_ann_id = start_ann_id
    for annotation in coco.get("annotations", []):
        if annotation.get("image_id") not in selected_ids:
            continue
        copied = dict(annotation)
        copied["id"] = next_ann_id
        annotations.append(copied)
        next_ann_id += 1
    return annotations, next_ann_id


def _next_numeric_id(images: Iterable[dict[str, Any]]) -> int:
    numeric_ids = [int(image["id"]) for image in images if isinstance(image.get("id"), int)]
    return (max(numeric_ids) + 1) if numeric_ids else 1


def _choose_paste_image(
    *,
    image_by_id: dict[int, dict[str, Any]],
    image_ids: list[int],
    base_image_id: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if len(image_ids) == 1:
        return image_by_id[image_ids[0]]

    paste_id = base_image_id
    while paste_id == base_image_id:
        paste_id = int(image_ids[int(rng.integers(0, len(image_ids)))])
    return image_by_id[paste_id]


def _select_object_indices(
    n_objects: int,
    *,
    percent: float,
    max_objects: int | None,
    rng: np.random.Generator,
) -> list[int]:
    if n_objects <= 0:
        return []
    n_select = max(1, math.ceil(n_objects * validate_subset_percent(percent) / 100.0))
    if max_objects is not None:
        n_select = min(n_select, int(max_objects))
    n_select = min(n_select, n_objects)
    return sorted(rng.choice(n_objects, size=n_select, replace=False).tolist())


def _scale_jitter_flip_to_canvas(
    sample: LoadedSample,
    *,
    target_h: int,
    target_w: int,
    rng: np.random.Generator,
    scale_min: float,
    scale_max: float,
    flip_prob: float,
) -> LoadedSample:
    """Scale-jitter one sample and crop/pad it back to a common canvas."""

    src_h, src_w = sample.image.shape[:2]
    scale = float(rng.uniform(scale_min, scale_max))
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    scaled_image = np.asarray(
        Image.fromarray(sample.image).resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    scaled_masks = [
        np.asarray(
            Image.fromarray(instance.mask).resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        for instance in sample.instances
    ]

    image_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    mask_canvases = [np.zeros((target_h, target_w), dtype=np.uint8) for _ in scaled_masks]

    src_x, dst_x, copy_w = _crop_or_pad_offsets(scaled_w, target_w, rng)
    src_y, dst_y, copy_h = _crop_or_pad_offsets(scaled_h, target_h, rng)
    image_canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = scaled_image[
        src_y : src_y + copy_h,
        src_x : src_x + copy_w,
    ]
    for canvas, mask in zip(mask_canvases, scaled_masks, strict=True):
        canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = mask[
            src_y : src_y + copy_h,
            src_x : src_x + copy_w,
        ]

    if rng.random() < flip_prob:
        image_canvas = np.ascontiguousarray(image_canvas[:, ::-1])
        mask_canvases = [np.ascontiguousarray(mask[:, ::-1]) for mask in mask_canvases]

    instances = [
        InstanceMask(
            category_id=instance.category_id,
            mask=mask,
            source_annotation_id=instance.source_annotation_id,
        )
        for instance, mask in zip(sample.instances, mask_canvases, strict=True)
        if mask.any()
    ]
    return LoadedSample(sample.image_id, sample.file_name, image_canvas, instances)


def _crop_or_pad_offsets(
    source_size: int,
    target_size: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if source_size >= target_size:
        src_start = int(rng.integers(0, source_size - target_size + 1))
        return src_start, 0, target_size
    dst_start = int(rng.integers(0, target_size - source_size + 1))
    return 0, dst_start, source_size


def _segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    if not segmentation:
        return None

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        decoded = mask_utils.decode(rle)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    else:
        return None

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def _annotation_from_mask(
    *,
    annotation_id: int,
    image_id: int,
    category_id: int,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    polygons = _mask_to_polygons(mask)
    if not polygons:
        return None
    bbox = _mask_to_bbox(mask)
    if bbox is None:
        return None
    area = float(mask.astype(bool).sum())
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": int(category_id),
        "segmentation": polygons,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        contour = contour.squeeze(1)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        if cv2.contourArea(contour.astype(np.float32)) <= 0:
            continue
        polygon = contour.astype(float).reshape(-1).tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def _mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [x1, y1, width, height]


def _generated_file_name(method: str, seed: int, base_image_id: int, aug_index: int) -> str:
    digest = hashlib.sha1(f"{method}:{seed}:{base_image_id}:{aug_index}".encode("utf-8")).hexdigest()[:10]
    return f"{method}_cp_seed{seed}_base{base_image_id:012d}_{aug_index:03d}_{digest}.jpg"


def _coco_with_split(
    source_coco: dict[str, Any],
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "info": source_coco.get("info", {}),
        "licenses": source_coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": source_coco.get("categories", []),
    }


def _write_lists(
    lists_dir: Path,
    *,
    train_original: list[str],
    train_augmented: list[str],
    val: list[str],
) -> None:
    _write_text_lines(lists_dir / "train_original.txt", train_original)
    _write_text_lines(lists_dir / "train_augmented.txt", train_augmented)
    _write_text_lines(lists_dir / "train_all.txt", [*train_original, *train_augmented])
    _write_text_lines(lists_dir / "val.txt", val)


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_dataset_yaml(output_root: Path, categories: list[dict[str, Any]]) -> None:
    names = {
        index: category["name"]
        for index, category in enumerate(sorted(categories, key=lambda category: category["id"]))
    }
    payload = {
        "path": str(output_root),
        "train": str(output_root / "train2017"),
        "val": str(output_root / "val2017"),
        "train_json": str(output_root / "annotations" / "instances_train2017.json"),
        "val_json": str(output_root / "annotations" / "instances_val2017.json"),
        "nc": len(names),
        "names": names,
        "channels": 3,
        "coco_eval": True,
    }
    (output_root / "coco_data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_manifest(
    output_root: Path,
    *,
    config: PremadeCocoConfig,
    selected_train_images: list[dict[str, Any]],
    selected_val_images: list[dict[str, Any]],
    generated_records: list[GeneratedImageRecord],
    effective_val_subset_percent: float,
) -> None:
    config_payload = asdict(config)
    for key in ("source_root", "output_root"):
        config_payload[key] = str(config_payload[key])
    payload = {
        "config": config_payload,
        "effective_val_subset_percent": effective_val_subset_percent,
        "selected_train_image_ids": [image["id"] for image in selected_train_images],
        "selected_val_image_ids": [image["id"] for image in selected_val_images],
        "generated_images": [asdict(record) for record in generated_records],
        "list_files": {
            "train_original": "lists/train_original.txt",
            "train_augmented": "lists/train_augmented.txt",
            "train_all": "lists/train_all.txt",
            "val": "lists/val.txt",
        },
    }
    _write_json(output_root / "manifest.json", payload)


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru for the command-line dataset builder."""

    logger.remove()
    logger.add(sys.stderr, level=level.upper(), enqueue=True)
    logger.enable(__name__)


def parse_args(argv: list[str] | None = None) -> PremadeCocoConfig:
    parser = argparse.ArgumentParser(description="Build a premade COCO2017 copy-paste dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("data.nosync/raw/coco2017"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_REGISTRY), default="simple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset-percent", type=float, default=50.0)
    parser.add_argument(
        "--val-subset-percent",
        type=float,
        default=None,
        help="Validation subset percentage. Defaults to --train-subset-percent when omitted.",
    )
    parser.add_argument("--copy-paste-percent", type=float, default=100.0)
    parser.add_argument("--augmented-per-image", type=int, default=1)
    parser.add_argument("--objects-paste-percent", type=float, default=50.0)
    parser.add_argument("--max-paste-objects", type=int, default=10)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--link-mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args(argv))
    args["show_progress"] = args.pop("progress")
    return PremadeCocoConfig(**args)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    configure_logging(config.log_level)
    output_root = build_premade_coco2017(config)
    print(f"Wrote premade COCO dataset to {output_root}")


if __name__ == "__main__":
    main()
