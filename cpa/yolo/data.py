from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import cv2
import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ultralytics.data.augment import (
    Albumentations,
    BaseMixTransform,
    Compose,
    CutMix,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.augment import (
    CopyPaste as UltralyticsCopyPaste,
)
from ultralytics.data.converter import merge_multi_segment
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.utils import IterableSimpleNamespace, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.tqdm import TQDM
import yaml

from cpa.augs.copy_paste import extract_bboxes, image_copy_paste


def _cfg_to_dict(cfg: Any) -> Any:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: _cfg_to_dict(v) for k, v in vars(cfg).items()}
    if isinstance(cfg, dict):
        return {k: _cfg_to_dict(v) for k, v in cfg.items()}
    return cfg


def resolve_path(path: str | Path, base: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(base) / candidate


def load_coco_names(json_file: str | Path) -> dict[int, str]:
    with Path(json_file).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    categories = sorted(coco["categories"], key=lambda category: category["id"])
    return {index: category["name"] for index, category in enumerate(categories)}


def distributed_context() -> tuple[int, int] | None:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        try:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            rank = 0
            world_size = 1

    if world_size <= 1:
        return None
    return rank, world_size


class RectBatchDistributedSampler(DistributedSampler):
    """Distributed sampler that preserves Ultralytics rectangular batch groups.

    ``YOLODataset`` with ``rect=True`` computes one padded image shape per
    contiguous batch. PyTorch's default distributed sampler shards validation
    data by striding indices across ranks, which can put samples from different
    rectangular batches into the same dataloader batch. This sampler assigns
    whole rectangular batches to ranks and pads only with samples from the same
    rectangular batch.
    """

    def __init__(
        self,
        dataset: COCOJsonDataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        self.batch_size = max(int(batch_size), 1)
        self.source_batches = math.ceil(len(dataset) / self.batch_size)
        self.batches_per_rank = math.ceil(self.source_batches / self.num_replicas) if self.source_batches else 0
        self.num_samples = self.batches_per_rank * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.source_batches == 0:
            return iter([])

        batch_ids = list(range(self.source_batches))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            batch_ids = [batch_ids[index] for index in torch.randperm(len(batch_ids), generator=generator).tolist()]

        total_batches = self.batches_per_rank * self.num_replicas
        padding = total_batches - len(batch_ids)
        if padding > 0:
            batch_ids.extend(batch_ids[:padding])

        start = self.rank * self.batches_per_rank
        stop = start + self.batches_per_rank
        rank_batch_ids = batch_ids[start:stop]

        indices: list[int] = []
        dataset_size = len(self.dataset)
        for batch_id in rank_batch_ids:
            batch_start = batch_id * self.batch_size
            batch_stop = min(batch_start + self.batch_size, dataset_size)
            batch_indices = list(range(batch_start, batch_stop))
            if len(batch_indices) < self.batch_size:
                batch_padding = self.batch_size - len(batch_indices)
                repeats = math.ceil(batch_padding / len(batch_indices))
                batch_indices.extend((batch_indices * repeats)[:batch_padding])
            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def build_yolo_hyp(cfg: Any) -> IterableSimpleNamespace:
    aug = _cfg_to_dict(cfg)
    return IterableSimpleNamespace(
        mosaic=aug.get("mosaic", 1.0),
        mixup=aug.get("mixup", 0.0),
        cutmix=aug.get("cutmix", 0.0),
        degrees=aug.get("degrees", 0.0),
        translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.5),
        shear=aug.get("shear", 0.0),
        perspective=aug.get("perspective", 0.0),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        bgr=aug.get("bgr", 0.0),
        mask_ratio=4,
        overlap_mask=True,
        deterministic=False,
    )


def _segments_to_masks(segments: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(segments) == 0:
        return np.zeros((0, height, width), dtype=np.uint8)

    masks = np.zeros((len(segments), height, width), dtype=np.uint8)
    for index, segment in enumerate(segments):
        polygon = np.round(segment).astype(np.int32)
        if polygon.ndim != 2 or polygon.shape[0] < 3:
            continue
        cv2.fillPoly(masks[index], [polygon], 1)
    return masks


def _mask_to_segment(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).squeeze(1).astype(np.float32)
    if contour.ndim != 2 or len(contour) < 3:
        return None
    return contour


def _masks_to_instances(masks: np.ndarray, segment_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(masks) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, segment_points, 2), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    boxes = []
    segments = []
    keep = []

    for mask in masks:
        segment = _mask_to_segment(mask)
        if segment is None:
            keep.append(False)
            continue
        keep.append(True)
        segments.append(segment)

    keep_array = np.asarray(keep, dtype=bool)
    if not segments:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, segment_points, 2), dtype=np.float32),
            keep_array,
        )

    for x1, y1, x2, y2 in extract_bboxes(
        [mask for mask, keep_mask in zip(masks, keep_array, strict=False) if keep_mask]
    ):
        height, width = masks.shape[1:]
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])

    resampled = np.stack(resample_segments(segments, n=segment_points), axis=0).astype(np.float32)
    return np.asarray(boxes, dtype=np.float32), resampled, keep_array


class CPACopyPaste(BaseMixTransform):
    """Ultralytics-compatible wrapper around the repo's mask-based copy-paste implementation."""

    def __init__(
        self,
        dataset: YOLODataset,
        pre_transform: Compose | None = None,
        *,
        p: float = 0.5,
        mode: str = "mixup",
        blend: bool = True,
        sigma: float = 3.0,
        pct_objects_paste: float = 0.5,
        max_paste_objects: int | None = None,
    ) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        if mode not in {"flip", "mixup"}:
            raise ValueError(f"Unsupported custom copy-paste mode: {mode}")
        self.mode = mode
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if len(labels["instances"].segments) == 0 or random.uniform(0.0, 1.0) > self.p:
            return labels

        if self.mode == "flip":
            return self._transform(labels)

        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(index) for index in indexes]
        if self.pre_transform is not None:
            mix_labels = [self.pre_transform(sample) for sample in mix_labels]

        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._transform(labels, labels["mix_labels"][0])
        labels.pop("mix_labels", None)
        return labels

    def _select_indexes(self, count: int) -> np.ndarray:
        if count == 0:
            return np.zeros((0,), dtype=np.int64)

        if self.pct_objects_paste > 0:
            n_select = max(1, int(round(count * self.pct_objects_paste)))
        else:
            n_select = count

        if self.max_paste_objects is not None:
            n_select = min(n_select, self.max_paste_objects)

        n_select = min(n_select, count)
        return np.random.choice(count, size=n_select, replace=False)

    def _transform(self, labels1: dict[str, Any], labels2: dict[str, Any] | None = None) -> dict[str, Any]:
        image = labels1["img"]
        height, width = image.shape[:2]
        cls = labels1["cls"]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(width, height)
        segment_points = instances.segments.shape[1] if len(instances.segments) else 1000

        if labels2 is None:
            paste_image = cv2.flip(image, 1)
            paste_instances = Instances(
                bboxes=instances.bboxes.copy(),
                segments=instances.segments.copy(),
                bbox_format="xyxy",
                normalized=False,
            )
            paste_instances.fliplr(width)
            paste_cls = cls.copy()
        else:
            paste_image = labels2["img"]
            paste_instances = labels2.pop("instances")
            paste_instances.convert_bbox(format="xyxy")
            paste_instances.denormalize(width, height)
            paste_cls = labels2.get("cls", cls)

        if paste_image.shape[:2] != image.shape[:2] or len(paste_instances) == 0:
            labels1["img"] = image
            labels1["cls"] = cls
            labels1["instances"] = instances
            return labels1

        base_masks = _segments_to_masks(instances.segments, height, width)
        paste_masks = _segments_to_masks(paste_instances.segments, height, width)
        if len(paste_masks) == 0:
            labels1["img"] = image
            labels1["cls"] = cls
            labels1["instances"] = instances
            return labels1

        selected_indexes = self._select_indexes(len(paste_masks))
        selected_masks = paste_masks[selected_indexes]
        alpha = np.any(selected_masks > 0, axis=0).astype(np.uint8)
        updated_image = image_copy_paste(image, paste_image, alpha, blend=self.blend, sigma=self.sigma)

        adjusted_base_masks = np.where(alpha[None].astype(bool), 0, base_masks).astype(np.uint8)
        base_boxes, base_segments, keep_base = _masks_to_instances(adjusted_base_masks, segment_points)
        paste_boxes, paste_segments, keep_paste = _masks_to_instances(selected_masks, segment_points)

        kept_base_cls = cls[keep_base] if len(keep_base) else np.zeros((0, 1), dtype=np.float32)
        selected_cls = (
            paste_cls[selected_indexes] if len(selected_indexes) else np.zeros((0, 1), dtype=np.float32)
        )
        kept_paste_cls = selected_cls[keep_paste] if len(keep_paste) else np.zeros((0, 1), dtype=np.float32)

        if len(base_boxes) and len(paste_boxes):
            combined_boxes = np.concatenate([base_boxes, paste_boxes], axis=0)
            combined_segments = np.concatenate([base_segments, paste_segments], axis=0)
            combined_cls = np.concatenate([kept_base_cls, kept_paste_cls], axis=0)
        elif len(base_boxes):
            combined_boxes = base_boxes
            combined_segments = base_segments
            combined_cls = kept_base_cls
        elif len(paste_boxes):
            combined_boxes = paste_boxes
            combined_segments = paste_segments
            combined_cls = kept_paste_cls
        else:
            combined_boxes = np.zeros((0, 4), dtype=np.float32)
            combined_segments = np.zeros((0, segment_points, 2), dtype=np.float32)
            combined_cls = np.zeros((0, 1), dtype=np.float32)

        labels1["img"] = updated_image
        labels1["cls"] = combined_cls.astype(np.float32)
        labels1["instances"] = Instances(
            bboxes=combined_boxes,
            segments=combined_segments,
            bbox_format="xyxy",
            normalized=False,
        )
        return labels1


def build_train_transforms(
    dataset: YOLODataset, imgsz: int, hyp: IterableSimpleNamespace, aug_cfg: Any
) -> Compose:
    aug = _cfg_to_dict(aug_cfg)
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])

    name = aug.get("name", "none")
    copy_paste_transform = None
    if name == "ultralytics_flip":
        pre_transform.insert(1, UltralyticsCopyPaste(p=aug.get("prob", 0.5), mode="flip"))
    elif name == "ultralytics_mixup":
        copy_paste_transform = UltralyticsCopyPaste(
            dataset=dataset,
            pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
            p=aug.get("prob", 0.5),
            mode="mixup",
        )
    elif name not in {"none", "disabled"}:
        copy_paste_transform = CPACopyPaste(
            dataset=dataset,
            pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
            p=aug.get("prob", 0.5),
            mode=aug.get("mode", "mixup"),
            blend=aug.get("blend", True),
            sigma=aug.get("sigma", 3.0),
            pct_objects_paste=aug.get("pct_objects_paste", 0.5),
            max_paste_objects=aug.get("max_paste_objects"),
        )

    transforms = [pre_transform]
    if copy_paste_transform is not None:
        transforms.append(copy_paste_transform)

    transforms.extend(
        [
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr),
        ]
    )
    return Compose(transforms)


class COCOJsonDataset(YOLODataset):
    """YOLODataset that reads COCO JSON directly and supports configurable copy-paste variants."""

    def __init__(self, *args: Any, json_file: str, augmentation_cfg: Any, **kwargs: Any) -> None:
        self.json_file = str(json_file)
        self.augmentation_cfg = augmentation_cfg
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        return []

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        cache = {"labels": []}
        self.im_files = []

        with Path(self.json_file).open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = {
            category["id"]: index
            for index, category in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
        }
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        for image_info in TQDM(coco["images"], desc=f"reading {Path(self.json_file).name}"):
            height, width = image_info["height"], image_info["width"]
            image_file = Path(self.img_path) / image_info["file_name"]
            if not image_file.exists():
                continue

            self.im_files.append(str(image_file))
            classes = []
            boxes = []
            segments = []

            for annotation in annotations_by_image.get(image_info["id"], []):
                if annotation.get("iscrowd", False):
                    continue
                if annotation["category_id"] not in categories:
                    continue

                box = np.asarray(annotation["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= width
                box[[1, 3]] /= height
                if box[2] <= 0 or box[3] <= 0:
                    continue

                segmentation = annotation.get("segmentation")
                if segmentation is None or not isinstance(segmentation, list) or len(segmentation) == 0:
                    continue
                if len(segmentation) > 1:
                    merged = np.concatenate(merge_multi_segment(segmentation), axis=0)
                else:
                    merged = np.asarray(segmentation[0], dtype=np.float32).reshape(-1, 2)

                merged[:, 0] /= width
                merged[:, 1] /= height
                if len(merged) < 3:
                    continue

                classes.append([categories[annotation["category_id"]]])
                boxes.append(box.tolist())
                segments.append(merged.astype(np.float32))

            cls_array = (
                np.asarray(classes, dtype=np.float32) if classes else np.zeros((0, 1), dtype=np.float32)
            )
            box_array = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            cache["labels"].append(
                {
                    "im_file": str(image_file),
                    "shape": (height, width),
                    "cls": cls_array,
                    "bboxes": box_array,
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        cache["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, cache, DATASET_CACHE_VERSION)
        return cache

    def get_labels(self) -> list[dict[str, Any]]:
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [label["im_file"] for label in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)

        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]

    def build_transforms(self, hyp: dict[str, Any] | None = None) -> Compose:
        if self.augment:
            assert hyp is not None
            transforms = build_train_transforms(self, self.imgsz, hyp, self.augmentation_cfg)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio if hyp is not None else 4,
                mask_overlap=hyp.overlap_mask if hyp is not None else True,
                bgr=hyp.bgr if self.augment and hyp is not None else 0.0,
            )
        )
        return transforms


class COCOJsonDataModule(L.LightningDataModule):
    def __init__(self, cfg: Any, project_root: Path, eval_batch_size: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.project_root = project_root
        self.root = resolve_path(cfg.root, project_root)
        self.train_json = resolve_path(cfg.train_json, self.root)
        self.val_json = resolve_path(cfg.val_json, self.root)
        self.train_images = resolve_path(cfg.train_images, self.root)
        self.val_images = resolve_path(cfg.val_images, self.root)
        self.eval_batch_size = eval_batch_size or cfg.batch_size
        self.names = load_coco_names(self.train_json)
        self.data = {
            "path": str(self.root),
            "train": str(self.train_images),
            "val": str(self.val_images),
            "train_json": str(self.train_json),
            "val_json": str(self.val_json),
            "nc": len(self.names),
            "names": self.names,
            "channels": 3,
            "coco_eval": bool(getattr(cfg, "coco_eval", False)),
        }
        self.hyp = build_yolo_hyp(cfg.augmentations)
        self.train_dataset: COCOJsonDataset | None = None
        self.val_dataset: COCOJsonDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = COCOJsonDataset(
                img_path=str(self.train_images),
                json_file=str(self.train_json),
                imgsz=self.cfg.imgsz,
                cache=self.cfg.cache,
                augment=True,
                hyp=self.hyp,
                prefix=colorstr("train: "),
                rect=False,
                batch_size=self.cfg.batch_size,
                stride=32,
                pad=0.0,
                single_cls=False,
                classes=None,
                fraction=1.0,
                data=self.data,
                task=self.cfg.task,
                augmentation_cfg=self.cfg.augmentations,
            )

        if stage in {None, "fit", "validate"}:
            self.val_dataset = COCOJsonDataset(
                img_path=str(self.val_images),
                json_file=str(self.val_json),
                imgsz=self.cfg.imgsz,
                cache=self.cfg.cache,
                augment=False,
                hyp=self.hyp,
                prefix=colorstr("val: "),
                rect=True,
                batch_size=self.eval_batch_size,
                stride=32,
                pad=0.5,
                single_cls=False,
                classes=None,
                fraction=1.0,
                data=self.data,
                task=self.cfg.task,
                augmentation_cfg=self.cfg.augmentations,
            )

    def _build_loader(
        self,
        dataset: COCOJsonDataset,
        *,
        batch_size: int,
        shuffle: bool,
        distributed_rect: bool = False,
    ) -> DataLoader:
        num_workers = int(self.cfg.num_workers)
        persistent_workers = bool(self.cfg.persistent_workers and num_workers > 0)
        sampler = None
        if distributed_rect and bool(getattr(dataset, "rect", False)):
            context = distributed_context()
            if context is not None:
                rank, world_size = context
                sampler = RectBatchDistributedSampler(
                    dataset,
                    batch_size=batch_size,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=bool(self.cfg.pin_memory),
            persistent_workers=persistent_workers,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the training dataloader.")
        return self._build_loader(self.train_dataset, batch_size=int(self.cfg.batch_size), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Call setup('validate') or setup('fit') before requesting the validation dataloader."
            )
        return self._build_loader(
            self.val_dataset,
            batch_size=int(self.eval_batch_size),
            shuffle=False,
            distributed_rect=True,
        )

    def full_val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Call setup('validate') or setup('fit') before requesting the validation dataloader."
            )
        return self._build_loader(self.val_dataset, batch_size=int(self.eval_batch_size), shuffle=False)

    def write_data_yaml(self, output_file: str | Path) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "path": str(self.root),
            "train": str(self.train_images),
            "val": str(self.val_images),
            "train_json": str(self.train_json),
            "val_json": str(self.val_json),
            "nc": len(self.names),
            "names": self.names,
            "channels": 3,
            "coco_eval": bool(getattr(self.cfg, "coco_eval", False)),
        }
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return output_path
