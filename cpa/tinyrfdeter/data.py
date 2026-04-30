from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Keep torch imported before AlbumentationsX in this local stack; the opposite order can crash conv2d.
os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")
import albumentations as A
import lightning as L

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
Split = Literal["train", "val"]


def build_transforms(image_size: int, *, train: bool) -> A.Compose:
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.05, p=0.5),
            ]
        )
    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(transforms)


def denormalize_image(image: Tensor) -> np.ndarray:
    mean = image.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = image.new_tensor(IMAGENET_STD)[:, None, None]
    image = (image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def subset_sequence(items: Sequence[Any], percent: float, seed: int) -> list[Any]:
    if percent <= 0 or percent > 100:
        raise ValueError("subset percent must be in (0, 100]")
    items = list(items)
    if percent >= 100:
        return items
    count = max(1, round(len(items) * percent / 100.0))
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(items)), count))
    return [items[index] for index in selected]


def decode_coco_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if segmentation is None or segmentation == []:
        return np.zeros((height, width), dtype=np.uint8)

    from pycocotools import mask as mask_utils

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
    elif isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        rles = (
            mask_utils.frPyObjects(segmentation, height, width) if isinstance(counts, list) else segmentation
        )
    else:
        return np.zeros((height, width), dtype=np.uint8)

    mask = mask_utils.decode(rles)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    return mask.astype(np.uint8)


def masks_to_boxes_xyxy(masks: Tensor) -> Tensor:
    if masks.numel() == 0:
        return masks.new_zeros((0, 4), dtype=torch.float32)

    boxes = []
    for mask in masks:
        y, x = torch.where(mask)
        if x.numel() == 0:
            boxes.append(mask.new_tensor([0, 0, 0, 0], dtype=torch.float32))
            continue
        boxes.append(
            mask.new_tensor(
                [
                    x.min().float(),
                    y.min().float(),
                    x.max().float() + 1,
                    y.max().float() + 1,
                ],
                dtype=torch.float32,
            )
        )
    return torch.stack(boxes)


def masks_to_boxes_xyxy_np(masks: np.ndarray) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    for mask in masks:
        y, x = np.where(mask > 0)
        if x.size == 0:
            boxes.append([0, 0, 0, 0])
            continue
        boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
    return np.asarray(boxes, dtype=np.float32)


def xyxy_to_normalized_cxcywh(boxes: Tensor, height: int, width: int) -> Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / width
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / height
    box_w = (boxes[:, 2] - boxes[:, 0]) / width
    box_h = (boxes[:, 3] - boxes[:, 1]) / height
    return torch.stack((cx, cy, box_w, box_h), dim=1).clamp(0, 1)


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    """Resolve COCO image records without letting absolute JSON paths override data-root."""
    file_path = Path(file_name)
    candidates = [
        image_dir / file_path.name,
        image_dir / file_path,
        image_dir.parent / file_path,
    ]
    if file_path.is_absolute():
        candidates.append(file_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find image for COCO record "
        f"{file_name!r}. Tried:\n{tried}\n"
        "If this is a premade dataset copied without raw COCO2017, regenerate it with "
        "--no-cleanup-aliases or copy the raw images referenced by the annotation JSON."
    )


class CocoPremadeInstanceSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: Split,
        image_size: int,
        train: bool,
        subset_percent: float = 100.0,
        subset_seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.image_dir = self.root / f"{split}2017"
        self.annotation_file = self.root / "annotations" / f"instances_{split}2017.json"
        self.transforms = build_transforms(self.image_size, train=train)

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = sorted(coco["categories"], key=lambda category: category["id"])
        self.class_names = [category["name"] for category in categories]
        self.category_to_label = {category["id"]: index for index, category in enumerate(categories)}
        images = subset_sequence(coco["images"], subset_percent, subset_seed)
        self.images = {image["id"]: image for image in images}

        annotations_by_image = {image_id: [] for image_id in self.images}
        for annotation in coco["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            image_id = annotation["image_id"]
            if image_id in annotations_by_image:
                annotations_by_image[image_id].append(annotation)

        self.annotations_by_image = annotations_by_image
        self.image_ids = list(self.images)

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = self.image_ids[index]
        info = self.images[image_id]
        height = int(info["height"])
        width = int(info["width"])
        image_path = resolve_image_path(self.image_dir, str(info["file_name"]))
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        masks = []
        labels = []
        for annotation in self.annotations_by_image.get(image_id, []):
            category_id = annotation["category_id"]
            if category_id not in self.category_to_label:
                continue
            mask = decode_coco_mask(annotation.get("segmentation"), height, width)
            if mask.max() == 0:
                continue
            masks.append(mask)
            labels.append(self.category_to_label[category_id])

        mask_array = np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
        transformed = self.transforms(image=image, masks=mask_array)
        image_np = np.ascontiguousarray(transformed["image"])
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float()

        transformed_masks = transformed.get("masks", [])
        if len(transformed_masks) == 0:
            mask_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            mask_array = np.ascontiguousarray(transformed_masks).astype(np.uint8, copy=False)
            keep = mask_array.reshape(mask_array.shape[0], -1).any(axis=1)
            mask_array = np.ascontiguousarray(mask_array[keep])
            label_array = np.asarray(labels, dtype=np.int64)[keep]
            mask_tensor = torch.from_numpy(mask_array)
            label_tensor = torch.from_numpy(label_array)
            boxes = xyxy_to_normalized_cxcywh(
                torch.from_numpy(masks_to_boxes_xyxy_np(mask_array)),
                self.image_size,
                self.image_size,
            )

        target = {
            "boxes": boxes.float(),
            "labels": label_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.int64),
        }
        return image_tensor, target


def collate_instances(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return torch.stack(list(images)), list(targets)


class CocoPremadeDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int,
        batch_size: int,
        num_workers: int,
        train_subset_percent: float = 100.0,
        val_subset_percent: float = 100.0,
        seed: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_subset_percent = float(train_subset_percent)
        self.val_subset_percent = float(val_subset_percent)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.train_dataset: CocoPremadeInstanceSegDataset | None = None
        self.val_dataset: CocoPremadeInstanceSegDataset | None = None

    @property
    def class_names(self) -> list[str]:
        dataset = self.train_dataset or self.val_dataset
        if dataset is None:
            return []
        return dataset.class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"} and self.train_dataset is None:
            self.train_dataset = CocoPremadeInstanceSegDataset(
                self.root,
                split="train",
                image_size=self.image_size,
                train=True,
                subset_percent=self.train_subset_percent,
                subset_seed=self.seed,
            )
        if stage in {None, "fit", "validate"} and self.val_dataset is None:
            self.val_dataset = CocoPremadeInstanceSegDataset(
                self.root,
                split="val",
                image_size=self.image_size,
                train=False,
                subset_percent=self.val_subset_percent,
                subset_seed=self.seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the train dataloader.")
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('validate') or setup('fit') before requesting the val dataloader.")
        return self._build_loader(self.val_dataset, shuffle=False)

    def _build_loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_instances,
        )
