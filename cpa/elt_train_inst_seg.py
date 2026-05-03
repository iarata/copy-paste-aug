#!/usr/bin/env python3
"""ELT-style Mask R-CNN training on copy-paste COCO subsets.

This script trains only on an augmented COCO-style training subset and validates
on the original COCO2017 validation split.  It implements an ELT-inspired
weight-shared looped transformer refiner on top of the FPN features and an
Intra-Loop Self Distillation (ILSD) training objective adapted to instance
segmentation.

The paper that motivates this design is ELT: Elastic Looped Transformers for
Visual Generation.  That paper is not an instance-segmentation paper; the code
below applies its architectural/training ideas to a TorchVision Mask R-CNN
pipeline for the copy-paste ablation experiment described in the prompt.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import contextlib
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm.auto import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - older PyTorch fallback
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]


def make_grad_scaler(device: torch.device, enabled: bool) -> GradScaler:
    try:
        return GradScaler(device.type, enabled=enabled)
    except TypeError:  # pragma: no cover - older torch.cuda.amp.GradScaler
        return GradScaler(enabled=enabled)


@contextlib.contextmanager
def amp_autocast(device: torch.device, enabled: bool):
    try:
        with autocast(device_type=device.type, enabled=enabled):
            yield
    except TypeError:  # pragma: no cover - older torch.cuda.amp.autocast
        with autocast(enabled=enabled):
            yield


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def read_image_list(path: Path) -> set[str]:
    """Read list files written by the copy-paste dataset builder.

    The premade dataset can contain generated file names such as
    ``simple_cp_seed42_base....jpg`` and, depending on symlink cleanup settings,
    absolute paths for original images.  Matching therefore accepts both exact
    entries and basenames.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(line)
        names.add(str(Path(line)))
        names.add(Path(line).name)
    return names


def image_name_matches(file_name: str, allowed: set[str]) -> bool:
    return file_name in allowed or str(Path(file_name)) in allowed or Path(file_name).name in allowed


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    joined = image_dir / file_name
    if joined.is_file():
        return joined
    basename_joined = image_dir / candidate.name
    if basename_joined.is_file():
        return basename_joined
    raise FileNotFoundError(
        f"Could not resolve image '{file_name}'. Tried absolute path, {joined}, and {basename_joined}."
    )


class RandomHorizontalFlipForDetection:
    def __init__(self, p: float = 0.0) -> None:
        self.p = float(p)

    def __call__(self, image: Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if self.p <= 0.0 or random.random() >= self.p:
            return image, target
        _, _, width = image.shape
        image = image.flip(-1)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            xmin = width - boxes[:, 2]
            xmax = width - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
        target = dict(target)
        target["boxes"] = boxes
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        return image, target


class CocoInstanceSegmentation(Dataset):
    """COCO-style instance segmentation dataset for TorchVision detection models."""

    def __init__(
        self,
        image_dir: Path,
        ann_file: Path,
        image_list: Path | None = None,
        transform: RandomHorizontalFlipForDetection | None = None,
        allow_empty: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.ann_file = Path(ann_file)
        self.coco = COCO(str(self.ann_file))
        self.transform = transform
        self.allow_empty = allow_empty

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: ix + 1 for ix, cat_id in enumerate(self.cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}
        cats = {cat["id"]: cat for cat in self.coco.loadCats(self.cat_ids)}
        self.class_labels = {0: "background"}
        self.class_labels.update(
            {self.cat_id_to_label[cat_id]: cats[cat_id]["name"] for cat_id in self.cat_ids}
        )

        ids = sorted(self.coco.imgs.keys())
        if image_list is not None:
            allowed = read_image_list(image_list)
            ids = [
                img_id
                for img_id in ids
                if image_name_matches(str(self.coco.imgs[img_id]["file_name"]), allowed)
            ]
            if not ids:
                raise ValueError(
                    f"No COCO image records from {ann_file} matched list {image_list}. "
                    "Check that --train-list points to lists/train_augmented.txt for augmented-only training."
                )
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def get_height_and_width(self, idx: int) -> tuple[int, int]:
        image_info = self.coco.imgs[self.ids[idx]]
        return int(image_info["height"]), int(image_info["width"])

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = int(self.ids[idx])
        image_info = self.coco.loadImgs([image_id])[0]
        path = resolve_image_path(self.image_dir, str(image_info["file_name"]))
        image_pil = Image.open(path).convert("RGB")
        width, height = image_pil.size

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in anns:
            if ann.get("ignore", 0):
                continue
            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_label:
                continue
            x, y, bw, bh = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + max(0.0, bw))
            y2 = min(float(height), y + max(0.0, bh))
            if x2 <= x1 or y2 <= y1:
                continue
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if mask.shape != (height, width):
                mask = np.asarray(
                    Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8
                )
            if mask.max() == 0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[cat_id])
            masks.append(mask)
            areas.append(float(ann.get("area", float(mask.sum()))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        image = torch.as_tensor(np.asarray(image_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0

        if boxes:
            target: dict[str, Tensor] = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
                "image_id": torch.as_tensor([image_id], dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.uint8),
            }
        else:
            if not self.allow_empty:
                raise RuntimeError(f"Image {image_id} has no valid instances after filtering.")
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": torch.as_tensor([image_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.uint8),
            }

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


def detection_collate(
    batch: Sequence[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


# -----------------------------------------------------------------------------
# ELT-inspired looped feature refiner
# -----------------------------------------------------------------------------


class WindowTransformerBlock(nn.Module):
    """Local-window transformer block for one FPN level.

    The block is intentionally weight-shared across loops.  Windowed attention
    avoids quadratic attention over high-resolution COCO feature maps.
    """

    def __init__(
        self,
        channels: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if channels % nhead != 0:
            raise ValueError(f"channels={channels} must be divisible by nhead={nhead}")
        self.window_size = int(window_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.window_size * self.window_size, channels))
        self.layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = height + pad_h, width + pad_w

        # B,C,H,W -> B*n_windows, ws*ws, C
        windows = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch, hp // ws, ws, wp // ws, ws, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, ws * ws, channels)
        )
        windows = self.layer(windows + self.pos_embed)
        x = (
            windows.view(batch, hp // ws, wp // ws, ws, ws, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch, hp, wp, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return x[:, :, :height, :width]


class LoopedLevelRefiner(nn.Module):
    """Composite block of unique transformer layers reused for N loops."""

    def __init__(
        self,
        channels: int,
        unique_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        window_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                WindowTransformerBlock(
                    channels=channels,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    window_size=window_size,
                )
                for _ in range(unique_layers)
            ]
        )

    def forward(self, x: Tensor, loops: int) -> Tensor:
        for _ in range(int(loops)):
            for layer in self.layers:
                x = layer(x)
        return x


class ELTFPNBackbone(nn.Module):
    """Wrap a TorchVision FPN backbone with ELT-style recurrent refinement."""

    def __init__(
        self,
        base_backbone: nn.Module,
        max_loops: int = 4,
        levels: Sequence[str] = ("1", "2", "3", "pool"),
        channels: int = 256,
        unique_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        self.base_backbone = base_backbone
        self.out_channels = getattr(base_backbone, "out_channels", channels)
        self.loop_budget = int(max_loops)
        self.max_loops = int(max_loops)
        self.levels = tuple(str(level) for level in levels)
        self.refiners = nn.ModuleDict(
            {
                level: LoopedLevelRefiner(
                    channels=self.out_channels,
                    unique_layers=unique_layers,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    window_size=window_size,
                )
                for level in self.levels
            }
        )
        self.last_features: OrderedDict[str, Tensor] = OrderedDict()

    def set_loop_budget(self, loops: int) -> None:
        loops = int(loops)
        if loops < 0:
            raise ValueError("ELT loop budget must be non-negative.")
        self.loop_budget = loops

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        features = self.base_backbone(x)
        refined: OrderedDict[str, Tensor] = OrderedDict()
        for key, value in features.items():
            key_str = str(key)
            if self.loop_budget > 0 and key_str in self.refiners:
                refined[key] = self.refiners[key_str](value, self.loop_budget)
            else:
                refined[key] = value
        self.last_features = refined
        return refined


def parse_elt_levels(levels: str) -> tuple[str, ...]:
    levels = levels.strip()
    if levels.lower() == "all":
        return ("0", "1", "2", "3", "pool")
    return tuple(part.strip() for part in levels.split(",") if part.strip())


def set_model_loop_budget(model: nn.Module, loops: int) -> None:
    for module in model.modules():
        if isinstance(module, ELTFPNBackbone):
            module.set_loop_budget(loops)


def get_elt_features(model: nn.Module) -> OrderedDict[str, Tensor]:
    for module in model.modules():
        if isinstance(module, ELTFPNBackbone):
            return module.last_features
    return OrderedDict()


def feature_distillation_loss(student: Mapping[str, Tensor], teacher: Mapping[str, Tensor]) -> Tensor:
    losses: list[Tensor] = []
    for key, s in student.items():
        if key not in teacher:
            continue
        t = teacher[key].to(device=s.device, dtype=s.dtype)
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(F.normalize(s.float(), dim=1), F.normalize(t.float(), dim=1)))
    if not losses:
        device = next(iter(student.values())).device if student else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def replace_maskrcnn_heads(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    init = args.init.lower()
    weights = None
    weights_backbone = None
    if init == "coco":
        if args.model == "maskrcnn_resnet50_fpn_v2":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        else:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    elif init == "imagenet":
        weights_backbone = ResNet50_Weights.DEFAULT
    elif init == "none":
        weights = None
        weights_backbone = None
    else:
        raise ValueError("--init must be one of: none, imagenet, coco")

    builder = maskrcnn_resnet50_fpn_v2 if args.model == "maskrcnn_resnet50_fpn_v2" else maskrcnn_resnet50_fpn
    if weights is None:
        model = builder(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_size=args.min_size,
            max_size=args.max_size,
        )
    else:
        # COCO heads have the original COCO category layout. Replace them so the
        # contiguous category mapping from the dataset is used consistently.
        model = builder(
            weights=weights,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_size=args.min_size,
            max_size=args.max_size,
        )
        model = replace_maskrcnn_heads(model, num_classes)

    if args.use_elt:
        model.backbone = ELTFPNBackbone(
            model.backbone,
            max_loops=args.elt_max_loops,
            levels=parse_elt_levels(args.elt_levels),
            channels=getattr(model.backbone, "out_channels", 256),
            unique_layers=args.elt_unique_layers,
            nhead=args.elt_heads,
            dim_feedforward=args.elt_ffn_dim,
            dropout=args.elt_dropout,
            window_size=args.elt_window_size,
        )
    return model


# -----------------------------------------------------------------------------
# COCO evaluation and W&B media
# -----------------------------------------------------------------------------


COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def convert_predictions_to_coco(
    outputs: Sequence[dict[str, Tensor]],
    targets: Sequence[dict[str, Tensor]],
    label_to_cat_id: Mapping[int, int],
    score_threshold: float,
    mask_threshold: float,
    max_dets_per_image: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bbox_results: list[dict[str, Any]] = []
    segm_results: list[dict[str, Any]] = []

    for output, target in zip(outputs, targets):
        image_id = int(target["image_id"].item())
        if len(output.get("scores", [])) == 0:
            continue
        scores = output["scores"].detach().cpu()
        keep = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
        if keep.numel() == 0:
            continue
        keep = keep[:max_dets_per_image]

        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        masks = output.get("masks")
        masks_np = masks.detach().cpu().numpy()[:, 0] if masks is not None else None

        for det_idx in keep.tolist():
            label = int(labels[det_idx])
            if label not in label_to_cat_id:
                continue
            category_id = int(label_to_cat_id[label])
            score = float(scores[det_idx].item())
            bbox_xywh = xyxy_to_xywh(boxes[det_idx])
            if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                continue
            bbox_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox_xywh,
                    "score": score,
                }
            )
            if masks_np is not None:
                binary_mask = masks_np[det_idx] >= mask_threshold
                if binary_mask.any():
                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": encode_binary_mask(binary_mask),
                            "bbox": bbox_xywh,
                            "score": score,
                        }
                    )
    return bbox_results, segm_results


def run_coco_eval(
    coco_gt: COCO,
    results: list[dict[str, Any]],
    iou_type: str,
    img_ids: Sequence[int],
) -> dict[str, float]:
    prefix = f"{iou_type}/"
    if not results:
        return {prefix + name: 0.0 for name in COCO_STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.params.imgIds = list(img_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = {prefix + name: float(value) for name, value in zip(COCO_STAT_NAMES, evaluator.stats)}
    metrics[prefix + "summary"] = stdout.getvalue()
    return metrics


def tensor_image_to_numpy(image: Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def masks_to_label_map(masks: Tensor, labels: Tensor, height: int, width: int) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.int32)
    if masks.numel() == 0:
        return label_map
    for mask, label in zip(masks.detach().cpu(), labels.detach().cpu()):
        label_map[mask.numpy().astype(bool)] = int(label.item())
    return label_map


def predictions_to_label_map(
    output: dict[str, Tensor],
    height: int,
    width: int,
    score_threshold: float,
    mask_threshold: float,
    top_k: int,
) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.int32)
    if "masks" not in output or len(output.get("scores", [])) == 0:
        return label_map
    scores = output["scores"].detach().cpu()
    keep = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
    if keep.numel() == 0:
        return label_map
    keep = keep[:top_k]
    # Draw lower-confidence predictions first so higher-confidence predictions win overlaps.
    keep = keep[torch.argsort(scores[keep])]
    masks = output["masks"].detach().cpu().numpy()[:, 0]
    labels = output["labels"].detach().cpu().numpy()
    for det_idx in keep.tolist():
        mask = masks[det_idx] >= mask_threshold
        if mask.shape != (height, width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST),
                dtype=bool,
            )
        label_map[mask] = int(labels[det_idx])
    return label_map


def wandb_boxes_for_output(
    output: dict[str, Tensor],
    class_labels: Mapping[int, str],
    score_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    scores = output.get("scores", torch.empty(0)).detach().cpu()
    if scores.numel() == 0:
        return {"pred_boxes": {"box_data": [], "class_labels": dict(class_labels)}}
    keep = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()[:top_k]
    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    box_data: list[dict[str, Any]] = []
    for det_idx in keep.tolist():
        x1, y1, x2, y2 = [float(v) for v in boxes[det_idx]]
        label = int(labels[det_idx])
        score = float(scores[det_idx].item())
        name = class_labels.get(label, f"class_{label}")
        box_data.append(
            {
                "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                "class_id": label,
                "box_caption": f"{name} {score:.2f}",
                "scores": {"score": score},
                "domain": "pixel",
            }
        )
    return {"pred_boxes": {"box_data": box_data, "class_labels": dict(class_labels)}}


def make_wandb_image(
    image: Tensor,
    target: dict[str, Tensor],
    output: dict[str, Tensor],
    class_labels: Mapping[int, str],
    score_threshold: float,
    mask_threshold: float,
    top_k: int,
    caption: str,
) -> Any:
    import wandb

    image_np = tensor_image_to_numpy(image)
    height, width = image_np.shape[:2]
    gt_map = masks_to_label_map(target["masks"], target["labels"], height, width)
    pred_map = predictions_to_label_map(output, height, width, score_threshold, mask_threshold, top_k)
    return wandb.Image(
        image_np,
        masks={
            "predictions": {"mask_data": pred_map, "class_labels": dict(class_labels)},
            "ground_truth": {"mask_data": gt_map, "class_labels": dict(class_labels)},
        },
        boxes=wandb_boxes_for_output(output, class_labels, score_threshold, top_k),
        caption=caption,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    dataset: CocoInstanceSegmentation,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    loop_budget: int,
    wandb_run: Any | None = None,
) -> dict[str, float]:
    model.eval()
    set_model_loop_budget(model, loop_budget)

    bbox_results: list[dict[str, Any]] = []
    segm_results: list[dict[str, Any]] = []
    img_ids: list[int] = []
    logged_images: list[Any] = []

    progress = tqdm(data_loader, desc=f"val loops={loop_budget}", disable=args.no_progress)
    for batch_idx, (images, targets) in enumerate(progress):
        if args.limit_val_batches and batch_idx >= args.limit_val_batches:
            break
        images_device = [image.to(device, non_blocking=True) for image in images]
        outputs = model(images_device)
        outputs = [{k: v.detach().cpu() for k, v in out.items()} for out in outputs]
        targets_cpu = [{k: v.detach().cpu() for k, v in target.items()} for target in targets]
        batch_bbox, batch_segm = convert_predictions_to_coco(
            outputs,
            targets_cpu,
            dataset.label_to_cat_id,
            score_threshold=args.eval_score_threshold,
            mask_threshold=args.mask_threshold,
            max_dets_per_image=args.max_dets_per_image,
        )
        bbox_results.extend(batch_bbox)
        segm_results.extend(batch_segm)
        img_ids.extend(int(target["image_id"].item()) for target in targets_cpu)

        if wandb_run is not None and len(logged_images) < args.wandb_log_images:
            for image, target, output in zip(images, targets_cpu, outputs):
                if len(logged_images) >= args.wandb_log_images:
                    break
                logged_images.append(
                    make_wandb_image(
                        image=image,
                        target=target,
                        output=output,
                        class_labels=dataset.class_labels,
                        score_threshold=args.wandb_score_threshold,
                        mask_threshold=args.mask_threshold,
                        top_k=args.wandb_top_k,
                        caption=f"epoch={epoch} loops={loop_budget} image_id={int(target['image_id'].item())}",
                    )
                )

    unique_img_ids = sorted(set(img_ids)) if img_ids else list(dataset.ids)
    bbox_metrics = run_coco_eval(dataset.coco, bbox_results, "bbox", unique_img_ids)
    segm_metrics = run_coco_eval(dataset.coco, segm_results, "segm", unique_img_ids)
    metrics: dict[str, float] = {}
    for key, value in {**bbox_metrics, **segm_metrics}.items():
        if key.endswith("/summary"):
            continue
        metrics[f"val/loops_{loop_budget}/{key}"] = float(value)

    if wandb_run is not None:
        log_payload = dict(metrics)
        log_payload["epoch"] = epoch
        if logged_images:
            log_payload[f"val/loops_{loop_budget}/predictions"] = logged_images
        wandb_run.log(log_payload)
    return metrics


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def move_targets_to_device(
    targets: Sequence[dict[str, Tensor]], device: torch.device
) -> list[dict[str, Tensor]]:
    return [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]


def sum_loss_dict(losses: Mapping[str, Tensor]) -> Tensor:
    if not losses:
        raise RuntimeError("Model returned an empty loss dictionary.")
    return sum(loss for loss in losses.values())


def ilsd_lambda(global_step: int, total_steps: int, explicit_decay_steps: int) -> float:
    decay_steps = explicit_decay_steps if explicit_decay_steps > 0 else max(1, total_steps)
    return max(0.0, 1.0 - min(float(global_step) / float(decay_steps), 1.0))


def sample_student_loops(min_loops: int, max_loops: int) -> int:
    if max_loops <= min_loops:
        return max_loops
    return random.randint(min_loops, max_loops - 1)


def has_invalid_loss(loss: Tensor) -> bool:
    return not torch.isfinite(loss.detach()).all().item()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    total_steps: int,
    wandb_run: Any | None = None,
) -> int:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running: MutableMapping[str, list[float]] = defaultdict(list)
    progress = tqdm(data_loader, desc=f"train epoch={epoch}", disable=args.no_progress)
    accum_since_step = 0

    for batch_idx, (images, targets) in enumerate(progress):
        if args.limit_train_batches and batch_idx >= args.limit_train_batches:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        lambda_gt = ilsd_lambda(global_step, total_steps, args.ilsd_decay_steps)
        student_loops = sample_student_loops(args.elt_min_loops, args.elt_max_loops)

        with amp_autocast(device, enabled=args.amp):
            set_model_loop_budget(model, args.elt_max_loops)
            teacher_losses = model(images, targets)
            teacher_total = sum_loss_dict(teacher_losses)
            total = teacher_total
            distill = torch.zeros((), device=device)
            student_total = torch.zeros((), device=device)

            if args.use_elt and args.ilsd and args.elt_max_loops > args.elt_min_loops:
                teacher_features = {key: value.detach() for key, value in get_elt_features(model).items()}
                set_model_loop_budget(model, student_loops)
                student_losses = model(images, targets)
                student_total = sum_loss_dict(student_losses)
                student_features = get_elt_features(model)
                distill = feature_distillation_loss(student_features, teacher_features)
                total = (
                    teacher_total
                    + args.student_loss_weight * lambda_gt * student_total
                    + args.distill_weight * (1.0 - lambda_gt) * distill
                )
            else:
                student_losses = {}

        if has_invalid_loss(total):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, batch={batch_idx}: {float(total.detach())}"
            )

        scaled_total = total / args.grad_accum_steps
        scaler.scale(scaled_total).backward()
        accum_since_step += 1

        if accum_since_step >= args.grad_accum_steps:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_since_step = 0

        for key, value in teacher_losses.items():
            running[f"train/teacher/{key}"].append(float(value.detach().cpu()))
        for key, value in student_losses.items():
            running[f"train/student/{key}"].append(float(value.detach().cpu()))
        running["train/loss"].append(float(total.detach().cpu()))
        running["train/teacher/total"].append(float(teacher_total.detach().cpu()))
        running["train/student/total"].append(float(student_total.detach().cpu()))
        running["train/distill_features"].append(float(distill.detach().cpu()))
        running["train/ilsd_lambda"].append(lambda_gt)
        running["train/student_loops"].append(float(student_loops))

        if wandb_run is not None and global_step % args.log_every == 0:
            payload = {
                key: float(np.mean(values[-args.log_every :])) for key, values in running.items() if values
            }
            payload.update(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elt/teacher_loops": args.elt_max_loops,
                    "elt/student_loops": student_loops,
                }
            )
            wandb_run.log(payload)

        progress.set_postfix(
            loss=f"{float(total.detach().cpu()):.3f}",
            teacher=f"{float(teacher_total.detach().cpu()):.3f}",
            lambda_gt=f"{lambda_gt:.2f}",
        )
        global_step += 1

    # Flush leftover gradients when the epoch or a debug limit ends mid-accumulation.
    if accum_since_step > 0:
        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return global_step


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


@dataclass
class RunPaths:
    output_dir: Path
    last_ckpt: Path
    best_ckpt: Path


def make_run_paths(output_dir: Path) -> RunPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        output_dir=output_dir, last_ckpt=output_dir / "last.pt", best_ckpt=output_dir / "best_segm_ap.pt"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    args: argparse.Namespace,
    train_dataset: CocoInstanceSegmentation,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "args": vars(args),
        "label_to_cat_id": train_dataset.label_to_cat_id,
        "class_labels": train_dataset.class_labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_metric", -1.0)),
    )


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ELT-style instance segmentation model on COCO copy-paste subsets."
    )

    # Dataset paths.
    parser.add_argument("--train-root", type=Path, required=True, help="COCO-style augmented subset root.")
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument(
        "--train-list",
        default="auto",
        help="Path to lists/train_augmented.txt, 'auto', or 'none'. Used when --train-only=augmented.",
    )
    parser.add_argument("--train-only", choices=("augmented", "all"), default="augmented")
    parser.add_argument("--val-root", type=Path, required=True, help="Original COCO2017 root for validation.")
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)

    # Model and ELT.
    parser.add_argument(
        "--model",
        choices=("maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2"),
        default="maskrcnn_resnet50_fpn_v2",
    )
    parser.add_argument(
        "--init",
        choices=("none", "imagenet", "coco"),
        default="none",
        help="Avoid --init=coco for clean COCO learning studies.",
    )
    parser.add_argument("--trainable-backbone-layers", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--use-elt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elt-max-loops", type=int, default=4)
    parser.add_argument("--elt-min-loops", type=int, default=1)
    parser.add_argument("--elt-unique-layers", type=int, default=1)
    parser.add_argument(
        "--elt-levels", default="1,2,3,pool", help="FPN levels to refine: e.g. '1,2,3,pool' or 'all'."
    )
    parser.add_argument("--elt-window-size", type=int, default=7)
    parser.add_argument("--elt-heads", type=int, default=8)
    parser.add_argument("--elt-ffn-dim", type=int, default=1024)
    parser.add_argument("--elt-dropout", type=float, default=0.0)
    parser.add_argument("--ilsd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ilsd-decay-steps", type=int, default=0, help="0 means decay over the whole run.")
    parser.add_argument(
        "--eval-loop-budgets", default="4", help="Comma-separated loop budgets to evaluate, e.g. '1,2,4'."
    )

    # Optimization.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--betas", default="0.9,0.999")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hflip-prob", type=float, default=0.0, help="Keep 0 for clean copy-paste-only ablations."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    # Evaluation and logging.
    parser.add_argument("--output-dir", type=Path, default=Path("runs/elt_instance_seg"))
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-score-threshold", type=float, default=0.001)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-dets-per-image", type=int, default=100)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="elt-coco-copy-paste")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-images", type=int, default=8)
    parser.add_argument("--wandb-top-k", type=int, default=15)
    parser.add_argument("--wandb-score-threshold", type=float, default=0.3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation only. Use with --resume for a trained checkpoint.",
    )
    parser.add_argument("--limit-train-batches", type=int, default=0, help="Debug only. 0 means full epoch.")
    parser.add_argument(
        "--limit-val-batches", type=int, default=0, help="Debug only. 0 means full validation."
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    args.train_image_dir = args.train_image_dir or (args.train_root / "train2017")
    args.train_json = args.train_json or (args.train_root / "annotations" / "instances_train2017.json")
    args.val_image_dir = args.val_image_dir or (args.val_root / "val2017")
    args.val_json = args.val_json or (args.val_root / "annotations" / "instances_val2017.json")

    if args.train_only == "augmented":
        if args.train_list == "auto":
            args.train_list = str(args.train_root / "lists" / "train_augmented.txt")
        elif args.train_list.lower() == "none":
            raise ValueError("--train-only=augmented requires --train-list or --train-list=auto.")
        args.train_list = Path(args.train_list)
    else:
        args.train_list = None


def parse_betas(raw: str) -> tuple[float, float]:
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 2:
        raise ValueError("--betas must be formatted like '0.9,0.999'.")
    return values[0], values[1]


def parse_eval_loops(raw: str, default_loop: int) -> list[int]:
    loops = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return loops or [default_loop]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
    return torch.device(requested)


def create_wandb_run(
    args: argparse.Namespace, train_dataset: CocoInstanceSegmentation, val_dataset: CocoInstanceSegmentation
) -> Any | None:
    if not args.wandb:
        return None
    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    config = vars(args).copy()
    config.update(
        {
            "num_train_images": len(train_dataset),
            "num_val_images": len(val_dataset),
            "num_classes_including_background": len(train_dataset.class_labels),
            "class_labels": train_dataset.class_labels,
        }
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags,
        config=config,
    )
    return run


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    args.amp = bool(args.amp and device.type == "cuda")

    run_paths = make_run_paths(args.output_dir)
    (run_paths.output_dir / "resolved_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    train_dataset = CocoInstanceSegmentation(
        image_dir=args.train_image_dir,
        ann_file=args.train_json,
        image_list=args.train_list,
        transform=RandomHorizontalFlipForDetection(args.hflip_prob),
    )
    val_dataset = CocoInstanceSegmentation(
        image_dir=args.val_image_dir,
        ann_file=args.val_json,
        image_list=None,
        transform=None,
    )

    if train_dataset.cat_ids != val_dataset.cat_ids:
        raise ValueError(
            "Train/val COCO category ids differ. For this experiment, use augmented COCO train annotations "
            "and original COCO2017 validation annotations with the same categories."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        persistent_workers=args.num_workers > 0,
    )

    num_classes = len(train_dataset.cat_ids) + 1
    model = build_model(args, num_classes=num_classes).to(device)
    betas = parse_betas(args.betas)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
    scaler = make_grad_scaler(device, enabled=args.amp)

    start_epoch = 1
    global_step = 0
    best_metric = -1.0
    if args.resume is not None:
        start_epoch, global_step, best_metric = load_checkpoint(args.resume, model, optimizer, scaler, device)

    wandb_run = create_wandb_run(args, train_dataset, val_dataset)
    eval_loop_budgets = parse_eval_loops(args.eval_loop_budgets, args.elt_max_loops)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))

    print(
        json.dumps(
            {
                "train_images": len(train_dataset),
                "val_images_original_coco2017": len(val_dataset),
                "num_classes_including_background": num_classes,
                "device": str(device),
                "amp": args.amp,
                "train_list": str(args.train_list) if args.train_list else None,
                "eval_loop_budgets": eval_loop_budgets,
            },
            indent=2,
        )
    )

    if args.eval_only:
        eval_epoch = max(0, start_epoch - 1)
        for loop_budget in eval_loop_budgets:
            validate(
                model=model,
                data_loader=val_loader,
                dataset=val_dataset,
                device=device,
                args=args,
                epoch=eval_epoch,
                loop_budget=loop_budget,
                wandb_run=wandb_run,
            )
        if wandb_run is not None:
            wandb_run.finish()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            wandb_run=wandb_run,
        )

        val_metrics: dict[str, float] = {}
        if epoch % args.validate_every == 0:
            for loop_budget in eval_loop_budgets:
                metrics = validate(
                    model=model,
                    data_loader=val_loader,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    epoch=epoch,
                    loop_budget=loop_budget,
                    wandb_run=wandb_run,
                )
                val_metrics.update(metrics)

        primary_key = f"val/loops_{args.elt_max_loops}/segm/AP"
        if primary_key not in val_metrics and val_metrics:
            # Use the first evaluated segmentation AP if max loops was not evaluated.
            primary_key = next((key for key in val_metrics if key.endswith("/segm/AP")), primary_key)
        current_metric = val_metrics.get(primary_key, best_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                run_paths.best_ckpt,
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_metric,
                args,
                train_dataset,
            )
        save_checkpoint(
            run_paths.last_ckpt,
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            best_metric,
            args,
            train_dataset,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                run_paths.output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_metric,
                args,
                train_dataset,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "epoch_seconds": time.time() - epoch_start,
                    "best/segm_AP": best_metric,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
