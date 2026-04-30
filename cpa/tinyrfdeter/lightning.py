from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from cpa.tinyrfdeter.data import CocoPremadeDataModule, denormalize_image
from cpa.tinyrfdeter.model import (
    TinyRFDETRSegCriterion,
    Variant,
    box_cxcywh_to_xyxy,
    build_tinyrfdetrseg,
    config_for_variant,
)
from cpa.tinyrfdeter.muon import MuSGD

MAP_METRIC_KEYS = (
    ("bbox_map", "mAP50-95_box"),
    ("bbox_map_50", "mAP50_box"),
    ("bbox_map_75", "mAP75_box"),
    ("map", "mAP50-95_box"),
    ("map_50", "mAP50_box"),
    ("map_75", "mAP75_box"),
    ("segm_map", "mAP50-95_mask"),
    ("segm_map_50", "mAP50_mask"),
    ("segm_map_75", "mAP75_mask"),
)


def _tensor_outputs(outputs: dict[str, Any]) -> dict[str, Tensor]:
    return {key: value for key, value in outputs.items() if isinstance(value, Tensor)}


def _targets_to_device(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    return [
        {
            key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _numel(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _is_no_decay_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim < 2 or name.endswith(".bias") or "norm" in lowered or "pos_embed" in lowered


def build_musgd_optimizer(
    module: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    muon_scale: float,
    sgd_scale: float,
) -> tuple[MuSGD, dict[str, Any]]:
    muon_params: list[torch.nn.Parameter] = []
    sgd_decay_params: list[torch.nn.Parameter] = []
    sgd_no_decay_params: list[torch.nn.Parameter] = []

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim in {2, 4}:
            muon_params.append(parameter)
        elif _is_no_decay_parameter(name, parameter):
            sgd_no_decay_params.append(parameter)
        else:
            sgd_decay_params.append(parameter)

    common = {"lr": lr, "momentum": momentum, "nesterov": nesterov}
    param_groups: list[dict[str, Any]] = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                **common,
                "weight_decay": weight_decay,
                "use_muon": True,
                "param_group": "muon",
            }
        )
    if sgd_decay_params:
        param_groups.append(
            {
                "params": sgd_decay_params,
                **common,
                "weight_decay": weight_decay,
                "use_muon": False,
                "param_group": "sgd_decay",
            }
        )
    if sgd_no_decay_params:
        param_groups.append(
            {
                "params": sgd_no_decay_params,
                **common,
                "weight_decay": 0.0,
                "use_muon": False,
                "param_group": "sgd_no_decay",
            }
        )

    info = {
        "optimizer": "musgd",
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": nesterov,
        "musgd_muon_scale": muon_scale,
        "musgd_sgd_scale": sgd_scale,
        "musgd_muon_params": len(muon_params),
        "musgd_muon_numel": _numel(muon_params),
        "musgd_sgd_decay_params": len(sgd_decay_params),
        "musgd_sgd_decay_numel": _numel(sgd_decay_params),
        "musgd_sgd_no_decay_params": len(sgd_no_decay_params),
        "musgd_sgd_no_decay_numel": _numel(sgd_no_decay_params),
    }
    return MuSGD(param_groups, muon=muon_scale, sgd=sgd_scale), info


def format_optimizer_info(info: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in info.items())


def _mask_iou_matrix(pred_masks: Tensor, target_masks: Tensor) -> Tensor:
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return pred_masks.new_zeros((pred_masks.shape[0], target_masks.shape[0]), dtype=torch.float32)
    pred = pred_masks.flatten(1).float()
    target = target_masks.flatten(1).float()
    inter = pred @ target.transpose(0, 1)
    union = pred.sum(1)[:, None] + target.sum(1)[None] - inter
    return inter / union.clamp(min=1)


class TinyRFDETRSegLightning(L.LightningModule):
    def __init__(
        self,
        *,
        variant: Variant,
        num_classes: int,
        class_names: list[str],
        image_size: int | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        momentum: float = 0.95,
        nesterov: bool = True,
        musgd_muon_scale: float = 0.5,
        musgd_sgd_scale: float = 0.5,
        score_threshold: float = 0.25,
        log_images_every_n_epochs: int = 1,
        max_log_images: int = 4,
        max_log_instances: int = 20,
        map_top_k: int = 100,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_names"])
        self.class_names = class_names
        self.model = build_tinyrfdetrseg(
            variant,
            num_classes=num_classes,
            image_size=image_size,
            return_aux=return_aux,
        )
        self.criterion = TinyRFDETRSegCriterion(num_classes=num_classes)
        self._optimizer_info: dict[str, Any] = {}
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError as exc:
            raise ImportError("torchmetrics is required for validation mAP logging.") from exc

        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type=("bbox", "segm"),
            max_detection_thresholds=[1, 10, 100],
            class_metrics=False,
        )

    def forward(self, images: Tensor) -> dict[str, Any]:
        return self.model(images)

    def training_step(self, batch: tuple[Tensor, list[dict[str, Tensor]]], batch_idx: int) -> Tensor:
        images, targets = batch
        outputs = _tensor_outputs(self(images))
        losses = self.criterion(outputs, targets)
        loss = self.criterion.weighted_loss(losses)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
        self.log_dict(
            {f"train/{name}": value for name, value in losses.items()},
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, list[dict[str, Tensor]]], batch_idx: int) -> Tensor:
        images, targets = batch
        outputs = _tensor_outputs(self(images))
        losses = self.criterion(outputs, targets)
        loss = self.criterion.weighted_loss(losses)
        metrics = self._validation_metrics(outputs, targets, image_size=images.shape[-2:])

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=images.shape[0])
        self.log_dict(
            {f"val/{name}": value for name, value in losses.items()},
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log_dict(metrics, on_epoch=True, batch_size=images.shape[0])
        self._log_wandb_images(batch_idx, images, targets, outputs)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = str(self.hparams.optimizer).lower()
        lr = float(self.hparams.lr)
        weight_decay = float(self.hparams.weight_decay)
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            self._optimizer_info = {
                "optimizer": "adamw",
                "lr": lr,
                "weight_decay": weight_decay,
                "trainable_params": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        elif optimizer_name == "musgd":
            optimizer, self._optimizer_info = build_musgd_optimizer(
                self,
                lr=lr,
                weight_decay=weight_decay,
                momentum=float(self.hparams.momentum),
                nesterov=bool(self.hparams.nesterov),
                muon_scale=float(self.hparams.musgd_muon_scale),
                sgd_scale=float(self.hparams.musgd_sgd_scale),
            )
        else:
            raise ValueError(f"Unsupported optimizer {self.hparams.optimizer!r}. Use 'adamw' or 'musgd'.")

        print(f"TinyRFDETR optimizer: {format_optimizer_info(self._optimizer_info)}")
        return optimizer

    def on_fit_start(self) -> None:
        if not self._optimizer_info or self.logger is None:
            return
        values = {f"optimizer/{key}": value for key, value in self._optimizer_info.items()}
        self.logger.log_hyperparams(values)
        experiment = getattr(self.logger, "experiment", None)
        config = getattr(experiment, "config", None)
        if config is not None and hasattr(config, "update"):
            try:
                config.update(values, allow_val_change=True)
            except TypeError:
                config.update(values)

    @torch.no_grad()
    def _validation_metrics(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        *,
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        pred_masks = F.interpolate(
            outputs["pred_masks"],
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        indices = self.criterion.matcher(outputs, targets)

        matched_ious = []
        for batch_index, (src_idx, target_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            ious = _mask_iou_matrix(
                pred_masks[batch_index, src_idx] > 0, targets[batch_index]["masks"][target_idx]
            )
            matched_ious.append(torch.diag(ious))
        matched_iou = torch.cat(matched_ious).mean() if matched_ious else pred_masks.new_tensor(0.0)

        true_positive = pred_masks.new_tensor(0.0)
        false_positive = pred_masks.new_tensor(0.0)
        false_negative = pred_masks.new_tensor(0.0)
        prediction_count = pred_masks.new_tensor(0.0)
        scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)

        for batch_index, target in enumerate(targets):
            keep = scores[batch_index] >= float(self.hparams.score_threshold)
            order = torch.argsort(scores[batch_index][keep], descending=True)
            pred_indices = torch.where(keep)[0][order]
            prediction_count += pred_indices.numel()

            used_targets = torch.zeros(len(target["labels"]), dtype=torch.bool, device=self.device)
            for pred_index in pred_indices:
                same_class = target["labels"] == labels[batch_index, pred_index]
                available = same_class & ~used_targets
                if not available.any():
                    false_positive += 1
                    continue
                candidate_ids = torch.where(available)[0]
                ious = _mask_iou_matrix(
                    pred_masks[batch_index, pred_index : pred_index + 1] > 0,
                    target["masks"][candidate_ids],
                )[0]
                best_iou, best_offset = ious.max(dim=0)
                if best_iou >= 0.5:
                    true_positive += 1
                    used_targets[candidate_ids[best_offset]] = True
                else:
                    false_positive += 1
            false_negative += (~used_targets).sum()

        precision = true_positive / (true_positive + false_positive).clamp(min=1)
        recall = true_positive / (true_positive + false_negative).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
        return {
            "val/matched_mask_iou": matched_iou,
            "val/mask_precision50": precision,
            "val/mask_recall50": recall,
            "val/mask_f1_50": f1,
            "val/predictions_per_image": prediction_count / max(len(targets), 1),
        }

    @torch.no_grad()
    def _map_predictions(
        self, outputs: dict[str, Tensor], *, image_size: tuple[int, int]
    ) -> list[dict[str, Tensor]]:
        logits = outputs["pred_logits"]
        boxes = self._normalized_boxes_to_pixels(outputs["pred_boxes"], image_size)
        masks = (
            F.interpolate(outputs["pred_masks"], size=image_size, mode="bilinear", align_corners=False) > 0
        )
        batch_size, _, num_classes = logits.shape
        top_k = min(int(self.hparams.map_top_k), logits.shape[1] * num_classes)
        scores, indexes = logits.sigmoid().flatten(1).topk(top_k, dim=1)

        predictions = []
        for batch_index in range(batch_size):
            query_idx = indexes[batch_index] // num_classes
            labels = indexes[batch_index] % num_classes
            predictions.append(
                {
                    "boxes": boxes[batch_index, query_idx],
                    "scores": scores[batch_index],
                    "labels": labels,
                    "masks": masks[batch_index, query_idx],
                }
            )
        return predictions

    @torch.no_grad()
    def _map_targets(
        self, targets: list[dict[str, Tensor]], *, image_size: tuple[int, int]
    ) -> list[dict[str, Tensor]]:
        return [
            {
                "boxes": self._normalized_boxes_to_pixels(target["boxes"].unsqueeze(0), image_size)[0],
                "labels": target["labels"],
                "masks": target["masks"].bool(),
            }
            for target in targets
        ]

    def _normalized_boxes_to_pixels(self, boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
        height, width = image_size
        scale = boxes.new_tensor([width, height, width, height])
        return box_cxcywh_to_xyxy(boxes).clamp(0, 1) * scale

    @torch.no_grad()
    def update_map_metric(self, images: Tensor, targets: list[dict[str, Tensor]]) -> None:
        outputs = _tensor_outputs(self(images))
        self.map_metric.update(
            self._map_predictions(outputs, image_size=images.shape[-2:]),
            self._map_targets(targets, image_size=images.shape[-2:]),
        )

    def compute_map_metrics(self, *, prefix: str) -> dict[str, float]:
        metrics = self.map_metric.compute()
        self.map_metric.reset()
        logged = {}
        for source, target in MAP_METRIC_KEYS:
            value = metrics.get(source)
            metric_name = f"{prefix}/{target}"
            if value is not None and metric_name not in logged:
                logged[metric_name] = float(value.detach().cpu().item())
        return logged

    @torch.no_grad()
    def _log_wandb_images(
        self,
        batch_idx: int,
        images: Tensor,
        targets: list[dict[str, Tensor]],
        outputs: dict[str, Tensor],
    ) -> None:
        if batch_idx != 0:
            return
        every_n = int(self.hparams.log_images_every_n_epochs)
        if every_n <= 0 or self.current_epoch % every_n != 0:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return

        try:
            import wandb
        except ImportError:
            return

        class_labels = {0: "background"}
        class_labels.update({index + 1: name for index, name in enumerate(self.class_names)})

        max_images = min(int(self.hparams.max_log_images), images.shape[0])
        pred_masks = F.interpolate(
            outputs["pred_masks"],
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
        logged = []
        for batch_index in range(max_images):
            image_np = denormalize_image(images[batch_index])
            gt_mask = self._semantic_mask_from_targets(targets[batch_index])
            pred_mask = self._semantic_mask_from_predictions(
                pred_masks[batch_index],
                scores[batch_index],
                labels[batch_index],
            )
            logged.append(
                wandb.Image(
                    image_np,
                    masks={
                        "ground_truth": {"mask_data": gt_mask, "class_labels": class_labels},
                        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
                    },
                )
            )

        experiment.log({"val/examples": logged}, step=self.global_step)

    def _semantic_mask_from_targets(self, target: dict[str, Tensor]) -> np.ndarray:
        height, width = target["masks"].shape[-2:]
        semantic = np.zeros((height, width), dtype=np.int32)
        for mask, label in zip(target["masks"], target["labels"], strict=True):
            semantic[mask.detach().cpu().numpy().astype(bool)] = int(label.item()) + 1
        return semantic

    def _semantic_mask_from_predictions(self, masks: Tensor, scores: Tensor, labels: Tensor) -> np.ndarray:
        height, width = masks.shape[-2:]
        semantic = np.zeros((height, width), dtype=np.int32)
        keep = scores >= float(self.hparams.score_threshold)
        order = torch.argsort(scores[keep], descending=True)
        pred_indices = torch.where(keep)[0][order][: int(self.hparams.max_log_instances)]
        for pred_index in pred_indices:
            semantic[(masks[pred_index] > 0).detach().cpu().numpy()] = int(labels[pred_index].item()) + 1
        return semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny RF-DETR segmentation model on premade COCO datasets."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data.nosync/processed/coco2017_simple_cp_seed42_sub50")
    )
    parser.add_argument(
        "--val-data-root",
        type=Path,
        default=None,
        help="Original COCO root for validation. Auto-detects a sibling coco2017 directory when omitted.",
    )
    parser.add_argument(
        "--train-image-set",
        choices=["augmented", "original", "all"],
        default="augmented",
        help="Which premade train list to use from data-root/lists. Defaults to augmented-only.",
    )
    parser.add_argument("--variant", choices=["n", "s", "m"], default="n")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--sharing-strategy",
        choices=["file_system", "file_descriptor"],
        default="file_system",
        help="PyTorch multiprocessing tensor sharing strategy. file_system avoids low open-file limits.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "musgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--musgd-muon-scale", type=float, default=0.5)
    parser.add_argument("--musgd-sgd-scale", type=float, default=0.5)
    parser.add_argument("--train-subset-percent", type=float, default=100.0)
    parser.add_argument("--val-subset-percent", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinyrfdeter"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="copy-paste-aug")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--map-top-k", type=int, default=100)
    parser.add_argument("--final-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-map-limit-batches", type=_limit_batches, default=1.0)
    parser.add_argument("--log-images-every-n-epochs", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--limit-train-batches", type=_limit_batches, default=1.0)
    parser.add_argument("--limit-val-batches", type=_limit_batches, default=1.0)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def _limit_batches(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def _trainer_devices(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _limited_batch_count(limit: int | float, total_batches: int) -> int:
    if total_batches <= 0:
        return 0
    if isinstance(limit, int):
        return min(limit, total_batches)
    if limit <= 0:
        return 0
    if limit <= 1:
        return min(max(1, math.ceil(total_batches * limit)), total_batches)
    return min(int(limit), total_batches)


def _has_coco_val(root: Path) -> bool:
    return (root / "annotations" / "instances_val2017.json").exists() and (root / "val2017").exists()


def resolve_val_root(train_root: Path, explicit_val_root: Path | None) -> Path:
    if explicit_val_root is not None:
        return explicit_val_root

    candidates = [
        train_root.parent / "coco2017",
        train_root.parent.parent / "raw" / "coco2017",
        train_root,
    ]
    for candidate in candidates:
        if _has_coco_val(candidate):
            return candidate
    return train_root


def set_torch_sharing_strategy(strategy: str) -> None:
    available = torch.multiprocessing.get_all_sharing_strategies()
    if strategy not in available:
        raise ValueError(f"Sharing strategy {strategy!r} is unavailable. Available: {sorted(available)}")
    torch.multiprocessing.set_sharing_strategy(strategy)


@torch.no_grad()
def run_final_map_benchmark(
    *,
    trainer: L.Trainer,
    model: TinyRFDETRSegLightning,
    datamodule: CocoPremadeDataModule,
    limit_batches: int | float,
) -> dict[str, float]:
    dataloader = datamodule.val_dataloader()
    total_batches = len(dataloader)
    max_batches = _limited_batch_count(limit_batches, total_batches)
    if max_batches <= 0:
        raise ValueError("--final-map-limit-batches must select at least one validation batch.")

    print(f"Running final mAP benchmark on {max_batches}/{total_batches} validation batches.")
    was_training = model.training
    model.eval()
    model.map_metric.reset()
    device = model.device

    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = _targets_to_device(targets, device)
        model.update_map_metric(images, targets)

    metrics = model.compute_map_metrics(prefix="benchmark")
    if trainer.logger is not None:
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
    summary = " ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))
    print(f"Final mAP benchmark: {summary}")

    if was_training:
        model.train()
    return metrics


def main() -> None:
    args = parse_args()
    set_torch_sharing_strategy(args.sharing_strategy)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    cfg = config_for_variant(args.variant, image_size=args.image_size)
    val_root = resolve_val_root(args.data_root, args.val_data_root)
    dm = CocoPremadeDataModule(
        train_root=args.data_root,
        val_root=val_root,
        image_size=cfg.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_image_set=args.train_image_set,
        train_subset_percent=args.train_subset_percent,
        val_subset_percent=args.val_subset_percent,
        seed=args.seed,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    dm.setup("fit")
    train_count = len(dm.train_dataset) if dm.train_dataset is not None else 0
    val_count = len(dm.val_dataset) if dm.val_dataset is not None else 0
    print(
        "TinyRFDETR data: "
        f"train_root={dm.train_root} train_image_set={args.train_image_set} train_images={train_count} "
        f"val_root={dm.val_root} val_images={val_count} "
        f"workers={args.num_workers} pin_memory={args.pin_memory} "
        f"prefetch_factor={args.prefetch_factor} sharing_strategy={args.sharing_strategy}"
    )

    model = TinyRFDETRSegLightning(
        variant=args.variant,
        num_classes=dm.num_classes,
        class_names=dm.class_names,
        image_size=cfg.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov,
        musgd_muon_scale=args.musgd_muon_scale,
        musgd_sgd_scale=args.musgd_sgd_scale,
        score_threshold=args.score_threshold,
        log_images_every_n_epochs=args.log_images_every_n_epochs,
        map_top_k=args.map_top_k,
    )

    logger = None
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=str(args.output_dir),
            log_model=False,
        )

    checkpoint_dir = args.output_dir / f"rf-deter-seg-{args.variant}"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:03d}-valloss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=_trainer_devices(str(args.devices)),
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(args.output_dir),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)
    if args.final_map:
        run_final_map_benchmark(
            trainer=trainer,
            model=model,
            datamodule=dm,
            limit_batches=args.final_map_limit_batches,
        )


if __name__ == "__main__":
    main()
