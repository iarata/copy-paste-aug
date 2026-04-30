from __future__ import annotations

import argparse
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
    build_tinyrfdetrseg,
    config_for_variant,
)


def _tensor_outputs(outputs: dict[str, Any]) -> dict[str, Tensor]:
    return {key: value for key, value in outputs.items() if isinstance(value, Tensor)}


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
        score_threshold: float = 0.25,
        log_images_every_n_epochs: int = 1,
        max_log_images: int = 4,
        max_log_instances: int = 20,
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )

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
    parser.add_argument("--variant", choices=["n", "s", "m"], default="n")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-subset-percent", type=float, default=100.0)
    parser.add_argument("--val-subset-percent", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinyrfdeter"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="copy-paste-aug")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
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


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    cfg = config_for_variant(args.variant, image_size=args.image_size)
    dm = CocoPremadeDataModule(
        args.data_root,
        image_size=cfg.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset_percent=args.train_subset_percent,
        val_subset_percent=args.val_subset_percent,
        seed=args.seed,
    )
    dm.setup("fit")

    model = TinyRFDETRSegLightning(
        variant=args.variant,
        num_classes=dm.num_classes,
        class_names=dm.class_names,
        image_size=cfg.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        score_threshold=args.score_threshold,
        log_images_every_n_epochs=args.log_images_every_n_epochs,
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


if __name__ == "__main__":
    main()
