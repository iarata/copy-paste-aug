from __future__ import annotations

from typing import Any

import cv2
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cpa.datasets import IMAGENET_MEAN, IMAGENET_STD


def _sine_position_encoding(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for sine/cosine positions, got {dim}.")

    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=device),
        torch.linspace(0, 1, width, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10_000 ** (omega / max(dim // 4 - 1, 1)))
    out_x = x.flatten()[:, None] * omega[None]
    out_y = y.flatten()[:, None] * omega[None]
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1).unsqueeze(0)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)
    return 1 - ((numerator + eps) / (denominator + eps))


def binary_mask_iou(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return pred_masks.new_zeros((pred_masks.shape[0], target_masks.shape[0]))
    pred = pred_masks.flatten(1).bool()
    target = target_masks.flatten(1).bool()
    intersection = (pred[:, None] & target[None]).sum(dim=-1).float()
    union = (pred[:, None] | target[None]).sum(dim=-1).float()
    return intersection / union.clamp_min(1.0)


def average_precision(scores: list[float], matches: list[int], total_targets: int) -> float:
    if total_targets == 0:
        return 0.0
    if not scores:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    true_positive = np.asarray(matches, dtype=np.float32)[order]
    false_positive = 1.0 - true_positive
    true_positive = np.cumsum(true_positive)
    false_positive = np.cumsum(false_positive)
    recall = true_positive / max(float(total_targets), 1.0)
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changing = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changing + 1] - recall[changing]) * precision[changing + 1]))


@torch.no_grad()
def evaluate_mask_map95(
    module: "SimpleInstanceSegmentationTransformerModule",
    dataloader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    module.eval()
    scores: list[float] = []
    matches: list[int] = []
    total_targets = 0
    device = module.device

    for batch in dataloader:
        images = batch["images"].to(device)
        outputs = module.model(images)
        pred_scores = outputs["object_logits"].sigmoid()
        pred_masks = outputs["mask_logits"].sigmoid() >= 0.5

        for sample_idx, target_masks in enumerate(batch["masks"]):
            target_masks = target_masks.to(device=device, dtype=torch.bool)
            total_targets += int(target_masks.shape[0])
            if target_masks.numel() == 0:
                for score in pred_scores[sample_idx].detach().cpu().tolist():
                    scores.append(float(score))
                    matches.append(0)
                continue

            resized_preds = F.interpolate(
                pred_masks[sample_idx].float()[:, None],
                size=target_masks.shape[-2:],
                mode="nearest",
            )[:, 0].bool()
            ious = binary_mask_iou(resized_preds, target_masks)
            used_targets: set[int] = set()
            order = torch.argsort(pred_scores[sample_idx], descending=True)
            for pred_index in order.tolist():
                score = float(pred_scores[sample_idx, pred_index].detach().cpu())
                scores.append(score)
                best_iou = 0.0
                best_target = -1
                for target_index in range(target_masks.shape[0]):
                    if target_index in used_targets:
                        continue
                    iou = float(ious[pred_index, target_index].detach().cpu())
                    if iou > best_iou:
                        best_iou = iou
                        best_target = target_index
                if best_iou >= 0.95 and best_target >= 0:
                    matches.append(1)
                    used_targets.add(best_target)
                else:
                    matches.append(0)

    map95 = average_precision(scores, matches, total_targets)
    return {
        "test/mAP95": map95,
        "benchmark/mAP95": map95,
        "test/instances": float(total_targets),
        "test/predictions": float(len(scores)),
    }


class SimpleInstanceSegmentationTransformer(nn.Module):
    """Small DETR-style mask transformer for quick copy-paste augmentation comparisons."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_queries: int = 20,
        patch_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=max(patch_size // 2, 1), padding=1),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.object_head = nn.Linear(embed_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        batch_size, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + _sine_position_encoding(height, width, channels, images.device)
        memory = self.encoder(tokens)

        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        mask_kernels = self.mask_embed(decoded)
        pixel_features = memory.transpose(1, 2).reshape(batch_size, channels, height, width)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_kernels, pixel_features) / (channels**0.5)
        object_logits = self.object_head(decoded).squeeze(-1)
        return {"mask_logits": mask_logits, "object_logits": object_logits}


class SimpleInstanceSegmentationTransformerModule(L.LightningModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_wandb_samples = int(getattr(cfg.models, "log_samples", 4))
        self._last_wandb_sample_epoch: dict[str, int] = {}
        self.model = SimpleInstanceSegmentationTransformer(
            embed_dim=int(getattr(cfg.models, "embed_dim", 128)),
            num_heads=int(getattr(cfg.models, "num_heads", 4)),
            encoder_layers=int(getattr(cfg.models, "encoder_layers", 2)),
            decoder_layers=int(getattr(cfg.models, "decoder_layers", 2)),
            num_queries=int(getattr(cfg.models, "num_queries", 20)),
            patch_size=int(getattr(cfg.models, "patch_size", 16)),
            dropout=float(getattr(cfg.models, "dropout", 0.1)),
        )
        self.save_hyperparameters(
            {
                "architecture": getattr(cfg.models, "architecture", "simple_instance_transformer"),
                "embed_dim": int(getattr(cfg.models, "embed_dim", 128)),
                "num_queries": int(getattr(cfg.models, "num_queries", 20)),
                "patch_size": int(getattr(cfg.models, "patch_size", 16)),
            }
        )

    def on_fit_start(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        experiment = self.logger.experiment
        experiment.define_metric("epoch")
        for pattern in ("train/*", "val/*", "samples/*", "lr-*"):
            experiment.define_metric(pattern, step_metric="epoch")

    def _resize_targets(self, masks: list[torch.Tensor], size: tuple[int, int]) -> list[torch.Tensor]:
        resized = []
        for sample_masks in masks:
            sample_masks = sample_masks.to(device=self.device, dtype=torch.float32)
            if sample_masks.numel() == 0:
                resized.append(sample_masks.reshape(0, *size))
                continue
            resized.append(F.interpolate(sample_masks[:, None], size=size, mode="nearest")[:, 0])
        return resized

    @torch.no_grad()
    def _match_predictions(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> list[tuple[int, int]]:
        if target_masks.numel() == 0:
            return []
        pred = pred_masks.sigmoid().flatten(1)
        target = target_masks.flatten(1)
        scores = (2 * pred @ target.T) / (
            pred.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0) + 1e-6
        )
        pairs: list[tuple[int, int]] = []
        used_pred: set[int] = set()
        used_target: set[int] = set()
        for _ in range(min(pred_masks.shape[0], target_masks.shape[0])):
            masked = scores.clone()
            if used_pred:
                masked[list(used_pred), :] = -1
            if used_target:
                masked[:, list(used_target)] = -1
            flat_index = int(masked.argmax().item())
            pred_index = flat_index // target_masks.shape[0]
            target_index = flat_index % target_masks.shape[0]
            if float(masked[pred_index, target_index]) < 0:
                break
            pairs.append((pred_index, target_index))
            used_pred.add(pred_index)
            used_target.add(target_index)
        return pairs

    def _compute_loss(
        self,
        batch: dict[str, Any],
        split: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        images = batch["images"].to(self.device)
        outputs = self.model(images)
        mask_logits = outputs["mask_logits"]
        object_logits = outputs["object_logits"]
        target_masks = self._resize_targets(batch["masks"], mask_logits.shape[-2:])

        object_targets = torch.zeros_like(object_logits)
        mask_losses = []
        dice_losses = []
        matched_count = 0

        for sample_idx, sample_targets in enumerate(target_masks):
            pairs = self._match_predictions(mask_logits[sample_idx].detach(), sample_targets)
            if not pairs:
                continue
            pred_indices = torch.as_tensor([p for p, _ in pairs], device=self.device, dtype=torch.long)
            target_indices = torch.as_tensor([t for _, t in pairs], device=self.device, dtype=torch.long)
            object_targets[sample_idx, pred_indices] = 1.0
            matched_pred = mask_logits[sample_idx, pred_indices]
            matched_target = sample_targets[target_indices]
            mask_losses.append(
                F.binary_cross_entropy_with_logits(matched_pred, matched_target, reduction="mean")
            )
            dice_losses.append(dice_loss(matched_pred, matched_target).mean())
            matched_count += len(pairs)

        object_loss = F.binary_cross_entropy_with_logits(object_logits, object_targets)
        zero = object_logits.sum() * 0.0
        mask_loss = torch.stack(mask_losses).mean() if mask_losses else zero
        mask_dice = torch.stack(dice_losses).mean() if dice_losses else zero
        loss = object_loss + mask_loss + mask_dice
        metrics = {
            f"{split}/loss": loss,
            f"{split}/object_loss": object_loss.detach(),
            f"{split}/mask_bce": mask_loss.detach(),
            f"{split}/mask_dice": mask_dice.detach(),
            f"{split}/matched_instances": torch.tensor(float(matched_count), device=self.device),
        }
        return loss, metrics, outputs

    def _log_metrics(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar="val/loss" in metrics,
            batch_size=batch_size,
            sync_dist=True,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics, outputs = self._compute_loss(batch, "train")
        self._log_metrics(metrics, batch["images"].shape[0])
        if batch_idx == 0:
            self._maybe_log_wandb_samples(batch, outputs, split="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics, outputs = self._compute_loss(batch, "val")
        self._log_metrics(metrics, batch["images"].shape[0])
        if batch_idx == 0:
            self._maybe_log_wandb_samples(batch, outputs, split="val")
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "epoch",
            float(self.current_epoch),
            on_step=False,
            on_epoch=True,
            logger=True,
            rank_zero_only=True,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(getattr(self.cfg.training, "lr0", 1e-4)),
            weight_decay=float(getattr(self.cfg.training, "weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(getattr(self.cfg.training, "epochs", 1)), 1),
            eta_min=float(getattr(self.cfg.training, "lr0", 1e-4))
            * float(getattr(self.cfg.training, "lrf", 0.01)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
        image = (image * std + mean).clamp(0, 1)
        return (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def _overlay_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        overlay = image.copy()
        if masks.size == 0:
            return overlay
        combined = masks.astype(bool).any(axis=0)
        overlay[combined] = (0.45 * overlay[combined] + 0.55 * np.asarray(color)).astype(np.uint8)
        return overlay

    def _render_sample_panel(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        sample_idx: int,
    ) -> np.ndarray:
        image = self._denormalize_image(batch["images"][sample_idx].to(self.device))
        height, width = image.shape[:2]
        target_masks = batch["masks"][sample_idx].detach().cpu().numpy()
        pred_scores = outputs["object_logits"][sample_idx].sigmoid().detach().cpu().numpy()
        pred_masks = outputs["mask_logits"][sample_idx].sigmoid().detach().cpu().numpy()
        keep = pred_scores >= float(getattr(self.cfg.models, "visualization_threshold", 0.5))
        pred_masks = pred_masks[keep] >= 0.5
        if pred_masks.size:
            pred_masks = np.stack(
                [
                    cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    for mask in pred_masks
                ]
            )
        else:
            pred_masks = np.zeros((0, height, width), dtype=np.uint8)

        target_panel = self._overlay_masks(image, target_masks, (39, 174, 96))
        pred_panel = self._overlay_masks(image, pred_masks, (235, 87, 87))
        cv2.putText(
            image,
            "input",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            target_panel,
            "targets",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            pred_panel,
            "predictions",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return np.concatenate([image, target_panel, pred_panel], axis=1)

    def _maybe_log_wandb_samples(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        split: str,
    ) -> None:
        if self._last_wandb_sample_epoch.get(split) == self.current_epoch:
            return
        if self.trainer is None or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        try:
            import wandb
        except ImportError:
            return

        count = min(int(batch["images"].shape[0]), self.max_wandb_samples)
        images = [
            wandb.Image(
                self._render_sample_panel(batch, outputs, sample_idx),
                caption=f"epoch={self.current_epoch} sample={sample_idx}",
            )
            for sample_idx in range(count)
        ]
        self.logger.experiment.log(
            {"epoch": int(self.current_epoch), f"samples/{split}_instance_transformer": images}
        )
        self._last_wandb_sample_epoch[split] = int(self.current_epoch)
