from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

Variant = Literal["n", "s", "m"]


@dataclass(frozen=True)
class TinyRFDETRSegConfig:
    variant: Variant
    image_size: int
    patch_size: int = 12
    hidden_dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    num_queries: int = 100
    num_classes: int = 80
    mask_downsample_ratio: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    return_aux: bool = False


VARIANT_CONFIGS: dict[Variant, dict[str, int]] = {
    "n": {"image_size": 312, "encoder_layers": 4, "decoder_layers": 4, "num_queries": 100},
    "s": {"image_size": 384, "encoder_layers": 6, "decoder_layers": 4, "num_queries": 100},
    "m": {"image_size": 432, "encoder_layers": 8, "decoder_layers": 5, "num_queries": 200},
}


def config_for_variant(
    variant: Variant,
    *,
    num_classes: int = 80,
    image_size: int | None = None,
    return_aux: bool = False,
) -> TinyRFDETRSegConfig:
    values = dict(VARIANT_CONFIGS[variant])
    if image_size is not None:
        values["image_size"] = int(image_size)
    return TinyRFDETRSegConfig(
        variant=variant,
        num_classes=int(num_classes),
        return_aux=bool(return_aux),
        **values,
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.gelu(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class PatchBackbone(nn.Module):
    """Small ViT-style patch backbone plus the single P4 projector used by RF-DETR seg configs."""

    def __init__(self, cfg: TinyRFDETRSegConfig) -> None:
        super().__init__()
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.grid_size = cfg.image_size // cfg.patch_size
        self.hidden_dim = cfg.hidden_dim

        self.patch_embed = nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, cfg.hidden_dim))
        self.encoder = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.hidden_dim * cfg.mlp_ratio,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(cfg.encoder_layers)
        )
        self.projector = nn.Sequential(
            LayerNorm2d(cfg.hidden_dim),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            LayerNorm2d(cfg.hidden_dim),
        )

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _, _, height, width = images.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"TinyRFDETRSeg expects square {self.image_size}x{self.image_size} tensors, "
                f"got {height}x{width}."
            )

        spatial = self.patch_embed(images)
        tokens = spatial.flatten(2).transpose(1, 2)
        pos = self.pos_embed.expand(tokens.shape[0], -1, -1)
        tokens = tokens + pos
        for layer in self.encoder:
            tokens = layer(tokens)
        spatial = tokens.transpose(1, 2).reshape(
            images.shape[0],
            self.hidden_dim,
            self.grid_size,
            self.grid_size,
        )
        spatial = self.projector(spatial)
        memory = spatial.flatten(2).transpose(1, 2)
        return spatial, memory, pos


def sine_embed_for_boxes(boxes: Tensor, num_pos_feats: int) -> Tensor:
    scale = 2.0 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=boxes.dtype, device=boxes.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    parts = []
    for coord_index in (1, 0, 3, 2):
        coord = boxes[..., coord_index] * scale
        embed = coord[..., None] / dim_t
        embed = torch.stack((embed[..., 0::2].sin(), embed[..., 1::2].cos()), dim=-1).flatten(-2)
        parts.append(embed)
    return torch.cat(parts, dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TinyRFDETRSegConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.mlp_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * cfg.mlp_ratio, cfg.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.norm3 = nn.LayerNorm(cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, query: Tensor, memory: Tensor, memory_pos: Tensor, query_pos: Tensor) -> Tensor:
        query_with_pos = query + query_pos
        attended = self.self_attn(query_with_pos, query_with_pos, query, need_weights=False)[0]
        query = self.norm1(query + self.drop(attended))

        attended = self.cross_attn(query + query_pos, memory + memory_pos, memory, need_weights=False)[0]
        query = self.norm2(query + self.drop(attended))
        query = self.norm3(query + self.drop(self.ffn(query)))
        return query


class RFDETRDecoder(nn.Module):
    def __init__(self, cfg: TinyRFDETRSegConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for sine box embeddings")

        self.query_feat = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.refpoint_embed = nn.Embedding(cfg.num_queries, 4)
        self.ref_point_head = MLP(cfg.hidden_dim * 2, cfg.hidden_dim, cfg.hidden_dim, 2)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.decoder_layers))
        self.bbox_embed = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, 3)

        nn.init.constant_(self.refpoint_embed.weight, 0.0)
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0.0)

    def forward(self, memory: Tensor, memory_pos: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = memory.shape[0]
        query = self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1)
        ref_unsigmoid = self.refpoint_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        hs_layers = []
        box_layers = []
        for layer_index, layer in enumerate(self.layers):
            query_pos = self.ref_point_head(
                sine_embed_for_boxes(ref_unsigmoid.sigmoid(), memory.shape[-1] // 2)
            )
            query = layer(query, memory, memory_pos, query_pos)
            new_ref_unsigmoid = ref_unsigmoid + self.bbox_embed(query)
            hs_layers.append(query)
            box_layers.append(new_ref_unsigmoid.sigmoid())
            if layer_index < len(self.layers) - 1:
                ref_unsigmoid = new_ref_unsigmoid.detach()

        return torch.stack(hs_layers), torch.stack(box_layers)


class DepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.pwconv(self.norm(x)))
        x = x.permute(0, 3, 1, 2)
        return x + residual


class QueryBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(self.norm(x))


class SegmentationHead(nn.Module):
    """RF-DETR mask head: projected spatial features dotted with projected query features."""

    def __init__(self, cfg: TinyRFDETRSegConfig) -> None:
        super().__init__()
        self.downsample_ratio = cfg.mask_downsample_ratio
        self.blocks = nn.ModuleList(DepthwiseConvBlock(cfg.hidden_dim) for _ in range(cfg.decoder_layers))
        self.spatial_proj = nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1)
        self.query_block = QueryBlock(cfg.hidden_dim)
        self.query_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_features: Tensor, query_features: Tensor, image_size: int) -> Tensor:
        target_size = (image_size // self.downsample_ratio, image_size // self.downsample_ratio)
        spatial = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        masks = []
        for block, query in zip(self.blocks, query_features, strict=True):
            spatial = block(spatial)
            spatial_proj = self.spatial_proj(spatial)
            query_proj = self.query_proj(self.query_block(query))
            masks.append(torch.einsum("bchw,bqc->bqhw", spatial_proj, query_proj) + self.bias)
        return torch.stack(masks)


class TinyRFDETRSeg(nn.Module):
    def __init__(self, cfg: TinyRFDETRSegConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = PatchBackbone(cfg)
        self.decoder = RFDETRDecoder(cfg)
        self.class_embed = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.mask_head = SegmentationHead(cfg)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(self, images: Tensor) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        spatial, memory, memory_pos = self.backbone(images)
        hs_layers, box_layers = self.decoder(memory, memory_pos)
        class_layers = torch.stack([self.class_embed(layer) for layer in hs_layers])
        mask_layers = self.mask_head(spatial, hs_layers, self.cfg.image_size)

        output: dict[str, Tensor | list[dict[str, Tensor]]] = {
            "pred_logits": class_layers[-1],
            "pred_boxes": box_layers[-1],
            "pred_masks": mask_layers[-1],
        }
        if self.cfg.return_aux:
            output["aux_outputs"] = [
                {
                    "pred_logits": logits,
                    "pred_boxes": boxes,
                    "pred_masks": masks,
                }
                for logits, boxes, masks in zip(
                    class_layers[:-1], box_layers[:-1], mask_layers[:-1], strict=True
                )
            ]
        return output


def build_tinyrfdetrseg(
    variant: Variant = "n",
    *,
    num_classes: int = 80,
    image_size: int | None = None,
    return_aux: bool = False,
) -> TinyRFDETRSeg:
    return TinyRFDETRSeg(
        config_for_variant(
            variant,
            num_classes=num_classes,
            image_size=image_size,
            return_aux=return_aux,
        )
    )


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        dim=-1,
    )


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def pairwise_dice_cost(pred_logits: Tensor, target_masks: Tensor) -> Tensor:
    pred = pred_logits.sigmoid().flatten(1)
    target = target_masks.flatten(1).to(pred.dtype)
    numerator = 2 * pred @ target.transpose(0, 1)
    denominator = pred.sum(-1)[:, None] + target.sum(-1)[None]
    return 1 - (numerator + 1) / denominator.clamp(min=1).add(1)


def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, num_boxes: float, alpha: float = 0.25) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** 2)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean(-1).sum() / num_boxes


def dice_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1).to(inputs.dtype)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def downsample_target_masks(masks: Tensor, size: tuple[int, int]) -> Tensor:
    if masks.numel() == 0:
        return masks.new_zeros((0, *size), dtype=torch.float32)
    return F.interpolate(masks[:, None].float(), size=size, mode="nearest").squeeze(1)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        from scipy.optimize import linear_sum_assignment

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_masks = outputs["pred_masks"]
        indices = []

        for batch_index, target in enumerate(targets):
            labels = target["labels"]
            if labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            logits = pred_logits[batch_index]
            prob = logits.sigmoid()
            neg_cost = (1 - 0.25) * (prob**2) * (-F.logsigmoid(-logits))
            pos_cost = 0.25 * ((1 - prob) ** 2) * (-F.logsigmoid(logits))
            cost_class = pos_cost[:, labels] - neg_cost[:, labels]

            target_boxes = target["boxes"]
            cost_bbox = torch.cdist(pred_boxes[batch_index], target_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )

            target_masks = downsample_target_masks(target["masks"], pred_masks.shape[-2:])
            cost_mask = pairwise_dice_cost(pred_masks[batch_index], target_masks)

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
                + self.cost_mask * cost_mask
            )
            pred_idx, target_idx = linear_sum_assignment(cost.detach().cpu().float().numpy())
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(target_idx, dtype=torch.int64, device=pred_logits.device),
                )
            )
        return indices


class TinyRFDETRSegCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_dict = {
            "loss_cls": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 1.0,
        }

    def _src_permutation(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = float(sum(len(target["labels"]) for target in targets))
        num_boxes = max(num_boxes, 1.0)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        losses.update(self.loss_masks(outputs, targets, indices, num_boxes))
        return losses

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        logits = outputs["pred_logits"]
        target_classes = torch.zeros_like(logits)
        batch_idx, src_idx = self._src_permutation(indices)
        if src_idx.numel() > 0:
            labels = torch.cat(
                [target["labels"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
            )
            target_classes[batch_idx, src_idx, labels] = 1
        return {"loss_cls": sigmoid_focal_loss(logits, target_classes, num_boxes)}

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        batch_idx, src_idx = self._src_permutation(indices)
        if src_idx.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
        target_boxes = torch.cat(
            [target["boxes"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou.sum() / num_boxes}

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        pred_masks = outputs["pred_masks"]
        batch_idx, src_idx = self._src_permutation(indices)
        if src_idx.numel() == 0:
            zero = pred_masks.sum() * 0
            return {"loss_mask_bce": zero, "loss_mask_dice": zero}

        src_masks = pred_masks[batch_idx, src_idx]
        target_masks = torch.cat(
            [target["masks"][target_idx] for target, (_, target_idx) in zip(targets, indices)]
        )
        target_masks = downsample_target_masks(target_masks, src_masks.shape[-2:])
        loss_bce = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction="none")
        loss_bce = loss_bce.flatten(1).mean(1).sum() / num_boxes
        return {
            "loss_mask_bce": loss_bce,
            "loss_mask_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

    def weighted_loss(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(losses[name] * self.weight_dict[name] for name in self.weight_dict)


@torch.no_grad()
def postprocess_instances(
    outputs: dict[str, Tensor],
    *,
    score_threshold: float = 0.25,
    top_k: int = 20,
    mask_threshold: float = 0.0,
) -> list[dict[str, Tensor]]:
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    masks = outputs["pred_masks"]
    batch_size, _, num_classes = logits.shape
    top_k = min(top_k, logits.shape[1] * num_classes)

    results = []
    probs = logits.sigmoid()
    top_scores, top_indexes = probs.flatten(1).topk(top_k, dim=1)
    for batch_index in range(batch_size):
        query_idx = top_indexes[batch_index] // num_classes
        labels = top_indexes[batch_index] % num_classes
        keep = top_scores[batch_index] >= score_threshold
        query_idx = query_idx[keep]
        results.append(
            {
                "scores": top_scores[batch_index][keep],
                "labels": labels[keep],
                "boxes": boxes[batch_index, query_idx],
                "masks": masks[batch_index, query_idx] > mask_threshold,
            }
        )
    return results
