"""COCO 2017 instance-segmentation dataset with albumentations and a Lightning DataModule.

Pipeline
--------
Training
  1. Pre-paste spatial augmentations (random scale, pad, crop, flip, colour jitter)
     applied **independently** to both the base image and a randomly sampled paste image.
  2. Copy-Paste compositing (:class:`~cpa.augs.CopyPaste`).
  3. Post-paste normalisation (ImageNet stats).

Validation
  LongestMaxSize → PadIfNeeded → Normalize — no Copy-Paste.

Notes
-----
* Crowd annotations (``iscrowd=1``) are skipped; their segmentation is RLE, not
  polygon, and they should not be used as copy-paste candidates.
* ``pycocotools`` is **not** required: the JSON annotation file is parsed directly
  with :mod:`json`, and polygon masks are rasterised with :mod:`PIL`.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset

from cpa.augs import CopyPaste
from cpa.utils.configs import DatasetConfig
from cpa.utils.dataset_subset import subset_sequence, validate_subset_percent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _cfg_imgsz(cfg: DatasetConfig) -> int:
    return int(getattr(cfg, "image_size", getattr(cfg, "imgsz")))


def _cfg_root(cfg: DatasetConfig) -> str:
    return str(getattr(cfg, "data_path", getattr(cfg, "root")))


def _cfg_train_ann(cfg: DatasetConfig) -> str:
    return str(getattr(cfg, "train_ann_file", getattr(cfg, "train_json")))


def _cfg_val_ann(cfg: DatasetConfig) -> str:
    return str(getattr(cfg, "val_ann_file", getattr(cfg, "val_json")))


def _cfg_train_images(cfg: DatasetConfig) -> str:
    return str(getattr(cfg, "train_img_dir", getattr(cfg, "train_images")))


def _cfg_val_images(cfg: DatasetConfig) -> str:
    return str(getattr(cfg, "val_img_dir", getattr(cfg, "val_images")))


def _cfg_train_subset_percent(cfg: DatasetConfig) -> float:
    return float(getattr(cfg, "train_subset_percent", 100.0))


def _cfg_val_subset_percent(cfg: DatasetConfig) -> float:
    return float(getattr(cfg, "val_subset_percent", 100.0))

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def polygon_to_mask(polygon: list[float], height: int, width: int) -> np.ndarray:
    """Rasterise a flat COCO polygon ``[x0,y0,x1,y1,…]`` into a (H, W) uint8 mask.

    Args:
        polygon: Flat list of alternating x/y coordinates (COCO format).
        height:  Image height in pixels.
        width:   Image width in pixels.

    Returns:
        Binary ``np.ndarray`` of shape ``(H, W)`` and dtype ``uint8``.
    """
    canvas = Image.new("L", (width, height), 0)
    xy = list(zip(polygon[::2], polygon[1::2]))
    if len(xy) >= 3:
        ImageDraw.Draw(canvas).polygon(xy, outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def _img_to_tensor(img: Any) -> torch.Tensor:
    """Convert an image to a ``(C, H, W)`` float32 tensor.

    ``albumentationsx`` >= 2.1 already returns a ``(C, H, W)`` ``torch.Tensor``;
    plain ``albumentations`` returns an ``(H, W, C)`` ``numpy.ndarray``.
    Both cases are handled here so the collate function is forward-compatible.
    """
    if isinstance(img, torch.Tensor):
        # Already (C, H, W) — just ensure float32
        return img.float()
    # numpy (H, W, C) → (C, H, W) tensor
    return torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1).float()  # type: ignore[attr-defined]


def _masks_to_tensor(masks: Any) -> torch.Tensor:
    """Convert a masks array to an ``(N, H, W)`` float32 tensor."""
    if isinstance(masks, torch.Tensor):
        return masks.float()
    return torch.as_tensor(np.ascontiguousarray(masks)).float()  # type: ignore[attr-defined]


def mask_to_coco_bbox(mask: np.ndarray) -> list[float] | None:
    """Derive a valid COCO ``[x, y, width, height]`` bbox from a binary mask."""
    y_idx, x_idx = np.where(mask > 0)
    if x_idx.size == 0 or y_idx.size == 0:
        return None

    x1 = float(x_idx.min())
    y1 = float(y_idx.min())
    x2 = float(x_idx.max() + 1)
    y2 = float(y_idx.max() + 1)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [x1, y1, width, height]


def coco_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of COCO samples into a batched dict.

    * ``images``  – ``(B, 3, H, W)`` float32 tensor (already normalised).
    * ``masks``   – ``list[Tensor]``  one ``(N_i, H, W)`` float32 tensor per sample.
    * ``bboxes``  – ``list[list]``    one ``[[x,y,w,h], …]`` list per sample.

    Variable instance counts per sample make stacking masks/bboxes impossible;
    they are therefore kept as Python lists.

    Compatible with both ``albumentationsx`` (returns CHW torch tensors) and
    plain ``albumentations`` (returns HWC numpy arrays).
    """
    images = torch.stack([_img_to_tensor(s["image"]) for s in batch])
    masks = [_masks_to_tensor(s["masks"]) for s in batch]
    # albumentationsx returns bboxes as numpy arrays; normalise to plain lists
    bboxes = [
        s["bboxes"].tolist() if isinstance(s["bboxes"], np.ndarray) else list(s["bboxes"]) for s in batch
    ]
    return {"images": images, "masks": masks, "bboxes": bboxes}


# ─────────────────────────────────────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────────────────────────────────────


def _bbox_params(min_visibility: float = 0.1) -> A.BboxParams:
    """Return shared :class:`albumentations.BboxParams` for COCO bboxes."""
    return A.BboxParams(coord_format="coco", min_visibility=min_visibility)


def build_train_transforms(cfg: DatasetConfig) -> A.Compose:
    """Build the full training augmentation pipeline.

    The pipeline contains a :class:`~cpa.augs.CopyPaste` node. Inside
    :class:`Coco2017Dataset`, this pipeline is automatically split at that node
    so that all transforms *before* it are applied independently to both the
    base image and the randomly sampled paste image.

    Args:
        cfg: Dataset configuration (image size, augmentation hyper-parameters).

    Returns:
        An :class:`albumentations.Compose` ready to be consumed by the dataset.
    """
    aug = cfg.augmentations
    sz = _cfg_imgsz(cfg)

    if getattr(aug, "name", "cpa") in {"none", "disabled"}:
        return build_val_transforms(cfg)

    return A.Compose(
        [
            # ── Pre-paste: independent spatial augmentations ──────────────────
            A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
            A.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0, p=1.0),
            A.RandomCrop(height=sz, width=sz, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            # ── Copy-Paste compositing ────────────────────────────────────────
            CopyPaste(
                blend=aug.blend,
                sigma=aug.sigma,
                pct_objects_paste=aug.pct_objects_paste,
                max_paste_objects=aug.max_paste_objects,
                p=aug.prob,
            ),
            # ── Post-paste: colour normalisation ─────────────────────────────
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def build_val_transforms(cfg: DatasetConfig) -> A.Compose:
    """Build the validation / inference transform pipeline (no Copy-Paste).

    Args:
        cfg: Dataset configuration (image size).

    Returns:
        An :class:`albumentations.Compose` ready to be consumed by the dataset.
    """
    sz = _cfg_imgsz(cfg)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=sz),
            A.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class Coco2017Dataset(Dataset):
    """COCO 2017 instance-segmentation dataset with albumentations transforms.

    When ``training=True`` the supplied *transforms* pipeline **must** contain
    exactly one :class:`~cpa.augs.CopyPaste` transform.  The pipeline is split
    at that node on first access:

    * **Pre-paste** transforms (everything before ``CopyPaste``) are applied
      independently to both the base image and a randomly sampled paste image.
    * The ``CopyPaste`` transform composites paste objects onto the base image.
    * **Post-paste** transforms (everything after ``CopyPaste``) are applied
      to the composited result.

    When ``training=False`` the full pipeline is applied directly (no split,
    no second image).

    Args:
        root:       Root directory of the COCO dataset (e.g.
                    ``data.nosync/raw/coco2017/``).
        ann_file:   Path to the instance annotation JSON relative to *root*
                    (e.g. ``annotations/instances_train2017.json``).
        img_dir:    Sub-directory containing the JPEG images relative to *root*
                    (e.g. ``train2017``).
        transforms: Albumentations :class:`~albumentations.Compose` pipeline.
                    Use :func:`build_train_transforms` / :func:`build_val_transforms`.
        training:   Whether to enable the Copy-Paste split logic.

    Example::

        from cpa.datasets import Coco2017Dataset, build_train_transforms
        from cpa.utils.configs import DatasetConfig

        cfg = DatasetConfig()
        ds = Coco2017Dataset(
            root=cfg.data_path,
            ann_file=cfg.train_ann_file,
            img_dir=cfg.train_img_dir,
            transforms=build_train_transforms(cfg),
            training=True,
        )
        sample = ds[0]
        # sample["image"]  → (H, W, 3) float32 ndarray (normalised)
        # sample["masks"]  → (N, H, W) float32 ndarray
        # sample["bboxes"] → list of [x, y, w, h]
    """

    def __init__(
        self,
        root: str | Path,
        ann_file: str,
        img_dir: str,
        transforms: A.Compose,
        training: bool = False,
        subset_percent: float = 100.0,
        subset_seed: int = 0,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.img_dir = self.root / img_dir
        self._full_transforms = transforms
        self.training = training
        self.subset_percent = validate_subset_percent(subset_percent)
        self.subset_seed = int(subset_seed)

        # Lazily populated by _split_transforms()
        self._pre_tf: A.Compose | None = None
        self._cp_tf: A.Compose | None = None
        self._post_tf: A.Compose | None = None

        if training:
            self._split_transforms(transforms)

        # ── Load and index COCO JSON ──────────────────────────────────────────
        ann_path = self.root / ann_file
        with ann_path.open("r", encoding="utf-8") as fh:
            coco: dict = json.load(fh)

        images = subset_sequence(list(coco["images"]), self.subset_percent, self.subset_seed)

        # image_id → image metadata
        self._img_info: dict[int, dict] = {img["id"]: img for img in images}

        # image_id → list of non-crowd annotations
        ann_by_img: dict[int, list[dict]] = {img_id: [] for img_id in self._img_info}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue  # skip RLE crowd annotations
            img_id = ann["image_id"]
            if img_id in ann_by_img:
                ann_by_img[img_id].append(ann)
        self._ann_by_img = ann_by_img

        # Stable, ordered list of image IDs (index ↔ id mapping)
        self._image_ids: list[int] = list(self._img_info.keys())

    # ── Transform splitting ───────────────────────────────────────────────────

    def _split_transforms(self, pipeline: A.Compose) -> None:
        """Split *pipeline* at the first :class:`~cpa.augs.CopyPaste` node.

        After splitting:
        * ``self._pre_tf``  – pre-paste compose (spatial augmentations).
        * ``self._cp_tf``   – compose containing only the ``CopyPaste`` node.
        * ``self._post_tf`` – post-paste compose (normalisation, …).

        If no ``CopyPaste`` is found the three attributes remain ``None`` and
        ``__getitem__`` falls through to the full-pipeline path.
        """
        split_idx: int | None = None
        for i, tf in enumerate(pipeline.transforms):
            if isinstance(tf, CopyPaste):
                split_idx = i
                break

        if split_idx is None:
            return  # no CopyPaste node – use full pipeline directly

        tfs = list(pipeline.transforms)
        pre = tfs[:split_idx]
        cp = tfs[split_idx]
        post = tfs[split_idx + 1 :]

        # Reuse the same BboxParams from the original compose so that
        # filtering / coordinate conversion is consistent across all parts.
        bp_proc = pipeline.processors.get("bboxes")
        bp: A.BboxParams | None = bp_proc.params if bp_proc is not None else None  # type: ignore[assignment]

        self._pre_tf = A.Compose(pre, bbox_params=bp)
        # CopyPaste exposes paste_image / paste_masks via ``targets_as_params``;
        # they are consumed internally and must NOT appear in additional_targets.
        self._cp_tf = A.Compose([cp], bbox_params=bp)
        self._post_tf = A.Compose(post, bbox_params=bp)

    # ── Raw data loading ──────────────────────────────────────────────────────

    def _load_raw(self, idx: int) -> dict[str, Any]:
        """Load one COCO sample without applying any transforms.

        Returns:
            A dict with keys:

            * ``"image"``  – ``(H, W, 3)`` uint8 ndarray.
            * ``"masks"``  – ``(N, H, W)`` uint8 ndarray (may be empty: N=0).
            * ``"bboxes"`` – ``list`` of ``[x, y, w, h]`` (COCO pixel format).
        """
        image_id = self._image_ids[idx]
        info = self._img_info[image_id]
        h, w = info["height"], info["width"]

        # ── Image ─────────────────────────────────────────────────────────────
        img_path = self.img_dir / info["file_name"]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # ── Instance masks + bboxes ───────────────────────────────────────────
        masks_list: list[np.ndarray] = []
        bboxes: list[list[float]] = []

        for ann in self._ann_by_img.get(image_id, []):
            seg = ann.get("segmentation")
            # Only handle polygon format (list of lists)
            if not seg or not isinstance(seg, list):
                continue

            # Merge all polygon parts for this instance into one binary mask
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    inst_mask |= polygon_to_mask(poly, h, w)

            if inst_mask.max() == 0:
                continue  # skip degenerate / empty masks

            bbox = mask_to_coco_bbox(inst_mask)
            if bbox is None:
                continue

            masks_list.append(inst_mask)
            bboxes.append(bbox)

        masks = np.stack(masks_list, axis=0) if masks_list else np.empty((0, h, w), dtype=np.uint8)
        return {"image": image, "masks": masks, "bboxes": bboxes}

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one augmented sample.

        Training path (``training=True`` and a ``CopyPaste`` node was found):
          1. Pre-paste transforms on the base image.
          2. Pre-paste transforms on a randomly chosen paste image.
          3. ``CopyPaste`` compositing.
          4. Post-paste transforms (normalisation).

        Validation path:
          Full pipeline applied directly to the raw sample.

        Returns:
            Dict with keys ``"image"``, ``"masks"``, ``"bboxes"``
            (plus any additional keys inserted by the albumentations pipeline).
        """
        data = self._load_raw(idx)

        if self._cp_tf is not None and self._pre_tf is not None and self._post_tf is not None:
            # ── Training: pre → copy-paste → post ─────────────────────────
            # Apply pre-paste transforms to the base image
            data = self._pre_tf(**data)

            # Apply the same pre-paste transforms to a randomly sampled image
            paste_idx = random.randint(0, len(self) - 1)
            paste = self._pre_tf(**self._load_raw(paste_idx))

            # Composite paste objects onto the base image.
            # paste_image / paste_masks are consumed via CopyPaste.targets_as_params
            # and will not appear in the output dict.
            data = self._cp_tf(
                image=data["image"],
                masks=data["masks"],
                bboxes=data["bboxes"],
                paste_image=paste["image"],
                paste_masks=paste["masks"],
            )

            # Apply post-paste transforms (e.g. Normalize)
            data = self._post_tf(**data)

        else:
            # ── Validation / no Copy-Paste ─────────────────────────────────
            data = self._full_transforms(**data)

        return data

    def __len__(self) -> int:
        return len(self._image_ids)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_images={len(self)}, "
            f"img_dir={self.img_dir}, "
            f"training={self.training})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lightning DataModule
# ─────────────────────────────────────────────────────────────────────────────


class Coco2017DataModule(L.LightningDataModule):
    """Lightning :class:`~lightning.LightningDataModule` for COCO 2017.

    Wraps :class:`Coco2017Dataset` for training and validation splits and
    exposes standard :meth:`train_dataloader` / :meth:`val_dataloader` methods
    compatible with :class:`lightning.Trainer`.

    The data module is fully configured from a :class:`~cpa.utils.configs.DatasetConfig`
    instance produced by Hydra — no extra arguments required.

    Args:
        cfg: Hydra-populated dataset configuration dataclass.

    Example — standalone usage::

        from cpa.utils.configs import DatasetConfig
        from cpa.datasets import Coco2017DataModule

        dm = Coco2017DataModule(DatasetConfig())
        dm.setup("fit")

        for batch in dm.train_dataloader():
            images = batch["images"]   # (B, 3, H, W) float32 Tensor
            masks  = batch["masks"]    # list[Tensor]  (N_i, H, W) per sample
            bboxes = batch["bboxes"]   # list[list]    [[x,y,w,h], …] per sample
            break

    Example — with Hydra::

        import hydra
        from omegaconf import DictConfig, OmegaConf
        from cpa.datasets import Coco2017DataModule
        from cpa.utils.configs import Config, register_configs

        register_configs()

        @hydra.main(config_path="../configs", config_name="default", version_base=None)
        def main(raw_cfg: DictConfig) -> None:
            cfg: Config = OmegaConf.to_object(raw_cfg)
            dm = Coco2017DataModule(cfg.dataset)
            dm.setup("fit")
            ...
    """

    def __init__(self, cfg: DatasetConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)
        self.train_ds: Coco2017Dataset | None = None
        self.val_ds: Coco2017Dataset | None = None

    # ── LightningDataModule interface ─────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train / val datasets.

        Args:
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``
                   or ``None`` (sets up all splits).
        """
        cfg = self.cfg

        if stage in ("fit", None):
            self.train_ds = Coco2017Dataset(
                root=_cfg_root(cfg),
                ann_file=_cfg_train_ann(cfg),
                img_dir=_cfg_train_images(cfg),
                transforms=build_train_transforms(cfg),
                training=True,
                subset_percent=_cfg_train_subset_percent(cfg),
                subset_seed=self.seed,
            )

        if stage in ("fit", "validate", None):
            self.val_ds = Coco2017Dataset(
                root=_cfg_root(cfg),
                ann_file=_cfg_val_ann(cfg),
                img_dir=_cfg_val_images(cfg),
                transforms=build_val_transforms(cfg),
                training=False,
                subset_percent=_cfg_val_subset_percent(cfg),
                subset_seed=self.seed,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training :class:`~torch.utils.data.DataLoader`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called with
                ``stage="fit"`` or ``stage=None``.
        """
        if self.train_ds is None:
            raise RuntimeError("train_ds is not initialised — call setup('fit') or setup(None) first.")
        return self._build_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation :class:`~torch.utils.data.DataLoader`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called with
                ``stage="fit"``, ``stage="validate"`` or ``stage=None``.
        """
        if self.val_ds is None:
            raise RuntimeError(
                "val_ds is not initialised — call setup('fit'), setup('validate') or setup(None) first."
            )
        return self._build_loader(self.val_ds, shuffle=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_loader(self, ds: Coco2017Dataset, *, shuffle: bool) -> DataLoader:
        cfg = self.cfg
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            # persistent_workers requires num_workers > 0
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            collate_fn=coco_collate_fn,
        )

    def __repr__(self) -> str:
        train_n = len(self.train_ds) if self.train_ds else "?"
        val_n = len(self.val_ds) if self.val_ds else "?"
        return f"{self.__class__.__name__}(train={train_n}, val={val_n}, batch_size={self.cfg.batch_size})"
