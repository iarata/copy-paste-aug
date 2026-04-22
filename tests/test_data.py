"""Tests for COCO 2017 dataset helpers, transforms, Coco2017Dataset, and Coco2017DataModule.

All tests are self-contained – sample images and annotation JSON files are
created synthetically in a temporary directory so no real COCO data is needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from cpa.augs import CopyPaste
from cpa.datasets import (
    Coco2017DataModule,
    Coco2017Dataset,
    build_train_transforms,
    build_val_transforms,
    coco_collate_fn,
    polygon_to_mask,
)
from cpa.utils.configs import AugmentationsConfig, DatasetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Shared test constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_H, IMG_W = 128, 128  # synthetic source image dimensions
IMG_SIZE = 64  # output size used in all transform configs
N_TRAIN = 4
N_VAL = 2


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_coco_json(
    image_ids: list[int],
    img_h: int,
    img_w: int,
    anns_per_image: int = 2,
) -> dict:
    """Return a minimal COCO instances JSON dict with rectangular polygon annotations."""
    images = [
        {"id": img_id, "file_name": f"{img_id:012d}.jpg", "height": img_h, "width": img_w}
        for img_id in image_ids
    ]
    annotations = []
    ann_id = 1
    for img_id in image_ids:
        for i in range(anns_per_image):
            x, y, bw, bh = 10 + i * 20, 10 + i * 20, 30, 30
            # Rectangular polygon: 4 corners, 8 coordinate values
            seg = [
                [
                    float(x),
                    float(y),
                    float(x + bw),
                    float(y),
                    float(x + bw),
                    float(y + bh),
                    float(x),
                    float(y + bh),
                ]
            ]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": seg,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }


@pytest.fixture(scope="session")
def coco_root(tmp_path_factory) -> Path:
    """Create a minimal COCO-style directory tree with synthetic images + annotations.

    Re-used across all tests in the session to avoid redundant disk I/O.
    """
    root = tmp_path_factory.mktemp("coco2017")

    train_dir = root / "train2017"
    val_dir = root / "val2017"
    ann_dir = root / "annotations"
    train_dir.mkdir()
    val_dir.mkdir()
    ann_dir.mkdir()

    rng = np.random.default_rng(0)

    train_ids = list(range(1, N_TRAIN + 1))
    val_ids = list(range(N_TRAIN + 1, N_TRAIN + N_VAL + 1))

    for img_id in train_ids:
        arr = rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(train_dir / f"{img_id:012d}.jpg")

    for img_id in val_ids:
        arr = rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(val_dir / f"{img_id:012d}.jpg")

    (ann_dir / "instances_train2017.json").write_text(
        json.dumps(_make_coco_json(train_ids, IMG_H, IMG_W, anns_per_image=2))
    )
    (ann_dir / "instances_val2017.json").write_text(
        json.dumps(_make_coco_json(val_ids, IMG_H, IMG_W, anns_per_image=1))
    )

    return root


@pytest.fixture
def base_cfg(coco_root) -> DatasetConfig:
    """DatasetConfig pointing at the synthetic dataset with test-friendly settings."""
    return DatasetConfig(
        root=str(coco_root),
        imgsz=IMG_SIZE,
        batch_size=2,
        num_workers=0,  # avoid multiprocessing in tests
        pin_memory=False,
        persistent_workers=False,
        train_json="annotations/instances_train2017.json",
        val_json="annotations/instances_val2017.json",
        train_images="train2017",
        val_images="val2017",
        augmentations=AugmentationsConfig(
            prob=1.0,  # always apply CopyPaste for predictable tests
            blend=False,  # skip Gaussian blur → faster
            sigma=1.0,
            pct_objects_paste=1.0,  # paste all available objects
            max_paste_objects=5,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# polygon_to_mask
# ─────────────────────────────────────────────────────────────────────────────


class TestPolygonToMask:
    def test_returns_correct_shape(self):
        poly = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
        mask = polygon_to_mask(poly, height=64, width=64)
        assert mask.shape == (64, 64)

    def test_returns_uint8(self):
        poly = [0.0, 0.0, 32.0, 0.0, 32.0, 32.0, 0.0, 32.0]
        mask = polygon_to_mask(poly, height=64, width=64)
        assert mask.dtype == np.uint8

    def test_interior_pixel_is_one(self):
        """A pixel strictly inside a large square polygon should be set to 1."""
        poly = [5.0, 5.0, 55.0, 5.0, 55.0, 55.0, 5.0, 55.0]
        mask = polygon_to_mask(poly, height=64, width=64)
        assert mask[30, 30] == 1

    def test_corner_pixels_are_zero(self):
        """Corners of the image outside a centred polygon must remain 0."""
        poly = [10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]
        mask = polygon_to_mask(poly, height=64, width=64)
        assert mask[0, 0] == 0
        assert mask[63, 63] == 0

    def test_degenerate_polygon_returns_zero_mask(self):
        """A polygon with fewer than 3 vertices should produce an all-zero mask."""
        poly = [10.0, 10.0, 20.0, 20.0]  # only 2 points
        mask = polygon_to_mask(poly, height=64, width=64)
        assert mask.max() == 0

    def test_full_image_polygon_has_nonzero_pixels(self):
        H, W = 32, 32
        poly = [0.0, 0.0, float(W), 0.0, float(W), float(H), 0.0, float(H)]
        mask = polygon_to_mask(poly, height=H, width=W)
        assert mask.sum() > 0

    def test_values_are_binary(self):
        poly = [5.0, 5.0, 55.0, 5.0, 55.0, 55.0, 5.0, 55.0]
        mask = polygon_to_mask(poly, height=64, width=64)
        unique = set(np.unique(mask).tolist())
        assert unique.issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# coco_collate_fn
# ─────────────────────────────────────────────────────────────────────────────


class TestCocoCollateFn:
    @staticmethod
    def _sample(n_instances: int = 2) -> dict:
        return {
            "image": np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32),
            "masks": np.zeros((n_instances, IMG_SIZE, IMG_SIZE), dtype=np.float32),
            "bboxes": [[10.0, 10.0, 20.0, 20.0]] * n_instances,
        }

    def test_output_keys(self):
        out = coco_collate_fn([self._sample()])
        assert set(out.keys()) == {"images", "masks", "bboxes"}

    def test_images_is_tensor(self):
        out = coco_collate_fn([self._sample(), self._sample()])
        assert isinstance(out["images"], torch.Tensor)

    def test_images_shape_bchw(self):
        batch = [self._sample(), self._sample(), self._sample()]
        out = coco_collate_fn(batch)
        assert out["images"].shape == (3, 3, IMG_SIZE, IMG_SIZE)

    def test_images_dtype_float32(self):
        out = coco_collate_fn([self._sample()])
        assert out["images"].is_floating_point()

    def test_channel_order_is_chw(self):
        out = coco_collate_fn([self._sample()])
        _, C, H, W = out["images"].shape
        assert C == 3
        assert H == IMG_SIZE
        assert W == IMG_SIZE

    def test_masks_is_list_of_tensors(self):
        batch = [self._sample(n_instances=2), self._sample(n_instances=3)]
        out = coco_collate_fn(batch)
        assert isinstance(out["masks"], list)
        assert len(out["masks"]) == 2
        assert isinstance(out["masks"][0], torch.Tensor)
        assert isinstance(out["masks"][1], torch.Tensor)

    def test_masks_preserve_per_sample_instance_count(self):
        batch = [self._sample(n_instances=2), self._sample(n_instances=4)]
        out = coco_collate_fn(batch)
        assert out["masks"][0].shape == (2, IMG_SIZE, IMG_SIZE)
        assert out["masks"][1].shape == (4, IMG_SIZE, IMG_SIZE)

    def test_bboxes_is_list(self):
        batch = [self._sample(n_instances=2), self._sample(n_instances=1)]
        out = coco_collate_fn(batch)
        assert isinstance(out["bboxes"], list)
        assert len(out["bboxes"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Build transform factories
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildTransforms:
    def test_train_returns_compose(self, base_cfg):
        assert isinstance(build_train_transforms(base_cfg), A.Compose)

    def test_val_returns_compose(self, base_cfg):
        assert isinstance(build_val_transforms(base_cfg), A.Compose)

    def test_train_contains_copy_paste(self, base_cfg):
        tf = build_train_transforms(base_cfg)
        assert any(isinstance(t, CopyPaste) for t in tf.transforms)

    def test_val_has_no_copy_paste(self, base_cfg):
        tf = build_val_transforms(base_cfg)
        assert not any(isinstance(t, CopyPaste) for t in tf.transforms)

    def test_train_copy_paste_params_match_config(self, base_cfg):
        tf = build_train_transforms(base_cfg)
        cp: CopyPaste = next(t for t in tf.transforms if isinstance(t, CopyPaste))
        aug = base_cfg.augmentations
        assert cp.blend == aug.blend
        assert cp.sigma == aug.sigma
        assert cp.pct_objects_paste == aug.pct_objects_paste
        assert cp.max_paste_objects == aug.max_paste_objects

    def test_train_has_bbox_params(self, base_cfg):
        tf = build_train_transforms(base_cfg)
        assert tf.processors.get("bboxes") is not None

    def test_val_has_bbox_params(self, base_cfg):
        tf = build_val_transforms(base_cfg)
        assert tf.processors.get("bboxes") is not None

    def test_train_copy_paste_is_not_first_or_last(self, base_cfg):
        """CopyPaste should sit between pre-paste and post-paste transforms."""
        tf = build_train_transforms(base_cfg)
        indices = [i for i, t in enumerate(tf.transforms) if isinstance(t, CopyPaste)]
        assert len(indices) == 1
        idx = indices[0]
        assert idx > 0, "No pre-paste transforms found before CopyPaste"
        assert idx < len(tf.transforms) - 1, "No post-paste transforms found after CopyPaste"


# ─────────────────────────────────────────────────────────────────────────────
# Coco2017Dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestCoco2017Dataset:
    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _train_ds(cfg: DatasetConfig) -> Coco2017Dataset:
        return Coco2017Dataset(
            root=cfg.root,
            ann_file=cfg.train_json,
            img_dir=cfg.train_images,
            transforms=build_train_transforms(cfg),
            training=True,
        )

    @staticmethod
    def _val_ds(cfg: DatasetConfig) -> Coco2017Dataset:
        return Coco2017Dataset(
            root=cfg.root,
            ann_file=cfg.val_json,
            img_dir=cfg.val_images,
            transforms=build_val_transforms(cfg),
            training=False,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def test_train_len(self, base_cfg):
        assert len(self._train_ds(base_cfg)) == N_TRAIN

    def test_val_len(self, base_cfg):
        assert len(self._val_ds(base_cfg)) == N_VAL

    def test_train_mode_splits_pipeline(self, base_cfg):
        """Training dataset must populate all three split-transform attributes."""
        ds = self._train_ds(base_cfg)
        assert ds._pre_tf is not None
        assert ds._cp_tf is not None
        assert ds._post_tf is not None

    def test_val_mode_does_not_split_pipeline(self, base_cfg):
        """Validation dataset must leave _cp_tf as None."""
        ds = self._val_ds(base_cfg)
        assert ds._cp_tf is None

    def test_pre_tf_has_no_copy_paste(self, base_cfg):
        """Pre-paste sub-pipeline must not contain a CopyPaste node."""
        ds = self._train_ds(base_cfg)
        assert ds._pre_tf is not None
        assert not any(isinstance(t, CopyPaste) for t in ds._pre_tf.transforms)

    def test_cp_tf_contains_copy_paste(self, base_cfg):
        """Copy-paste sub-pipeline must contain exactly one CopyPaste node."""
        ds = self._train_ds(base_cfg)
        assert ds._cp_tf is not None
        cp_nodes = [t for t in ds._cp_tf.transforms if isinstance(t, CopyPaste)]
        assert len(cp_nodes) == 1

    # ── _load_raw ─────────────────────────────────────────────────────────────

    def test_load_raw_keys(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert {"image", "masks", "bboxes"} <= set(raw.keys())

    def test_load_raw_image_is_uint8(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert raw["image"].dtype == np.uint8

    def test_load_raw_image_is_rgb(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert raw["image"].ndim == 3
        assert raw["image"].shape[2] == 3

    def test_load_raw_image_dimensions_match_source(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        h, w, _ = raw["image"].shape
        assert h == IMG_H
        assert w == IMG_W

    def test_load_raw_masks_ndim(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert raw["masks"].ndim == 3  # (N, H, W)

    def test_load_raw_masks_count_matches_annotations(self, base_cfg):
        """Fixture creates 2 non-crowd annotations per training image."""
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert raw["masks"].shape[0] == 2

    def test_load_raw_bboxes_count_matches_annotations(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert len(raw["bboxes"]) == 2

    def test_load_raw_masks_spatial_dims(self, base_cfg):
        raw = self._train_ds(base_cfg)._load_raw(0)
        assert raw["masks"].shape[1] == IMG_H
        assert raw["masks"].shape[2] == IMG_W

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def test_train_getitem_image_is_float32(self, base_cfg):
        # albumentationsx >= 2.1 returns a torch.Tensor, not a numpy array
        sample = self._train_ds(base_cfg)[0]
        assert sample["image"].is_floating_point()

    def test_train_getitem_image_shape(self, base_cfg):
        # albumentationsx >= 2.1 outputs CHW tensors, not HWC numpy arrays
        sample = self._train_ds(base_cfg)[0]
        assert sample["image"].shape == (3, IMG_SIZE, IMG_SIZE)

    def test_train_getitem_masks_ndim(self, base_cfg):
        sample = self._train_ds(base_cfg)[0]
        assert sample["masks"].ndim == 3

    def test_train_getitem_masks_spatial_size(self, base_cfg):
        sample = self._train_ds(base_cfg)[0]
        assert sample["masks"].shape[1] == IMG_SIZE
        assert sample["masks"].shape[2] == IMG_SIZE

    def test_val_getitem_image_is_float32(self, base_cfg):
        sample = self._val_ds(base_cfg)[0]
        assert sample["image"].is_floating_point()

    def test_val_getitem_image_shape(self, base_cfg):
        sample = self._val_ds(base_cfg)[0]
        assert sample["image"].shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_getitem_masks_spatial_size(self, base_cfg):
        sample = self._val_ds(base_cfg)[0]
        assert sample["masks"].shape[1] == IMG_SIZE
        assert sample["masks"].shape[2] == IMG_SIZE

    def test_getitem_is_repeatable(self, base_cfg):
        """Two consecutive calls with the same index should both succeed."""
        ds = self._val_ds(base_cfg)
        s1 = ds[0]
        s2 = ds[0]
        # Both must produce valid float images of the same shape
        assert s1["image"].shape == s2["image"].shape

    # ── Crowd-annotation exclusion ────────────────────────────────────────────

    def test_crowd_annotations_are_excluded(self, tmp_path):
        """iscrowd=1 annotations must not appear as instance masks."""
        img_dir = tmp_path / "imgs"
        ann_dir = tmp_path / "anns"
        img_dir.mkdir()
        ann_dir.mkdir()

        Image.fromarray(np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)).save(
            img_dir / "000000000001.jpg"
        )

        coco_json = {
            "images": [{"id": 1, "file_name": "000000000001.jpg", "height": IMG_H, "width": IMG_W}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[5.0, 5.0, 35.0, 5.0, 35.0, 35.0, 5.0, 35.0]],
                    "bbox": [5.0, 5.0, 30.0, 30.0],
                    "area": 900.0,
                    "iscrowd": 1,  # ← crowd annotation – must be excluded
                }
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco_json))

        cfg = DatasetConfig(
            root=str(tmp_path),
            imgsz=IMG_SIZE,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            train_json="anns/instances.json",
            val_json="anns/instances.json",
            train_images="imgs",
            val_images="imgs",
        )
        ds = Coco2017Dataset(
            root=cfg.root,
            ann_file=cfg.val_json,
            img_dir=cfg.val_images,
            transforms=build_val_transforms(cfg),
            training=False,
        )
        assert ds._load_raw(0)["masks"].shape[0] == 0

    # ── Empty annotation list ─────────────────────────────────────────────────

    def test_image_with_no_annotations_yields_empty_masks(self, tmp_path):
        img_dir = tmp_path / "imgs"
        ann_dir = tmp_path / "anns"
        img_dir.mkdir()
        ann_dir.mkdir()

        Image.fromarray(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)).save(img_dir / "000000000001.jpg")
        coco_json = {
            "images": [{"id": 1, "file_name": "000000000001.jpg", "height": IMG_H, "width": IMG_W}],
            "annotations": [],
            "categories": [],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco_json))

        cfg = DatasetConfig(
            root=str(tmp_path),
            imgsz=IMG_SIZE,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            train_json="anns/instances.json",
            val_json="anns/instances.json",
            train_images="imgs",
            val_images="imgs",
        )
        ds = Coco2017Dataset(
            root=cfg.root,
            ann_file=cfg.val_json,
            img_dir=cfg.val_images,
            transforms=build_val_transforms(cfg),
            training=False,
        )
        raw = ds._load_raw(0)
        assert raw["masks"].shape[0] == 0
        assert raw["bboxes"] == []

    # ── __repr__ ─────────────────────────────────────────────────────────────

    def test_repr_contains_class_name(self, base_cfg):
        assert "Coco2017Dataset" in repr(self._val_ds(base_cfg))


# ─────────────────────────────────────────────────────────────────────────────
# Coco2017DataModule
# ─────────────────────────────────────────────────────────────────────────────


class TestCoco2017DataModule:
    # ── setup() ───────────────────────────────────────────────────────────────

    def test_setup_fit_creates_both_datasets(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert dm.train_ds is not None
        assert dm.val_ds is not None

    def test_setup_validate_creates_only_val(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("validate")
        assert dm.train_ds is None
        assert dm.val_ds is not None

    def test_setup_none_creates_both(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup(None)
        assert dm.train_ds is not None
        assert dm.val_ds is not None

    def test_train_ds_is_training_mode(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert dm.train_ds is not None
        assert dm.train_ds.training is True

    def test_val_ds_is_not_training_mode(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert dm.val_ds is not None
        assert dm.val_ds.training is False

    # ── RuntimeError before setup ─────────────────────────────────────────────

    def test_train_dataloader_raises_before_setup(self, base_cfg):
        with pytest.raises(RuntimeError):
            Coco2017DataModule(base_cfg).train_dataloader()

    def test_val_dataloader_raises_before_setup(self, base_cfg):
        with pytest.raises(RuntimeError):
            Coco2017DataModule(base_cfg).val_dataloader()

    # ── DataLoader lengths ────────────────────────────────────────────────────

    def test_train_dataloader_num_batches(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        expected = math.ceil(N_TRAIN / base_cfg.batch_size)
        assert len(dm.train_dataloader()) == expected

    def test_val_dataloader_num_batches(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        expected = math.ceil(N_VAL / base_cfg.batch_size)
        assert len(dm.val_dataloader()) == expected

    # ── Batch structure ───────────────────────────────────────────────────────

    def test_train_batch_keys(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert set(batch.keys()) == {"images", "masks", "bboxes"}

    def test_val_batch_keys(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        assert set(batch.keys()) == {"images", "masks", "bboxes"}

    def test_train_batch_images_shape(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        _, C, H, W = batch["images"].shape
        assert C == 3
        assert H == IMG_SIZE
        assert W == IMG_SIZE

    def test_val_batch_images_shape(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        _, C, H, W = batch["images"].shape
        assert C == 3
        assert H == IMG_SIZE
        assert W == IMG_SIZE

    def test_train_batch_images_dtype(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["images"].is_floating_point()

    def test_val_batch_images_dtype(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        assert batch["images"].is_floating_point()

    def test_train_batch_masks_is_list(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch["masks"], list)
        assert len(batch["masks"]) == base_cfg.batch_size

    def test_val_batch_masks_is_list(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        assert isinstance(batch["masks"], list)

    def test_train_batch_masks_spatial_dims(self, base_cfg):
        """Each per-sample mask tensor must have H=W=IMG_SIZE."""
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        for mask_tensor in batch["masks"]:
            assert mask_tensor.ndim == 3  # (N, H, W)
            assert mask_tensor.shape[1] == IMG_SIZE
            assert mask_tensor.shape[2] == IMG_SIZE

    def test_val_batch_masks_spatial_dims(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        for mask_tensor in batch["masks"]:
            assert mask_tensor.ndim == 3
            assert mask_tensor.shape[1] == IMG_SIZE
            assert mask_tensor.shape[2] == IMG_SIZE

    def test_train_batch_bboxes_is_list(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch["bboxes"], list)
        assert len(batch["bboxes"]) == base_cfg.batch_size

    # ── Sampler / shuffle ─────────────────────────────────────────────────────

    def test_train_dataloader_uses_random_sampler(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert isinstance(dm.train_dataloader().sampler, RandomSampler)

    def test_val_dataloader_uses_sequential_sampler(self, base_cfg):
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert isinstance(dm.val_dataloader().sampler, SequentialSampler)

    # ── persistent_workers guard ──────────────────────────────────────────────

    def test_persistent_workers_false_when_num_workers_zero(self, base_cfg):
        """PyTorch forbids persistent_workers=True with num_workers=0."""
        dm = Coco2017DataModule(base_cfg)
        dm.setup("fit")
        assert not dm.train_dataloader().persistent_workers
        assert not dm.val_dataloader().persistent_workers

    # ── __repr__ ─────────────────────────────────────────────────────────────

    def test_repr_contains_class_name(self, base_cfg):
        assert "Coco2017DataModule" in repr(Coco2017DataModule(base_cfg))
