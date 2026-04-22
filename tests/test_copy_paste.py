"""Tests for the CopyPaste augmentation and its functional helpers.

Covers
------
* ``extract_bboxes``        – bounding-box extraction from binary masks.
* ``image_copy_paste``      – image compositing with / without Gaussian blending.
* ``mask_copy_paste``       – single semantic mask paste.
* ``masks_copy_paste``      – instance-mask batch paste.
* ``bboxes_copy_paste``     – bounding-box update after paste.
* ``CopyPaste``             – albumentations v2 DualTransform (init + full pipeline).

The test suite is intentionally self-contained: all sample data is created
synthetically so no external datasets are required.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from cpa.augs.copy_paste import (
    CopyPaste,
    bboxes_copy_paste,
    extract_bboxes,
    image_copy_paste,
    mask_copy_paste,
    masks_copy_paste,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_sample(
    height: int = 256,
    width: int = 256,
    n_objects: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Create a synthetic (image, masks, bboxes) tuple.

    Args:
        height: Image height.
        width: Image width.
        n_objects: Number of annotated objects.
        seed: RNG seed for reproducibility.

    Returns:
        image: ``(H, W, 3)`` uint8 array.
        masks: ``(N, H, W)`` uint8 stacked instance masks.
        bboxes: COCO-format list ``[[x, y, w, h], ...]``.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    if n_objects == 0:
        return image, np.empty((0, height, width), dtype=np.uint8), []

    masks_list: list[np.ndarray] = []
    bboxes: list[list[float]] = []

    for _ in range(n_objects):
        mask = np.zeros((height, width), dtype=np.uint8)
        x1 = int(rng.integers(10, width // 2))
        y1 = int(rng.integers(10, height // 2))
        x2 = min(x1 + int(rng.integers(30, max(31, width // 3))), width - 1)
        y2 = min(y1 + int(rng.integers(30, max(31, height // 3))), height - 1)
        mask[y1:y2, x1:x2] = 1
        masks_list.append(mask)
        bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    return image, np.stack(masks_list), bboxes


def make_pipeline(min_visibility: float = 0.0) -> A.Compose:
    """Reproduce the reference augmentation pipeline from the project README."""
    return A.Compose(
        [
            A.RandomScale(scale_limit=(-0.9, 1.0), p=1),
            A.PadIfNeeded(256, 256, border_mode=0),
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
        ],
        bbox_params=A.BboxParams(coord_format="coco", min_visibility=min_visibility),
    )


# ─────────────────────────────────────────────────────────────────────────────
# extract_bboxes
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractBboxes:
    def test_empty_input_returns_empty_list(self):
        assert extract_bboxes([]) == []

    def test_single_mask_square_image(self):
        """Bbox coords must match the mask footprint in a square image."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 20:70] = 1  # rows 10-49, cols 20-69
        ((x1, y1, x2, y2),) = extract_bboxes([mask])
        assert x1 == pytest.approx(20 / 100)
        assert y1 == pytest.approx(10 / 100)
        assert x2 == pytest.approx(70 / 100)
        assert y2 == pytest.approx(50 / 100)

    def test_all_zero_mask_yields_zero_bbox(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert extract_bboxes([mask]) == [(0.0, 0.0, 0.0, 0.0)]

    def test_non_square_image_normalises_axes_independently(self):
        """x should be normalised by width, y by height."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:50, 20:60] = 1
        ((x1, y1, x2, y2),) = extract_bboxes([mask])
        assert x1 == pytest.approx(20 / 200)
        assert y1 == pytest.approx(10 / 100)
        assert x2 == pytest.approx(60 / 200)
        assert y2 == pytest.approx(50 / 100)

    def test_multiple_masks_returns_correct_count(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m1[0:10, 0:10] = 1
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[50:80, 50:90] = 1
        result = extract_bboxes([m1, m2])
        assert len(result) == 2

    def test_coords_in_unit_range(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            mask = np.zeros((128, 128), dtype=np.uint8)
            r, c = rng.integers(10, 100, size=2)
            mask[r : r + 20, c : c + 20] = 1
            ((x1, y1, x2, y2),) = extract_bboxes([mask])
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0

    def test_full_mask_gives_unit_bbox(self):
        mask = np.ones((50, 80), dtype=np.uint8)
        ((x1, y1, x2, y2),) = extract_bboxes([mask])
        assert x1 == pytest.approx(0.0)
        assert y1 == pytest.approx(0.0)
        assert x2 == pytest.approx(1.0)
        assert y2 == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# image_copy_paste
# ─────────────────────────────────────────────────────────────────────────────


class TestImageCopyPaste:
    def test_none_alpha_returns_original_unchanged(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 200
        result = image_copy_paste(img, paste, alpha=None)
        np.testing.assert_array_equal(result, img)

    def test_full_alpha_no_blend_equals_paste(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 50
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 200
        alpha = np.ones((64, 64), dtype=np.uint8)
        result = image_copy_paste(img, paste, alpha, blend=False)
        np.testing.assert_array_equal(result, paste)

    def test_zero_alpha_no_blend_returns_original(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 77
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 200
        alpha = np.zeros((64, 64), dtype=np.uint8)
        result = image_copy_paste(img, paste, alpha, blend=False)
        np.testing.assert_array_equal(result, img)

    def test_output_dtype_preserved(self):
        img = np.ones((64, 64, 3), dtype=np.uint8)
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 200
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[10:20, 10:20] = 1
        result = image_copy_paste(img, paste, alpha, blend=True)
        assert result.dtype == np.uint8

    def test_pasted_region_is_filled_no_blend(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 200
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[10:20, 10:20] = 1
        result = image_copy_paste(img, paste, alpha, blend=False)
        np.testing.assert_array_equal(result[10:20, 10:20], 200)
        np.testing.assert_array_equal(result[30:40, 30:40], 0)

    def test_blend_produces_different_output_than_hard_paste(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        paste = np.ones((64, 64, 3), dtype=np.uint8) * 255
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[20:40, 20:40] = 1
        hard = image_copy_paste(img, paste, alpha.copy(), blend=False)
        soft = image_copy_paste(img, paste, alpha.copy(), blend=True, sigma=2)
        assert not np.array_equal(hard, soft), "Blending should soften edges"

    def test_output_clipped_to_uint8_range(self):
        img = np.full((32, 32, 3), 200, dtype=np.uint8)
        paste = np.full((32, 32, 3), 200, dtype=np.uint8)
        alpha = np.ones((32, 32), dtype=np.uint8)
        result = image_copy_paste(img, paste, alpha, blend=True, sigma=1)
        assert result.max() <= 255
        assert result.min() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# mask_copy_paste
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskCopyPaste:
    def test_none_alpha_returns_original(self):
        mask = np.ones((50, 50), dtype=np.uint8) * 3
        result = mask_copy_paste(mask, paste_mask=None, alpha=None)
        np.testing.assert_array_equal(result, mask)

    def test_none_paste_mask_returns_original(self):
        mask = np.ones((50, 50), dtype=np.uint8) * 7
        alpha = np.ones((50, 50), dtype=np.uint8)
        result = mask_copy_paste(mask, paste_mask=None, alpha=alpha)
        np.testing.assert_array_equal(result, mask)

    def test_paste_replaces_alpha_region(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        paste = np.ones((50, 50), dtype=np.uint8) * 5
        alpha = np.zeros((50, 50), dtype=np.uint8)
        alpha[10:20, 10:20] = 1
        result = mask_copy_paste(mask, paste, alpha)
        assert result[15, 15] == 5  # inside paste region
        assert result[0, 0] == 0  # outside paste region

    def test_dtype_preserved(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        paste = np.ones((50, 50), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.uint8)
        result = mask_copy_paste(mask, paste, alpha)
        assert result.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# masks_copy_paste
# ─────────────────────────────────────────────────────────────────────────────


class TestMasksCopyPaste:
    def test_none_alpha_returns_original_unchanged(self):
        masks = np.ones((2, 64, 64), dtype=np.uint8)
        paste = np.empty((0, 64, 64), dtype=np.uint8)
        result = masks_copy_paste(masks, paste, alpha=None)
        np.testing.assert_array_equal(result, masks)

    def test_paste_increases_mask_count(self):
        masks = np.ones((2, 64, 64), dtype=np.uint8)
        paste = np.ones((3, 64, 64), dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        result = masks_copy_paste(masks, paste, alpha)
        assert result.shape[0] == 5

    def test_occluded_pixels_zeroed_in_original_masks(self):
        masks = np.ones((1, 64, 64), dtype=np.uint8)
        paste = np.ones((1, 64, 64), dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[0:32, 0:32] = 1  # top-left quadrant is pasted over
        result = masks_copy_paste(masks, paste, alpha)
        # Occluded region of original mask must be 0
        assert result[0, 0:32, 0:32].max() == 0
        # Non-occluded region must still be 1
        assert result[0, 32:64, 32:64].min() == 1

    def test_channel_dim_preserved_when_present(self):
        masks = np.ones((2, 64, 64, 1), dtype=np.uint8)
        paste = np.ones((2, 64, 64), dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        result = masks_copy_paste(masks, paste, alpha)
        assert result.ndim == 4
        assert result.shape == (4, 64, 64, 1)

    def test_empty_paste_keeps_original(self):
        masks = np.ones((3, 32, 32), dtype=np.uint8)
        paste = np.empty((0, 32, 32), dtype=np.uint8)
        alpha = np.ones((32, 32), dtype=np.uint8)
        result = masks_copy_paste(masks, paste, alpha)
        # No new masks appended, but original pixels outside alpha are zeroed
        assert result.shape[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# bboxes_copy_paste
# ─────────────────────────────────────────────────────────────────────────────


class TestBboxesCopyPaste:
    def _make_bbox_array(self, n: int = 2, extra_cols: int = 0) -> np.ndarray:
        coords = np.tile([0.1, 0.1, 0.5, 0.5], (n, 1)).astype(np.float32)
        if extra_cols:
            extra = np.arange(n * extra_cols, dtype=np.float32).reshape(n, extra_cols)
            return np.hstack([coords, extra])
        return coords

    def _make_mask(self, h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    def test_none_alpha_returns_bboxes_unchanged(self):
        bboxes = self._make_bbox_array(3)
        result = bboxes_copy_paste(bboxes, orig_masks=None, paste_masks=np.empty((0,)), alpha=None)
        np.testing.assert_array_equal(result, bboxes)

    def test_paste_bboxes_appended(self):
        bboxes = self._make_bbox_array(2)
        orig = np.stack([self._make_mask(100, 100, 10, 10, 50, 50)] * 2)
        paste_m = np.stack([self._make_mask(100, 100, 60, 60, 90, 90)])
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[60:90, 60:90] = 1
        result = bboxes_copy_paste(bboxes, orig, paste_m, alpha)
        assert result.shape[0] == 3  # 2 original + 1 pasted

    def test_extra_label_cols_preserved_for_original(self):
        bboxes = self._make_bbox_array(2, extra_cols=1)
        orig = np.stack([self._make_mask(100, 100, 10, 10, 40, 40)] * 2)
        paste_m = np.stack([self._make_mask(100, 100, 60, 60, 80, 80)])
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[60:80, 60:80] = 1
        result = bboxes_copy_paste(bboxes, orig, paste_m, alpha)
        assert result.shape[1] == 5  # 4 coords + 1 label col
        # Original extra col values must match input
        np.testing.assert_array_equal(result[:2, 4], bboxes[:2, 4])

    def test_fully_occluded_object_gets_zero_bbox(self):
        """An object whose mask is fully covered becomes a (0,0,0,0) bbox."""
        bboxes = np.array([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)
        orig = np.zeros((1, 100, 100), dtype=np.uint8)
        orig[0, 10:40, 10:40] = 1
        alpha = np.ones((100, 100), dtype=np.uint8)  # covers everything
        paste_m = np.empty((0,), dtype=np.uint8)
        result = bboxes_copy_paste(bboxes, orig, paste_m, alpha)
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0, 0.0])

    def test_no_original_bboxes_returns_paste_bboxes(self):
        bboxes = np.empty((0, 4), dtype=np.float32)
        paste_m = np.stack([self._make_mask(100, 100, 20, 20, 60, 60)])
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[20:60, 20:60] = 1
        result = bboxes_copy_paste(bboxes, None, paste_m, alpha)
        assert result.shape == (1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# CopyPaste – initialisation
# ─────────────────────────────────────────────────────────────────────────────


class TestCopyPasteInit:
    def test_default_hyperparameters(self):
        cp = CopyPaste()
        assert cp.blend is True
        assert cp.sigma == pytest.approx(3.0)
        assert cp.pct_objects_paste == pytest.approx(0.1)
        assert cp.max_paste_objects is None
        assert cp.p == pytest.approx(0.5)

    def test_custom_hyperparameters(self):
        cp = CopyPaste(blend=False, sigma=2.0, pct_objects_paste=0.8, max_paste_objects=5, p=0.9)
        assert cp.blend is False
        assert cp.sigma == pytest.approx(2.0)
        assert cp.pct_objects_paste == pytest.approx(0.8)
        assert cp.max_paste_objects == 5
        assert cp.p == pytest.approx(0.9)

    def test_get_transform_init_args_names(self):
        names = CopyPaste().get_transform_init_args_names()
        assert set(names) >= {"blend", "sigma", "pct_objects_paste", "max_paste_objects"}

    def test_targets_as_params_contains_required_keys(self):
        tap = CopyPaste().targets_as_params
        assert "paste_image" in tap
        assert "paste_masks" in tap

    def test_class_fullname(self):
        assert CopyPaste.get_class_fullname() == "copypaste.CopyPaste"


# ─────────────────────────────────────────────────────────────────────────────
# CopyPaste – full pipeline integration
# ─────────────────────────────────────────────────────────────────────────────


class TestCopyPastePipeline:
    """Integration tests using the reference A.Compose pipeline."""

    def _run(
        self,
        n_base: int = 3,
        n_paste: int = 3,
        height: int = 256,
        width: int = 256,
        seed: int = 0,
    ) -> dict:
        transform = make_pipeline()
        img, masks, bboxes = make_sample(height, width, n_base, seed=seed)
        paste_img, paste_masks, _ = make_sample(height, width, n_paste, seed=seed + 1)
        np.random.seed(seed)
        return transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )

    # ── Output shapes and dtypes ──────────────────────────────────────────────

    def test_image_shape_is_256x256x3(self):
        out = self._run()
        assert out["image"].shape == (256, 256, 3)

    def test_image_dtype_is_uint8(self):
        out = self._run()
        assert out["image"].dtype == np.uint8

    def test_masks_are_2d_and_256x256(self):
        out = self._run(n_base=2, n_paste=2)
        assert out["masks"].ndim == 3
        assert out["masks"].shape[1:] == (256, 256)

    def test_paste_increases_mask_count(self):
        """After copy-paste the mask count must be ≥ n_base."""
        out = self._run(n_base=2, n_paste=4)
        assert out["masks"].shape[0] >= 2

    def test_bbox_count_matches_mask_count(self):
        """Number of output bboxes must equal number of output masks."""
        out = self._run(n_base=2, n_paste=2)
        assert len(out["bboxes"]) == out["masks"].shape[0]

    def test_output_bbox_values_in_valid_coco_range(self):
        """COCO bboxes (x, y, w, h) must be non-negative and fit the 256 canvas."""
        out = self._run(n_base=3, n_paste=3)
        for x, y, w, h in out["bboxes"]:
            assert x >= 0 and y >= 0
            assert w > 0 and h > 0
            assert x + w <= 256 + 1e-4
            assert y + h <= 256 + 1e-4

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_no_base_objects(self):
        transform = make_pipeline()
        img, masks, bboxes = make_sample(n_objects=0)
        paste_img, paste_masks, _ = make_sample(n_objects=3, seed=99)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["image"].shape == (256, 256, 3)
        # Any pasted objects should be reflected in masks / bboxes
        assert out["masks"].shape[0] == len(out["bboxes"])

    def test_no_paste_objects(self):
        transform = make_pipeline()
        img, masks, bboxes = make_sample(n_objects=3)
        paste_img, paste_masks, _ = make_sample(n_objects=0, seed=77)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["image"].shape == (256, 256, 3)

    def test_no_objects_either_side(self):
        transform = make_pipeline()
        img, masks, bboxes = make_sample(n_objects=0)
        paste_img, paste_masks, _ = make_sample(n_objects=0, seed=77)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["image"].shape == (256, 256, 3)
        assert out["masks"].shape[0] == 0
        assert len(out["bboxes"]) == 0

    def test_single_object_each_side(self):
        out = self._run(n_base=1, n_paste=1)
        assert out["image"].shape == (256, 256, 3)

    # ── Probability gate ──────────────────────────────────────────────────────

    def test_p_zero_skips_paste(self):
        """With p=0 the transform must be a no-op; mask count stays at n_base."""
        transform = A.Compose(
            [CopyPaste(p=0)],
            bbox_params=A.BboxParams(coord_format="coco"),
        )
        img, masks, bboxes = make_sample(n_objects=2)
        paste_img, paste_masks, _ = make_sample(n_objects=3, seed=99)
        np.random.seed(0)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["masks"].shape[0] == 2

    def test_p_one_always_pastes(self):
        """With p=1 and n_paste > 0, output masks count must exceed n_base."""
        transform = A.Compose(
            [CopyPaste(p=1, pct_objects_paste=1.0)],
            bbox_params=A.BboxParams(coord_format="coco"),
        )
        img, masks, bboxes = make_sample(n_objects=2)
        paste_img, paste_masks, _ = make_sample(n_objects=3, seed=5)
        np.random.seed(0)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["masks"].shape[0] > 2

    # ── max_paste_objects ─────────────────────────────────────────────────────

    def test_max_paste_objects_respected(self):
        cap = 1
        transform = A.Compose(
            [CopyPaste(p=1, pct_objects_paste=1.0, max_paste_objects=cap)],
            bbox_params=A.BboxParams(coord_format="coco"),
        )
        img, masks, bboxes = make_sample(n_objects=2, seed=0)
        paste_img, paste_masks, _ = make_sample(n_objects=5, seed=1)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        n_pasted = out["masks"].shape[0] - 2
        assert n_pasted <= cap

    # ── Paste keys do not bleed into output ───────────────────────────────────

    def test_paste_keys_absent_from_output(self):
        out = self._run()
        for key in ("paste_image", "paste_masks", "paste_bboxes", "paste_keypoints"):
            assert key not in out, f"Key '{key}' should not appear in transform output"

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_deterministic_with_same_seed(self):
        """Two Compose pipelines built with the same seed must produce identical output."""
        img, masks, bboxes = make_sample(seed=7)
        paste_img, paste_masks, _ = make_sample(seed=8)
        call_kwargs = dict(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )

        def _seeded_pipeline() -> A.Compose:
            return A.Compose(
                [
                    A.RandomScale(scale_limit=(-0.9, 1.0), p=1),
                    A.PadIfNeeded(256, 256, border_mode=0),
                    A.RandomCrop(256, 256),
                    A.HorizontalFlip(p=0.5),
                    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
                ],
                bbox_params=A.BboxParams(coord_format="coco"),
                seed=42,
            )

        r1 = _seeded_pipeline()(**call_kwargs)  # type: ignore[call-arg]
        r2 = _seeded_pipeline()(**call_kwargs)  # type: ignore[call-arg]
        np.testing.assert_array_equal(r1["image"], r2["image"])
        np.testing.assert_array_equal(r1["masks"], r2["masks"])

    # ── Blend vs no-blend ─────────────────────────────────────────────────────

    def test_blend_false_produces_hard_edges(self):
        """blend=False output should differ from blend=True for the same data."""

        def _run_blend(blend: bool, seed: int = 99) -> np.ndarray:
            transform = A.Compose(
                [CopyPaste(blend=blend, sigma=2, pct_objects_paste=1.0, p=1)],
                bbox_params=A.BboxParams(coord_format="coco"),
            )
            img, masks, bboxes = make_sample(n_objects=2, seed=seed)
            paste_img, paste_masks, _ = make_sample(n_objects=2, seed=seed + 10)
            np.random.seed(seed)
            return transform(
                image=img,
                masks=masks,
                bboxes=bboxes,
                paste_image=paste_img,
                paste_masks=paste_masks,
            )["image"]

        hard = _run_blend(blend=False)
        soft = _run_blend(blend=True)
        # Images differ because Gaussian blending softens the alpha boundary
        assert not np.array_equal(hard, soft)

    # ── Non-square input ──────────────────────────────────────────────────────

    def test_non_square_input_image(self):
        transform = A.Compose(
            [
                A.PadIfNeeded(128, 192, border_mode=0),
                A.RandomCrop(128, 192),
                CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
            ],
            bbox_params=A.BboxParams(coord_format="coco"),
        )
        img, masks, bboxes = make_sample(height=128, width=192, n_objects=2)
        paste_img, paste_masks, _ = make_sample(height=128, width=192, n_objects=2, seed=5)
        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks,
        )
        assert out["image"].shape == (128, 192, 3)
        assert out["masks"].shape[1:] == (128, 192)

    # ── paste_masks as Python list (backward-compat) ──────────────────────────

    def test_paste_masks_as_list_of_arrays(self):
        """Users may pass paste_masks as a plain Python list instead of ndarray."""
        transform = A.Compose(
            [CopyPaste(p=1, pct_objects_paste=1.0)],
            bbox_params=A.BboxParams(coord_format="coco"),
        )
        img, masks, bboxes = make_sample(n_objects=1)
        paste_img, paste_masks_arr, _ = make_sample(n_objects=2, seed=9)
        paste_masks_list = list(paste_masks_arr)  # list of (H, W) arrays

        out = transform(
            image=img,
            masks=masks,
            bboxes=bboxes,
            paste_image=paste_img,
            paste_masks=paste_masks_list,
        )
        assert out["image"].shape == (256, 256, 3)
