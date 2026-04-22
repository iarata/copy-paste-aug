"""Copy-Paste augmentation for instance segmentation.

Changes vs the original v1
----------------------------------------------
* ``get_params_dependent_on_targets`` → ``get_params_dependent_on_data``.
* ``apply_with_params`` no longer takes a ``force_apply`` argument.
* Masks are stacked ``np.ndarray (N, H, W)`` (not a list).
* Bboxes are ``np.ndarray (N, 4+)`` in normalised albumentations format.
* ``always_apply`` removed (deprecated in v2).
* ``mask_copy_paste`` implemented (was ``raise NotImplementedError``).
* Variable names in ``extract_bboxes`` corrected (axes were mislabelled).
* Paste keys consumed in ``apply_with_params``; they do not leak downstream.
* ``BboxParams`` uses ``coord_format=`` (v2) instead of ``format=`` (v1).

References:
    Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method
    for Instance Segmentation", CVPR 2021.  https://arxiv.org/abs/2012.07177
"""

from __future__ import annotations

import random
from typing import Any

import albumentations as A
import numpy as np
from skimage.filters import gaussian

# ─────────────────────────────────────────────────────────────────────────────
# Functional helpers
# ─────────────────────────────────────────────────────────────────────────────


def image_copy_paste(
    img: np.ndarray,
    paste_img: np.ndarray,
    alpha: np.ndarray | None,
    blend: bool = True,
    sigma: float = 1.0,
) -> np.ndarray:
    """Composite *paste_img* onto *img* using a binary / float alpha mask.

    Args:
        img: Base image ``(H, W, C)`` uint8.
        paste_img: Image to paste ``(H, W, C)`` uint8.
        alpha: Paste region ``(H, W)`` – 1 = paste, 0 = keep original.
               ``None`` returns *img* unchanged.
        blend: When ``True`` a Gaussian blur is applied to *alpha* for soft,
               anti-aliased edges.
        sigma: Standard deviation for the Gaussian blur (``blend=True``).

    Returns:
        Composited image with the same dtype as *img*.
    """
    if alpha is None:
        return img

    img_dtype = img.dtype
    a = alpha.astype(np.float32)

    if blend:
        a = gaussian(a, sigma=sigma, preserve_range=True).astype(np.float32)

    a3 = a[..., None]  # (H, W, 1) broadcast over channels
    out = paste_img.astype(np.float32) * a3 + img.astype(np.float32) * (1.0 - a3)
    return np.clip(out, 0, 255).astype(img_dtype)


def mask_copy_paste(
    mask: np.ndarray,
    paste_mask: np.ndarray | None,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Paste *paste_mask* onto a semantic segmentation *mask* where *alpha* > 0.

    Args:
        mask: Base semantic mask ``(H, W)``.
        paste_mask: Semantic mask to paste ``(H, W)``, or ``None``.
        alpha: Binary paste region ``(H, W)``, or ``None``.

    Returns:
        Updated mask with the same dtype as the input.
    """
    if alpha is not None and paste_mask is not None:
        return np.where(alpha > 0, paste_mask, mask).astype(mask.dtype)
    return mask


def masks_copy_paste(
    masks: np.ndarray,
    paste_masks: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Apply copy-paste to a batch of instance-segmentation masks.

    Pixels of every *original* mask that fall within the paste region
    (``alpha > 0``) are zeroed out.  The *paste_masks* are then appended.

    Args:
        masks: Original instance masks ``(N, H, W)`` or ``(N, H, W, 1)``.
        paste_masks: Masks to paste ``(M, H, W)``.
        alpha: Binary paste region ``(H, W)``, or ``None``.

    Returns:
        Updated mask batch ``(N+M, H, W[, 1])`` – same channel convention as
        the input.
    """
    has_channel = masks.ndim == 4
    masks_2d = masks[..., 0] if has_channel else masks  # (N, H, W)

    if alpha is not None and len(paste_masks) > 0:
        alpha_bool = alpha.astype(bool)
        adjusted = np.where(~alpha_bool, masks_2d, 0).astype(masks_2d.dtype)
        combined = np.concatenate([adjusted, paste_masks], axis=0)  # (N+M, H, W)
    else:
        combined = masks_2d

    return combined[..., None] if has_channel else combined


def extract_bboxes(
    masks: list[np.ndarray],
) -> list[tuple[float, float, float, float]]:
    """Derive normalised ``(x_min, y_min, x_max, y_max)`` bboxes from binary masks.

    Args:
        masks: Binary / uint8 masks, each of shape ``(H, W)``.

    Returns:
        List of ``(x_min, y_min, x_max, y_max)`` bboxes in the albumentations
        internal normalised format ``[0, 1]``.
    """
    if not masks:
        return []

    h, w = masks[0].shape
    bboxes: list[tuple[float, float, float, float]] = []

    for mask in masks:
        # axis=0 collapses rows  → column-presence array  → x-axis indices
        x_idx = np.where(np.any(mask, axis=0))[0]
        # axis=1 collapses cols  → row-presence array     → y-axis indices
        y_idx = np.where(np.any(mask, axis=1))[0]

        if x_idx.size:
            x1 = float(x_idx[0]) / w
            x2 = float(x_idx[-1] + 1) / w
            y1 = float(y_idx[0]) / h
            y2 = float(y_idx[-1] + 1) / h
        else:
            x1 = y1 = x2 = y2 = 0.0

        bboxes.append((x1, y1, x2, y2))

    return bboxes


def bboxes_copy_paste(
    bboxes: np.ndarray,
    orig_masks: np.ndarray | None,
    paste_masks: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Recompute bboxes for original objects and append paste-object bboxes.

    After copy-paste some original objects are partially or fully occluded.
    Their bboxes are recomputed from the *adjusted* (post-paste) masks.
    Paste-object bboxes are derived directly from *paste_masks*.

    Args:
        bboxes: Original bboxes in albumentations internal format ``(N, 4+)``.
        orig_masks: Original instance masks ``(N, H, W)`` before paste, or
                    ``None`` (bboxes are kept as-is in that case).
        paste_masks: Selected paste masks ``(M, H, W)``.
        alpha: Binary paste region ``(H, W)``, or ``None``.

    Returns:
        Updated bboxes ``(N+M, 4+)`` in albumentations internal format.
    """
    if alpha is None:
        return bboxes

    n_orig = len(bboxes)
    # Number of extra columns beyond the 4 coordinate columns (e.g. label fields)
    extra_cols = bboxes.shape[1] - 4 if (bboxes.ndim == 2 and bboxes.shape[1] > 4) else 0

    # ── Recompute original bboxes after occlusion ─────────────────────────────
    if orig_masks is not None and n_orig > 0:
        alpha_bool = alpha.astype(bool)
        # Guard: orig_masks may contain more entries than bboxes if some bboxes
        # were previously filtered; we align on bboxes count.
        om = orig_masks[:n_orig]  # (n_orig, H, W)
        adjusted = (~alpha_bool & (om > 0)).astype(np.uint8)
        adj_coords = np.array(extract_bboxes(list(adjusted)), dtype=np.float32)  # (n_orig, 4)
        adj_full: np.ndarray = np.hstack([adj_coords, bboxes[:n_orig, 4:]]) if extra_cols else adj_coords
    else:
        adj_full = bboxes  # keep as-is when no mask info is available

    # ── Append paste bboxes ───────────────────────────────────────────────────
    if len(paste_masks) > 0:
        paste_coords = np.array(extract_bboxes(list(paste_masks)), dtype=np.float32)  # (M, 4)
        if extra_cols:
            pad = np.zeros((len(paste_coords), extra_cols), dtype=np.float32)
            paste_full: np.ndarray = np.hstack([paste_coords, pad])
        else:
            paste_full = paste_coords

        return np.vstack([adj_full, paste_full]) if len(adj_full) else paste_full

    return adj_full


def keypoints_copy_paste(
    keypoints: np.ndarray,
    paste_keypoints: np.ndarray | None,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Remove keypoints occluded by pasted objects and append paste keypoints.

    Args:
        keypoints: Original keypoints ``(N, 2+)``.
        paste_keypoints: Keypoints to append, or ``None``.
        alpha: Binary paste mask ``(H, W)``, or ``None``.

    Returns:
        Updated keypoints array.
    """
    if alpha is None:
        return keypoints

    visible = [kp for kp in keypoints if alpha[int(kp[1]), int(kp[0])] == 0]

    if paste_keypoints is not None and len(paste_keypoints):
        visible.extend(list(paste_keypoints))

    return np.array(visible, dtype=keypoints.dtype) if visible else keypoints[:0]


# ─────────────────────────────────────────────────────────────────────────────
# DualTransform class
# ─────────────────────────────────────────────────────────────────────────────


class CopyPaste(A.DualTransform):
    """Copy-Paste augmentation for instance segmentation (albumentations v2).

    Randomly selects a fraction of annotated objects from a *paste image*
    and composites them onto the base image, updating instance masks and
    bounding boxes accordingly.

    Typical usage inside ``A.Compose``::

        transform = A.Compose(
            [
                A.RandomScale(scale_limit=(-0.9, 1.0), p=1),
                A.PadIfNeeded(256, 256, border_mode=0),
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
                CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
            ],
            bbox_params=A.BboxParams(coord_format="coco"),
        )

        result = transform(
            image=image,           # (H, W, 3) uint8
            masks=masks,           # (N, H, W) stacked instance masks
            bboxes=bboxes,         # COCO format [[x, y, w, h], ...]
            paste_image=paste_img, # (H, W, 3) uint8
            paste_masks=paste_masks,  # (M, H, W) stacked instance masks
        )

    Args:
        blend: Apply a Gaussian blur to the paste alpha mask for smooth,
               anti-aliased object edges.
        sigma: Standard deviation of the Gaussian blur (``blend=True``).
        pct_objects_paste: Fraction of paste objects to randomly sample.
               ``0`` → paste *all* available objects.
        max_paste_objects: Hard upper bound on the number of pasted objects.
               ``None`` → no limit.
        p: Probability of applying the transform.

    Note:
        ``paste_image`` and ``paste_masks`` **must** be passed as keyword
        arguments when calling the composed transform.  They are consumed
        internally and will *not* appear in the output dictionary.
    """

    def __init__(
        self,
        blend: bool = True,
        sigma: float = 3.0,
        pct_objects_paste: float = 0.1,
        max_paste_objects: int | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects

    @classmethod
    def get_class_fullname(cls) -> str:  # noqa: D102
        return "copypaste.CopyPaste"

    @property
    def targets_as_params(self) -> list[str]:
        """Keys that *must* be present in the call-time data dict."""
        return ["paste_image", "paste_masks"]

    # ── Parameter computation ─────────────────────────────────────────────────

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Select objects to paste and build the composite alpha mask."""
        paste_image: np.ndarray = data["paste_image"]

        # Normalise paste_masks → (M, H, W) ndarray
        raw_pm = data["paste_masks"]
        if isinstance(raw_pm, (list, tuple)):
            paste_masks: np.ndarray = (
                np.stack(raw_pm) if len(raw_pm) else np.empty((0, *paste_image.shape[:2]), dtype=np.uint8)
            )
        else:
            paste_masks = np.asarray(raw_pm)

        # Extract 2-D original masks (strip channel dim added by albumentations)
        raw_om = data.get("masks")
        if raw_om is not None and len(raw_om):
            arr = np.asarray(raw_om)
            orig_masks_2d: np.ndarray = arr[..., 0] if arr.ndim == 4 else arr
        else:
            h, w = paste_masks.shape[1:3] if len(paste_masks) else paste_image.shape[:2]
            orig_masks_2d = np.empty((0, h, w), dtype=np.uint8)

        n_objects = len(paste_masks)
        _no_paste: dict[str, Any] = {
            "paste_img": None,
            "alpha": None,
            "sel_paste_masks": np.empty((0, *orig_masks_2d.shape[1:]), dtype=np.uint8),
            "orig_masks_2d": orig_masks_2d,
        }

        if n_objects == 0:
            return _no_paste

        # How many objects to sample
        if self.pct_objects_paste > 0:
            n_select = max(1, int(n_objects * self.pct_objects_paste))
        else:
            n_select = n_objects  # pct == 0  →  paste everything

        if self.max_paste_objects is not None:
            n_select = min(n_select, self.max_paste_objects)
        n_select = min(n_select, n_objects)

        idx = self.random_generator.choice(n_objects, size=n_select, replace=False)
        sel: np.ndarray = paste_masks[idx]  # (n_select, H, W)

        # Binary union of selected masks → composite alpha
        alpha: np.ndarray = np.any(sel > 0, axis=0).astype(np.uint8)  # (H, W)

        return {
            "paste_img": paste_image,
            "alpha": alpha,
            "sel_paste_masks": sel,
            "orig_masks_2d": orig_masks_2d,
        }

    # ── Transform dispatch ────────────────────────────────────────────────────

    def apply_with_params(
        self,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Strip paste_* keys before dispatching to standard target functions.

        The paste inputs have already been consumed by
        :meth:`get_params_dependent_on_data`; they must not be passed to
        downstream transforms.
        """
        _PASTE_KEYS = frozenset({"paste_image", "paste_masks", "paste_bboxes", "paste_keypoints"})
        filtered = {k: v for k, v in kwargs.items() if k not in _PASTE_KEYS}
        return super().apply_with_params(params, **filtered)

    def apply(
        self,
        img: np.ndarray,
        paste_img: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        if paste_img is None or alpha is None:
            return img
        return image_copy_paste(img, paste_img, alpha, blend=self.blend, sigma=self.sigma)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        alpha: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Occlude pixels of a *semantic* mask that are covered by pasted objects."""
        if alpha is not None:
            return np.where(alpha > 0, 0, mask).astype(mask.dtype)
        return mask

    def apply_to_masks(
        self,
        masks: np.ndarray,
        alpha: np.ndarray | None = None,
        sel_paste_masks: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        if sel_paste_masks is None or len(sel_paste_masks) == 0:
            sel_paste_masks = np.empty((0, *masks.shape[1:3]), dtype=np.uint8)
        return masks_copy_paste(masks, sel_paste_masks, alpha)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        alpha: np.ndarray | None = None,
        orig_masks_2d: np.ndarray | None = None,
        sel_paste_masks: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        paste = (
            sel_paste_masks
            if sel_paste_masks is not None and len(sel_paste_masks)
            else np.empty((0,), dtype=np.uint8)
        )
        return bboxes_copy_paste(bboxes, orig_masks_2d, paste, alpha)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        alpha: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Remove occluded keypoints.  Paste-side keypoints are not yet supported."""
        if alpha is not None and len(keypoints):
            visible = [kp for kp in keypoints if alpha[int(kp[1]), int(kp[0])] == 0]
            return np.array(visible, dtype=keypoints.dtype) if visible else keypoints[:0]
        return keypoints

    def get_transform_init_args_names(self) -> tuple[str, ...]:  # noqa: D102
        return ("blend", "sigma", "pct_objects_paste", "max_paste_objects")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-class decorator
# ─────────────────────────────────────────────────────────────────────────────


def copy_paste_class(dataset_class):  # noqa: C901
    """Class decorator that wires Copy-Paste augmentation into a dataset.

    The decorated dataset must expose:

    * ``self.transforms`` – an ``A.Compose`` pipeline containing a
      :class:`CopyPaste` transform.
    * ``self.load_example(idx)`` – method returning a data dict with keys
      ``image``, ``masks``, ``bboxes`` (and optionally ``keypoints``).
    * ``self.__len__()`` – dataset length.

    The decorator splits the pipeline at the ``CopyPaste`` step, applies
    pre-paste transforms to both the target and the randomly chosen paste
    sample, runs the copy-paste step, then applies any post-paste transforms.
    """

    def _split_transforms(self) -> None:
        split_index: int | None = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == "copypaste.CopyPaste":
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index + 1 :]

            bbox_params = None
            keypoint_params = None
            paste_additional_targets: dict[str, str] = {}

            if "bboxes" in self.transforms.processors:
                bbox_params = self.transforms.processors["bboxes"].params
                paste_additional_targets["paste_bboxes"] = "bboxes"
                if bbox_params.label_fields:
                    msg = (
                        "Copy-paste does not support bbox label_fields! "
                        "Expected bbox format: (x, y, w, h[, extra_field])"
                    )
                    raise ValueError(msg)

            if "keypoints" in self.transforms.processors:
                keypoint_params = self.transforms.processors["keypoints"].params
                paste_additional_targets["paste_keypoints"] = "keypoints"
                if keypoint_params.label_fields:
                    raise ValueError("Copy-paste does not support keypoint label fields!")

            if self.transforms.additional_targets:
                raise ValueError("Copy-paste does not support additional_targets on the outer Compose!")

            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste],
                bbox_params,
                keypoint_params,
                additional_targets=paste_additional_targets,
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not hasattr(self, "post_transforms"):
            self._split_transforms()

        img_data = self.load_example(idx)

        if self.copy_paste is not None:
            paste_idx = random.randint(0, len(self) - 1)
            paste_data = self.load_example(paste_idx)
            # Prefix all paste-sample keys with "paste_"
            paste_data = {"paste_" + k: v for k, v in paste_data.items()}

            img_data = self.copy_paste(**img_data, **paste_data)
            img_data = self.post_transforms(**img_data)
            img_data["paste_index"] = paste_idx

        return img_data

    setattr(dataset_class, "_split_transforms", _split_transforms)
    setattr(dataset_class, "__getitem__", __getitem__)
    return dataset_class


if __name__ == "__main__":
    # Quick smoke-test ─────────────────────────────────────────────────────────
    import albumentations as A  # noqa: F811
    import numpy as np  # noqa: F811

    def _make_sample(
        h: int = 256,
        w: int = 256,
        n: int = 3,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        if n == 0:
            return image, np.empty((0, h, w), dtype=np.uint8), []
        masks_list, bboxes = [], []
        for _ in range(n):
            mask = np.zeros((h, w), dtype=np.uint8)
            x1 = int(rng.integers(10, w // 2))
            y1 = int(rng.integers(10, h // 2))
            x2 = min(x1 + int(rng.integers(30, w // 3)), w - 1)
            y2 = min(y1 + int(rng.integers(30, h // 3)), h - 1)
            mask[y1:y2, x1:x2] = 1
            masks_list.append(mask)
            bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        return image, np.stack(masks_list), bboxes

    transform = A.Compose(
        [
            A.RandomScale(scale_limit=(-0.9, 1.0), p=1),
            A.PadIfNeeded(256, 256, border_mode=0),
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
        ],
        bbox_params=A.BboxParams(coord_format="coco"),
    )

    img, masks, bboxes = _make_sample(seed=0)
    paste_img, paste_masks, _ = _make_sample(seed=1)

    result = transform(
        image=img,
        masks=masks,
        bboxes=bboxes,
        paste_image=paste_img,
        paste_masks=paste_masks,
    )

    print(f"image  : {result['image'].shape}  dtype={result['image'].dtype}")
    print(f"masks  : {result['masks'].shape}")
    print(f"bboxes : {len(result['bboxes'])} boxes")
    assert result["image"].shape == (256, 256, 3)
    print("Smoke-test passed ✓")
