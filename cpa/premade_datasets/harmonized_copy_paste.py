"""Copy-paste method with Libcom image-harmonization inference.

The random geometric operations intentionally mirror ``SimpleCopyPasteMethod``:
base/paste scale jittering, horizontal flips, and object subset selection all
consume the same NumPy RNG calls in the same order.  Harmonization happens only
after the composite image and updated masks are fixed.

This module calls the underlying PCTNet/LBM model code directly instead of
``ImageHarmonizationModel`` because Libcom's public wrapper checks for CUDA
before accepting a device.  The model code itself can run on CPU/MPS/CUDA when
the selected backend supports the required PyTorch operations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps the code importable.
    fcntl = None
import math
import os
from pathlib import Path
import shutil
import sys
from threading import Lock
import types
from typing import Any
import zipfile

import cv2
from diffusers import FlowMatchEulerDiscreteScheduler
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor

from cpa.augs.copy_paste import image_copy_paste
from cpa.utils.dataset_subset import validate_subset_percent

_MODEL_CACHE: dict[tuple[str, str, torch.dtype], "_BaseHarmonizer"] = {}
_MODEL_CACHE_LOCK = Lock()
_MODEL_INFERENCE_LOCK = Lock()
_LIBCOM_HF_REPO = "BCMIZB/Libcom_pretrained_models"


class HarmonizedCopyPasteMethod:
    """Simple copy-paste followed by direct Libcom foreground harmonization."""

    name = "harmonized"

    def generate(
        self,
        *,
        base: Any,
        paste: Any,
        rng: np.random.Generator,
        config: Any,
    ) -> tuple[np.ndarray, list[Any], list[int]]:
        target_h, target_w = base.image.shape[:2]
        base_aug = _scale_jitter_flip_to_canvas(
            base,
            target_h=target_h,
            target_w=target_w,
            rng=rng,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            flip_prob=config.flip_prob,
        )
        paste_aug = _scale_jitter_flip_to_canvas(
            paste,
            target_h=target_h,
            target_w=target_w,
            rng=rng,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            flip_prob=config.flip_prob,
        )

        if not paste_aug.instances:
            return base_aug.image, base_aug.instances, []

        selected_indices = _select_object_indices(
            len(paste_aug.instances),
            percent=config.objects_paste_percent,
            max_objects=config.max_paste_objects,
            rng=rng,
        )
        selected_paste_instances = [paste_aug.instances[index] for index in selected_indices]
        selected_ids = [
            instance.source_annotation_id
            for instance in selected_paste_instances
            if instance.source_annotation_id is not None
        ]

        alpha = np.any([instance.mask > 0 for instance in selected_paste_instances], axis=0).astype(np.uint8)
        image = image_copy_paste(
            base_aug.image,
            paste_aug.image,
            alpha,
            blend=config.blend,
            sigma=config.sigma,
        )

        output_instances: list[Any] = []
        alpha_bool = alpha.astype(bool)
        for instance in base_aug.instances:
            visible_mask = instance.mask.copy()
            visible_mask[alpha_bool] = 0
            if visible_mask.any():
                output_instances.append(_instance_with_mask(instance, visible_mask))

        output_instances.extend(selected_paste_instances)
        image = _harmonize_image(image, alpha, config, rng)
        return image, output_instances, selected_ids


def normalize_harmonization_model_type(model_type: str) -> str:
    """Normalize accepted CLI aliases to Libcom's model type names."""

    normalized = model_type.strip().upper()
    if normalized in {"PCTNET", "PCNET"}:
        return "PCTNet"
    if normalized == "LBM":
        return "LBM"
    raise ValueError("harmonization_model_type must be one of: PCTNet, PCNet, LBM.")


def _harmonize_image(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    config: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if alpha.max() == 0:
        return image_rgb

    model_type = normalize_harmonization_model_type(config.harmonization_model_type)
    model = _get_harmonization_model(config)
    mask = (alpha > 0).astype(np.uint8) * 255
    composite_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    kwargs: dict[str, int] = {}
    if model_type == "LBM":
        kwargs["steps"] = int(config.harmonization_steps)
        kwargs["resolution"] = int(config.harmonization_resolution)

    _seed_torch(int(rng.integers(0, 2**31 - 1)))
    # Keep threaded runs stable and avoid concurrent mutation in model internals.
    with _MODEL_INFERENCE_LOCK:
        harmonized_bgr = model(composite_bgr, mask, **kwargs)

    harmonized_bgr = np.asarray(harmonized_bgr)
    if harmonized_bgr.dtype != np.uint8:
        harmonized_bgr = np.clip(harmonized_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(harmonized_bgr, cv2.COLOR_BGR2RGB)


def _get_harmonization_model(config: Any) -> "_BaseHarmonizer":
    model_type = normalize_harmonization_model_type(config.harmonization_model_type)
    device = _resolve_device(config.harmonization_device)
    dtype = _resolve_dtype(model_type, device)
    cache_key = (model_type, str(device), dtype)
    with _MODEL_CACHE_LOCK:
        if cache_key not in _MODEL_CACHE:
            if model_type == "LBM":
                _MODEL_CACHE[cache_key] = _LBMHarmonizer(device=device, dtype=dtype)
            else:
                _MODEL_CACHE[cache_key] = _PCTNetHarmonizer(device=device)
        return _MODEL_CACHE[cache_key]


class _BaseHarmonizer:
    def __call__(self, composite_bgr: np.ndarray, composite_mask: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError


class _PCTNetHarmonizer(_BaseHarmonizer):
    """Direct PCTNet inference without Libcom's CUDA-only wrapper."""

    def __init__(self, device: torch.device) -> None:
        _prepare_libcom_imports()
        from libcom.image_harmonization.source.pct_net import PCTNet

        self.device = device
        model_root = _libcom_model_root()
        weight_path = model_root / "pretrained_models" / "PCTNet.pth"
        lut_path = model_root / "pretrained_models" / "IdentityLUT33.txt"
        with _download_lock(model_root / "pretrained_models" / ".pctnet_download.lock"):
            _download_pretrained_file(weight_path)
            _download_pretrained_file(lut_path)
        model = PCTNet()
        model.load_state_dict(_load_torch_state_dict(weight_path))
        self.model = model.to(self.device).eval()
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def __call__(self, composite_bgr: np.ndarray, composite_mask: np.ndarray, **kwargs: Any) -> np.ndarray:
        image_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
        mask = _ensure_gray_mask(composite_mask).astype(np.float32) / 255.0
        image_lr = cv2.resize(image_rgb, (256, 256))
        mask_lr = cv2.resize(mask, (256, 256))

        image_t = self.to_tensor(image_rgb).float().to(self.device)
        mask_t = self.to_tensor(mask).float().to(self.device)
        image_lr_t = self.to_tensor(image_lr).float().to(self.device)
        mask_lr_t = self.to_tensor(mask_lr).float().to(self.device)

        outputs = self.model(image_lr_t, image_t, mask_lr_t, mask_t)
        if outputs.ndim == 4:
            outputs = outputs.squeeze(0)
        output_rgb = torch.clamp(255.0 * outputs.permute(1, 2, 0), 0, 255).cpu().numpy()
        return cv2.cvtColor(output_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


class _LBMHarmonizer(_BaseHarmonizer):
    """Direct LBM inference without hard-coded CUDA tensor moves."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        _prepare_libcom_imports()
        from lbm.inference import get_model

        self.device = device
        self.dtype = dtype
        model_root = _libcom_model_root()
        lbm_dir = model_root / "pretrained_models" / "lbm_ckpt"
        with _download_lock(model_root / "pretrained_models" / ".lbm_download.lock"):
            _download_pretrained_folder(lbm_dir)
        self.model = get_model(str(lbm_dir), torch_dtype=dtype, device=str(device))
        self.model.bridge_noise_sigma = 0.005
        if self.model.sampling_noise_scheduler is None:
            self.model.sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="scheduler",
            )
        self.model.eval()
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def __call__(self, composite_bgr: np.ndarray, composite_mask: np.ndarray, **kwargs: Any) -> np.ndarray:
        steps = int(kwargs.get("steps", 4))
        inference_size = int(kwargs.get("resolution", 1024))
        if inference_size % 8 != 0:
            raise ValueError("harmonization_resolution must be divisible by 8 for LBM.")
        latent_size = inference_size // 8

        source_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(source_rgb)
        mask = Image.fromarray(_ensure_gray_mask(composite_mask)).convert("L")
        original_width, original_height = source.size

        source_resized = source.resize((inference_size, inference_size), Image.Resampling.BILINEAR)
        source_t = (self.to_tensor(source_resized).unsqueeze(0) * 2 - 1).to(
            self.device,
            dtype=self.dtype,
        )
        batch = {"source_image_paste": source_t}

        mask_latent = mask.resize((latent_size, latent_size), Image.Resampling.BILINEAR)
        mask_t = self.to_tensor(mask_latent).unsqueeze(0).to(self.device, dtype=self.dtype)

        z_source = self.model.vae.encode(batch["source_image_paste"])
        output_tensor = self.model.sample(
            z=z_source,
            num_steps=steps,
            conditioner_inputs=batch,
            max_samples=1,
            mask=mask_t,
        ).clamp(-1, 1)

        output_rgb = (output_tensor[0].float().cpu() + 1) / 2
        output_rgb = torch.clamp(255.0 * output_rgb.permute(1, 2, 0), 0, 255).numpy().astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(output_bgr, (original_width, original_height))


def _prepare_libcom_imports() -> None:
    libcom_root = Path(__file__).resolve().parents[1] / "libcom"
    if str(libcom_root) not in sys.path:
        sys.path.insert(0, str(libcom_root))

    libcom_pkg = libcom_root / "libcom"
    _ensure_namespace_package("libcom", libcom_pkg)
    _ensure_namespace_package("libcom.image_harmonization", libcom_pkg / "image_harmonization")
    _ensure_namespace_package(
        "libcom.image_harmonization.source",
        libcom_pkg / "image_harmonization" / "source",
    )
    _ensure_namespace_package("libcom.utils", libcom_pkg / "utils")

    lbm_src = libcom_root / "libcom" / "image_harmonization" / "source" / "src"
    if str(lbm_src) not in sys.path:
        sys.path.insert(0, str(lbm_src))


def _ensure_namespace_package(module_name: str, path: Path) -> None:
    """Expose Libcom subpackages without executing Libcom's broad top-level imports."""

    path_str = str(path)
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__path__ = [path_str]
        module.__package__ = module_name
        sys.modules[module_name] = module
        return

    module_paths = getattr(module, "__path__", None)
    if module_paths is not None and path_str not in module_paths:
        module_paths.append(path_str)


def _libcom_model_root() -> Path:
    default_root = Path(__file__).resolve().parents[1] / "libcom" / "libcom" / "image_harmonization"
    return Path(os.environ.get("LIBCOM_MODEL_DIR", default_root)).resolve()


def _download_pretrained_file(target_path: Path) -> Path:
    """Download one Libcom checkpoint file with the current Hugging Face Hub API."""

    if target_path.exists():
        if not target_path.is_file():
            raise FileExistsError(f"Expected checkpoint file, found directory: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        _hf_hub_download(
            repo_id=_LIBCOM_HF_REPO,
            filename=target_path.name,
            cache_dir=str(target_path.parent),
        )
    )
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Download failed for {target_path.name}: {downloaded_path}")

    shutil.copyfile(downloaded_path, target_path, follow_symlinks=True)
    return target_path


def _download_pretrained_folder(folder_path: Path) -> Path:
    """Download and unpack a Libcom checkpoint folder archive."""

    if folder_path.exists():
        if not folder_path.is_dir():
            raise FileExistsError(f"Expected checkpoint directory, found file: {folder_path}")
        if any(folder_path.iterdir()):
            return folder_path
        shutil.rmtree(folder_path)

    zip_path = folder_path.with_name(f"{folder_path.name}.zip")
    _download_pretrained_file(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(folder_path.parent)
    zip_path.unlink(missing_ok=True)

    if not folder_path.is_dir() or not any(folder_path.iterdir()):
        raise FileNotFoundError(f"Downloaded archive did not create checkpoint folder: {folder_path}")
    return folder_path


def _hf_hub_download(**kwargs: Any) -> str:
    """Small wrapper so tests can patch Hugging Face downloads without network access."""

    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


@contextmanager
def _download_lock(lock_path: Path) -> Iterator[None]:
    """Serialize first-time checkpoint downloads across process workers."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        if fcntl is None:
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_dtype(model_type: str, device: torch.device) -> torch.dtype:
    if model_type != "LBM":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_torch_state_dict(path: Path) -> Any:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _ensure_gray_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return mask[:, :, 0]
    return mask


def _resolve_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.isdigit():
        if not torch.cuda.is_available():
            raise ValueError("CUDA device id was requested, but CUDA is not available.")
        return torch.device(f"cuda:{requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("'mps' was requested, but MPS is not available.")
        return torch.device("mps")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"{device!r} was requested, but CUDA is not available.")
        return torch.device(requested)
    raise ValueError("harmonization_device must be auto, cpu, mps, cuda, cuda:N, or a CUDA device id.")


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _instance_with_mask(instance: Any, mask: np.ndarray) -> Any:
    return type(instance)(
        category_id=instance.category_id,
        mask=mask,
        source_annotation_id=instance.source_annotation_id,
    )


def _select_object_indices(
    n_objects: int,
    *,
    percent: float,
    max_objects: int | None,
    rng: np.random.Generator,
) -> list[int]:
    if n_objects <= 0:
        return []
    n_select = max(1, math.ceil(n_objects * validate_subset_percent(percent) / 100.0))
    if max_objects is not None:
        n_select = min(n_select, int(max_objects))
    n_select = min(n_select, n_objects)
    return sorted(rng.choice(n_objects, size=n_select, replace=False).tolist())


def _scale_jitter_flip_to_canvas(
    sample: Any,
    *,
    target_h: int,
    target_w: int,
    rng: np.random.Generator,
    scale_min: float,
    scale_max: float,
    flip_prob: float,
) -> Any:
    src_h, src_w = sample.image.shape[:2]
    scale = float(rng.uniform(scale_min, scale_max))
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    scaled_image = np.asarray(
        Image.fromarray(sample.image).resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    scaled_masks = [
        np.asarray(
            Image.fromarray(instance.mask).resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        for instance in sample.instances
    ]

    image_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    mask_canvases = [np.zeros((target_h, target_w), dtype=np.uint8) for _ in scaled_masks]

    src_x, dst_x, copy_w = _crop_or_pad_offsets(scaled_w, target_w, rng)
    src_y, dst_y, copy_h = _crop_or_pad_offsets(scaled_h, target_h, rng)
    image_canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = scaled_image[
        src_y : src_y + copy_h,
        src_x : src_x + copy_w,
    ]
    for canvas, mask in zip(mask_canvases, scaled_masks, strict=True):
        canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = mask[
            src_y : src_y + copy_h,
            src_x : src_x + copy_w,
        ]

    if rng.random() < flip_prob:
        image_canvas = np.ascontiguousarray(image_canvas[:, ::-1])
        mask_canvases = [np.ascontiguousarray(mask[:, ::-1]) for mask in mask_canvases]

    instances = [
        _instance_with_mask(instance, mask)
        for instance, mask in zip(sample.instances, mask_canvases, strict=True)
        if mask.any()
    ]
    return type(sample)(sample.image_id, sample.file_name, image_canvas, instances)


def _crop_or_pad_offsets(
    source_size: int,
    target_size: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if source_size >= target_size:
        src_start = int(rng.integers(0, source_size - target_size + 1))
        return src_start, 0, target_size
    dst_start = int(rng.integers(0, target_size - source_size + 1))
    return 0, dst_start, source_size
