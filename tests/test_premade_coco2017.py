from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from cpa.premade_datasets import coco2017 as premade_coco2017
from cpa.premade_datasets import harmonized_copy_paste
from cpa.premade_datasets.coco2017 import (
    PremadeCocoConfig,
    build_premade_coco2017,
    select_category_balanced_images,
)


def _rectangle(x: int, y: int, w: int, h: int) -> list[float]:
    return [
        float(x),
        float(y),
        float(x + w),
        float(y),
        float(x + w),
        float(y + h),
        float(x),
        float(y + h),
    ]


def _make_source_coco(root: Path) -> Path:
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    ann_dir = root / "annotations"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    for image_id in range(1, 7):
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(train_dir / f"{image_id:012d}.jpg")
    for image_id in range(101, 107):
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(val_dir / f"{image_id:012d}.jpg")

    train_images = [
        {"id": image_id, "file_name": f"{image_id:012d}.jpg", "height": 64, "width": 64}
        for image_id in range(1, 7)
    ]
    train_specs = [
        (1, 1, 5, 5),
        (2, 1, 8, 8),
        (3, 1, 10, 10),
        (4, 2, 12, 12),
        (5, 2, 14, 14),
        (6, 2, 16, 16),
    ]
    train_annotations = []
    for ann_id, (image_id, category_id, x, y) in enumerate(train_specs, start=1):
        train_annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [_rectangle(x, y, 18, 18)],
                "bbox": [float(x), float(y), 18.0, 18.0],
                "area": 324.0,
                "iscrowd": 0,
            }
        )

    val_images = [
        {"id": image_id, "file_name": f"{image_id:012d}.jpg", "height": 64, "width": 64}
        for image_id in range(101, 107)
    ]
    val_specs = [
        (101, 1, 8, 8),
        (102, 1, 10, 10),
        (103, 1, 12, 12),
        (104, 2, 14, 14),
        (105, 2, 16, 16),
        (106, 2, 18, 18),
    ]
    val_annotations = []
    for ann_id, (image_id, category_id, x, y) in enumerate(val_specs, start=1):
        val_annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [_rectangle(x, y, 12, 12)],
                "bbox": [float(x), float(y), 12.0, 12.0],
                "area": 144.0,
                "iscrowd": 0,
            }
        )
    categories = [{"id": 1, "name": "cat_1"}, {"id": 2, "name": "cat_2"}]
    (ann_dir / "instances_train2017.json").write_text(
        json.dumps({"images": train_images, "annotations": train_annotations, "categories": categories}),
        encoding="utf-8",
    )
    (ann_dir / "instances_val2017.json").write_text(
        json.dumps({"images": val_images, "annotations": val_annotations, "categories": categories}),
        encoding="utf-8",
    )
    return root


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_generated_image(output_root: Path) -> np.ndarray:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    file_name = manifest["generated_images"][0]["file_name"]
    return np.asarray(Image.open(output_root / "train2017" / file_name).convert("RGB"), dtype=np.uint8)


def test_category_balanced_subset_keeps_each_class_coverage(tmp_path: Path):
    source = _make_source_coco(tmp_path / "source")
    coco = json.loads((source / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))

    selected = select_category_balanced_images(coco, percent=50.0, seed=7)
    selected_ids = {image["id"] for image in selected}
    selected_categories = {
        annotation["category_id"]
        for annotation in coco["annotations"]
        if annotation["image_id"] in selected_ids
    }

    assert selected_categories == {1, 2}
    assert (
        len(
            [
                ann
                for ann in coco["annotations"]
                if ann["image_id"] in selected_ids and ann["category_id"] == 1
            ]
        )
        >= 2
    )
    assert (
        len(
            [
                ann
                for ann in coco["annotations"]
                if ann["image_id"] in selected_ids and ann["category_id"] == 2
            ]
        )
        >= 2
    )


def test_build_premade_coco_is_reproducible_and_separate(tmp_path: Path):
    source = _make_source_coco(tmp_path / "source")
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    common = {
        "source_root": source,
        "method": "simple",
        "seed": 11,
        "train_subset_percent": 50.0,
        "copy_paste_percent": 50.0,
        "objects_paste_percent": 100.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "flip_prob": 0.0,
        "blend": False,
        "link_mode": "copy",
        "show_progress": False,
    }

    build_premade_coco2017(PremadeCocoConfig(output_root=output_a, **common))
    build_premade_coco2017(PremadeCocoConfig(output_root=output_b, **common))

    train_a = json.loads((output_a / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    train_b = json.loads((output_b / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    val_a = json.loads((output_a / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    val_b = json.loads((output_b / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    manifest_a = json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((output_b / "manifest.json").read_text(encoding="utf-8"))

    assert train_a == train_b
    assert val_a == val_b
    assert manifest_a["selected_train_image_ids"] == manifest_b["selected_train_image_ids"]
    assert manifest_a["selected_val_image_ids"] == manifest_b["selected_val_image_ids"]
    assert manifest_a["effective_val_subset_percent"] == 50.0
    assert len(manifest_a["selected_val_image_ids"]) < 6
    assert {annotation["category_id"] for annotation in val_a["annotations"]} == {1, 2}
    assert len(manifest_a["generated_images"]) == len(
        (output_a / "lists" / "train_augmented.txt").read_text().splitlines()
    )
    for record in manifest_a["generated_images"]:
        file_name = record["file_name"]
        assert _file_sha256(output_a / "train2017" / file_name) == _file_sha256(
            output_b / "train2017" / file_name
        )
    assert (source / "annotations" / "instances_train2017.json").exists()
    assert (output_a / "annotations" / "instances_train2017.json").exists()
    assert output_a != source


def test_harmonized_method_preserves_simple_randomness_with_identity_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _make_source_coco(tmp_path / "source")
    output_simple = tmp_path / "simple"
    output_harmonized = tmp_path / "harmonized"
    calls: list[tuple[tuple[int, ...], int]] = []

    class IdentityHarmonizer:
        def __call__(self, composite_image: np.ndarray, composite_mask: np.ndarray, **kwargs):
            calls.append((composite_image.shape, int(composite_mask.max())))
            return composite_image

    monkeypatch.setattr(
        harmonized_copy_paste,
        "_get_harmonization_model",
        lambda config: IdentityHarmonizer(),
    )
    common = {
        "source_root": source,
        "seed": 23,
        "train_subset_percent": 50.0,
        "copy_paste_percent": 50.0,
        "objects_paste_percent": 100.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "flip_prob": 0.0,
        "blend": False,
        "link_mode": "copy",
        "show_progress": False,
    }

    build_premade_coco2017(PremadeCocoConfig(output_root=output_simple, method="simple", **common))
    build_premade_coco2017(
        PremadeCocoConfig(
            output_root=output_harmonized,
            method="harmonized",
            harmonization_model_type="PCNet",
            harmonization_device="cpu",
            **common,
        )
    )

    simple_manifest = json.loads((output_simple / "manifest.json").read_text(encoding="utf-8"))
    harmonized_manifest = json.loads((output_harmonized / "manifest.json").read_text(encoding="utf-8"))

    assert simple_manifest["selected_train_image_ids"] == harmonized_manifest["selected_train_image_ids"]
    assert [
        (record["base_image_id"], record["paste_image_id"], record["paste_annotation_ids"])
        for record in simple_manifest["generated_images"]
    ] == [
        (record["base_image_id"], record["paste_image_id"], record["paste_annotation_ids"])
        for record in harmonized_manifest["generated_images"]
    ]
    np.testing.assert_array_equal(
        _first_generated_image(output_simple), _first_generated_image(output_harmonized)
    )
    assert calls
    assert calls[0][1] == 255


def test_resume_reuses_completed_harmonized_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _make_source_coco(tmp_path / "source")
    output = tmp_path / "resume"

    class FailingHarmonizer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, composite_image: np.ndarray, composite_mask: np.ndarray, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("stop after one completed task")
            return composite_image

    failing = FailingHarmonizer()
    monkeypatch.setattr(harmonized_copy_paste, "_get_harmonization_model", lambda config: failing)
    common = {
        "source_root": source,
        "output_root": output,
        "method": "harmonized",
        "seed": 31,
        "train_subset_percent": 50.0,
        "copy_paste_percent": 100.0,
        "objects_paste_percent": 100.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "flip_prob": 0.0,
        "blend": False,
        "link_mode": "copy",
        "workers": 1,
        "show_progress": False,
    }

    with pytest.raises(RuntimeError, match="stop after one completed task"):
        build_premade_coco2017(PremadeCocoConfig(**common))
    assert (output / ".premade_resume" / "tasks").exists()

    resumed_calls = 0

    class IdentityHarmonizer:
        def __call__(self, composite_image: np.ndarray, composite_mask: np.ndarray, **kwargs):
            nonlocal resumed_calls
            resumed_calls += 1
            return composite_image

    monkeypatch.setattr(
        harmonized_copy_paste, "_get_harmonization_model", lambda config: IdentityHarmonizer()
    )
    build_premade_coco2017(PremadeCocoConfig(resume=True, **common))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["generated_images"]) > 1
    assert resumed_calls == len(manifest["generated_images"]) - 1
    assert not (output / ".premade_resume").exists()


def test_harmonized_device_and_dtype_helpers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert harmonized_copy_paste._resolve_device("auto") == torch.device("cpu")
    assert harmonized_copy_paste._resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="MPS"):
        harmonized_copy_paste._resolve_device("mps")
    assert harmonized_copy_paste._resolve_dtype("LBM", torch.device("cpu")) == torch.float32
    assert harmonized_copy_paste._resolve_dtype("LBM", torch.device("mps")) == torch.float32
    assert harmonized_copy_paste._resolve_dtype("LBM", torch.device("cuda:0")) == torch.bfloat16
    assert harmonized_copy_paste.normalize_harmonization_model_type("PCNet") == "PCTNet"


def test_libcom_premade_imports_do_not_execute_top_level_init(monkeypatch: pytest.MonkeyPatch):
    for module_name in list(sys.modules):
        if module_name == "libcom" or module_name.startswith("libcom."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    harmonized_copy_paste._prepare_libcom_imports()
    pct_module = importlib.import_module("libcom.image_harmonization.source.pct_net")

    assert hasattr(pct_module, "PCTNet")
    assert getattr(sys.modules["libcom"], "__file__", None) is None
    assert "libcom.shadow_generation" not in sys.modules
    assert "libcom.reflection_generation" not in sys.modules


def test_harmonized_checkpoint_download_uses_current_huggingface_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cached_file = tmp_path / "hf-cache" / "PCTNet.pth"
    cached_file.parent.mkdir()
    cached_file.write_bytes(b"checkpoint")
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(cached_file)

    monkeypatch.setattr(harmonized_copy_paste, "_hf_hub_download", fake_hf_hub_download)

    target = tmp_path / "pretrained_models" / "PCTNet.pth"
    result = harmonized_copy_paste._download_pretrained_file(target)

    assert result == target
    assert target.read_bytes() == b"checkpoint"
    assert calls == [
        {
            "repo_id": "BCMIZB/Libcom_pretrained_models",
            "filename": "PCTNet.pth",
            "cache_dir": str(target.parent),
        }
    ]
    assert "progress" not in calls[0]


def test_harmonized_checkpoint_folder_download_extracts_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cached_zip = tmp_path / "hf-cache" / "lbm_ckpt.zip"
    cached_zip.parent.mkdir()
    with zipfile.ZipFile(cached_zip, "w") as archive:
        archive.writestr("lbm_ckpt/config.yaml", "model: test\n")
        archive.writestr("lbm_ckpt/weights.bin", b"weights")

    def fake_hf_hub_download(**kwargs):
        assert kwargs["filename"] == "lbm_ckpt.zip"
        return str(cached_zip)

    monkeypatch.setattr(harmonized_copy_paste, "_hf_hub_download", fake_hf_hub_download)

    folder = tmp_path / "pretrained_models" / "lbm_ckpt"
    result = harmonized_copy_paste._download_pretrained_folder(folder)

    assert result == folder
    assert (folder / "config.yaml").read_text(encoding="utf-8") == "model: test\n"
    assert (folder / "weights.bin").read_bytes() == b"weights"
    assert not (folder.parent / "lbm_ckpt.zip").exists()


def test_harmonized_accelerator_caps_default_in_flight(tmp_path: Path):
    source = tmp_path / "source"
    config = PremadeCocoConfig(
        source_root=source,
        output_root=tmp_path / "out",
        method="harmonized",
        harmonization_device="cuda",
        workers=12,
    )

    assert premade_coco2017._effective_max_in_flight(config, 100) == 1
    assert premade_coco2017._effective_max_in_flight(config, 0) == 0
    assert (
        premade_coco2017._effective_max_in_flight(
            PremadeCocoConfig(
                source_root=source,
                output_root=tmp_path / "out_override",
                method="harmonized",
                harmonization_device="cuda",
                workers=12,
                max_in_flight=3,
            ),
            100,
        )
        == 3
    )
    assert (
        premade_coco2017._effective_max_in_flight(
            PremadeCocoConfig(
                source_root=source,
                output_root=tmp_path / "out_cpu",
                method="harmonized",
                harmonization_device="cpu",
                workers=12,
            ),
            100,
        )
        == 24
    )


def test_cleanup_removes_original_symlink_aliases_and_keeps_dataset_loadable(tmp_path: Path):
    source = _make_source_coco(tmp_path / "source")
    output = tmp_path / "out"

    build_premade_coco2017(
        PremadeCocoConfig(
            source_root=source,
            output_root=output,
            method="simple",
            seed=17,
            train_subset_percent=50.0,
            copy_paste_percent=50.0,
            objects_paste_percent=100.0,
            scale_min=1.0,
            scale_max=1.0,
            flip_prob=0.0,
            blend=False,
            link_mode="symlink",
            show_progress=False,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    train_json = json.loads((output / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    source_train = json.loads(
        (source / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
    )
    source_image_by_id = {image["id"]: image for image in source_train["images"]}

    for image_id in manifest["selected_train_image_ids"]:
        source_file = source_image_by_id[image_id]["file_name"]
        assert not (output / "train2017" / source_file).exists()

    original_records = [
        image for image in train_json["images"] if image["id"] in set(manifest["selected_train_image_ids"])
    ]
    assert original_records
    for image in original_records:
        image_path = Path(image["file_name"])
        assert image_path.is_absolute()
        assert image_path.exists()

    for record in manifest["generated_images"]:
        assert (output / "train2017" / record["file_name"]).exists()
    assert manifest["cleanup"]["removed_train_aliases"] == len(manifest["selected_train_image_ids"])


def test_parallel_backends_match_single_worker_output(tmp_path: Path):
    source = _make_source_coco(tmp_path / "source")
    output_single = tmp_path / "single"
    output_thread = tmp_path / "thread"
    output_process = tmp_path / "process"
    common = {
        "source_root": source,
        "method": "simple",
        "seed": 19,
        "train_subset_percent": 50.0,
        "copy_paste_percent": 100.0,
        "objects_paste_percent": 100.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "flip_prob": 0.0,
        "blend": False,
        "link_mode": "copy",
        "show_progress": False,
    }

    build_premade_coco2017(PremadeCocoConfig(output_root=output_single, workers=1, **common))
    build_premade_coco2017(
        PremadeCocoConfig(output_root=output_thread, workers=2, parallel_backend="thread", **common)
    )
    build_premade_coco2017(
        PremadeCocoConfig(output_root=output_process, workers=2, parallel_backend="process", **common)
    )

    train_single = json.loads(
        (output_single / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
    )
    train_thread = json.loads(
        (output_thread / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
    )
    train_process = json.loads(
        (output_process / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output_single / "manifest.json").read_text(encoding="utf-8"))

    assert train_thread == train_single
    assert train_process == train_single
    for record in manifest["generated_images"]:
        file_name = record["file_name"]
        expected_hash = _file_sha256(output_single / "train2017" / file_name)
        assert _file_sha256(output_thread / "train2017" / file_name) == expected_hash
        assert _file_sha256(output_process / "train2017" / file_name) == expected_hash


def test_output_annotations_reference_existing_images(tmp_path: Path):
    source = _make_source_coco(tmp_path / "source")
    output = tmp_path / "out"

    build_premade_coco2017(
        PremadeCocoConfig(
            source_root=source,
            output_root=output,
            seed=3,
            train_subset_percent=50.0,
            copy_paste_percent=100.0,
            objects_paste_percent=100.0,
            scale_min=1.0,
            scale_max=1.0,
            flip_prob=0.0,
            blend=False,
            link_mode="copy",
            show_progress=False,
        )
    )

    train = json.loads((output / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    val = json.loads((output / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    image_ids = {image["id"] for image in train["images"]}
    val_image_ids = {image["id"] for image in val["images"]}
    assert {annotation["image_id"] for annotation in train["annotations"]}.issubset(image_ids)
    assert {annotation["image_id"] for annotation in val["annotations"]}.issubset(val_image_ids)
    assert all((output / "train2017" / image["file_name"]).exists() for image in train["images"])
    assert all((output / "val2017" / image["file_name"]).exists() for image in val["images"])
    assert (output / "coco_data.yaml").exists()
