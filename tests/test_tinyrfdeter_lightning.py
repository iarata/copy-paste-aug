from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from cpa.tinyrfdeter.data import CocoPremadeDataModule
from cpa.tinyrfdeter.lightning import checkpoint_dir_for_run, training_dataset_name


class _TinyInstanceDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, 4, 4, dtype=torch.uint8),
            "image_id": torch.tensor(index),
        }


def test_training_dataset_name_uses_safe_data_root_folder():
    assert training_dataset_name(Path("data/processed/coco2017 harmonized/lbm")) == "lbm"
    assert training_dataset_name(Path("data/processed/coco2017 harmonized seed42")) == (
        "coco2017_harmonized_seed42"
    )
    assert training_dataset_name(Path("/")) == "dataset"


def test_checkpoint_dir_for_run_includes_variant_and_dataset_name(tmp_path: Path):
    checkpoint_dir = checkpoint_dir_for_run(
        tmp_path / "outputs" / "tinyrfdeter",
        variant="n",
        data_root=Path("data/processed/coco2017_harmonized_lbm_seed42_sub50"),
    )

    assert checkpoint_dir == (
        tmp_path / "outputs" / "tinyrfdeter" / "rf-deter-seg-n-coco2017_harmonized_lbm_seed42_sub50"
    )


def test_tinyrfdeter_datamodule_uses_separate_validation_loader_settings(tmp_path: Path):
    dm = CocoPremadeDataModule(
        tmp_path / "train",
        val_root=tmp_path / "val",
        image_size=4,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        val_batch_size=2,
        val_num_workers=0,
        val_pin_memory=False,
        val_persistent_workers=False,
    )
    dm.train_dataset = _TinyInstanceDataset()
    dm.val_dataset = _TinyInstanceDataset()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 2
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    assert not train_loader.pin_memory
    assert not val_loader.pin_memory
