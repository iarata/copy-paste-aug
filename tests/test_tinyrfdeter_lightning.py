from __future__ import annotations

from pathlib import Path

from cpa.tinyrfdeter.lightning import checkpoint_dir_for_run, training_dataset_name


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
