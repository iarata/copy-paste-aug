from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch

from cpa.datasets import Coco2017DataModule
from cpa.modeling import SimpleInstanceSegmentationTransformerModule, evaluate_mask_map95
from cpa.utils.configs import Config, register_configs
from cpa.yolo.data import COCOJsonDataModule, resolve_path
from cpa.yolo.lightning import YOLO26LightningModule, evaluate_checkpoint

register_configs()


def resolve_precision(precision: str) -> str:
    if precision != "auto":
        return precision
    if torch.cuda.is_available():
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def maybe_make_wandb_logger(cfg: Config) -> WandbLogger | None:
    if cfg.wandb.mode:
        os.environ["WANDB_MODE"] = str(cfg.wandb.mode)

    if str(cfg.wandb.mode).lower() == "disabled":
        return None

    experiment_name = cfg.experiment_name or Path.cwd().name
    tags = list(cfg.wandb.tags) if isinstance(cfg.wandb.tags, list) else None
    return WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=experiment_name,
        save_dir=str(Path.cwd()),
        log_model=bool(cfg.wandb.log_model),
        tags=tags,
    )


def log_wandb_run_config(wandb_logger: WandbLogger | None, payload: dict[str, object]) -> None:
    if wandb_logger is None:
        return
    wandb_logger.log_hyperparams(payload)


def update_wandb_summary(wandb_logger: WandbLogger | None, payload: dict[str, object]) -> None:
    if wandb_logger is None:
        return

    summary = getattr(wandb_logger.experiment, "summary", None)
    if hasattr(summary, "update"):
        summary.update(payload)


def build_trainer(cfg: Config, wandb_logger: WandbLogger | None, default_root_dir: Path) -> L.Trainer:
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_root_dir / "checkpoints",
        filename="epoch-{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    trainer = L.Trainer(
        default_root_dir=str(default_root_dir),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=resolve_precision(cfg.training.precision),
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        deterministic=bool(cfg.debug),
        fast_dev_run=1 if cfg.debug else False,
    )
    return trainer


def run_fit(cfg: Config, project_root: Path) -> None:
    if getattr(cfg.models, "architecture", "yolo26") == "simple_instance_transformer":
        datamodule = Coco2017DataModule(cfg.dataset, seed=cfg.seed)
        module = SimpleInstanceSegmentationTransformerModule(cfg)
        wandb_logger = maybe_make_wandb_logger(cfg)
        log_wandb_run_config(
            wandb_logger,
            {
                "model_name": cfg.models.name,
                "architecture": cfg.models.architecture,
                "task": cfg.dataset.task,
                "augmentation": cfg.dataset.augmentations.name,
                "copy_paste_prob": cfg.dataset.augmentations.prob,
                "train_subset_percent": getattr(cfg.dataset, "train_subset_percent", 100.0),
                "val_subset_percent": getattr(cfg.dataset, "val_subset_percent", 100.0),
                "seed": cfg.seed,
                "debug": bool(cfg.debug),
            },
        )
        trainer = build_trainer(cfg, wandb_logger, Path.cwd())
        resume = cfg.training.resume_from_checkpoint
        ckpt_path = str(resolve_path(resume, project_root)) if resume else None
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        if trainer.fast_dev_run:
            logger.info("Skipping post-fit mAP95 test because fast_dev_run is enabled.")
            return
        if cfg.evaluation.run_after_fit:
            datamodule.setup("validate")
            metrics = evaluate_mask_map95(module, datamodule.val_dataloader())
            logger.info("Post-fit simple transformer metrics: {}", metrics)
            if wandb_logger is not None:
                wandb_logger.log_metrics(metrics, step=int(trainer.global_step))
                update_wandb_summary(wandb_logger, metrics)
        return

    eval_batch_size = cfg.evaluation.batch_size or cfg.dataset.batch_size
    datamodule = COCOJsonDataModule(
        cfg.dataset,
        project_root=project_root,
        eval_batch_size=eval_batch_size,
        seed=cfg.seed,
    )
    module = YOLO26LightningModule(cfg, datamodule.names, project_root=project_root)
    wandb_logger = maybe_make_wandb_logger(cfg)
    log_wandb_run_config(
        wandb_logger,
        {
            "model_name": cfg.models.name,
            "model_scale": getattr(cfg.models, "scale", None),
            "model_weights": getattr(cfg.models, "weights", None),
            "resolved_model_source": module.resolved_model_source,
            "task": cfg.dataset.task,
            "train_subset_percent": getattr(cfg.dataset, "train_subset_percent", 100.0),
            "val_subset_percent": getattr(cfg.dataset, "val_subset_percent", 100.0),
            "seed": cfg.seed,
            "debug": bool(cfg.debug),
        },
    )
    trainer = build_trainer(cfg, wandb_logger, Path.cwd())

    resume = cfg.training.resume_from_checkpoint
    ckpt_path = None
    if resume:
        ckpt_path = str(resolve_path(resume, project_root))

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if trainer.fast_dev_run:
        logger.info("Skipping post-fit evaluation because fast_dev_run is enabled.")
        return

    if cfg.evaluation.run_after_fit:
        best_checkpoint = (
            trainer.checkpoint_callback.best_model_path or trainer.checkpoint_callback.last_model_path
        )
        if not best_checkpoint:
            logger.warning("No checkpoint was produced during training, skipping post-fit evaluation.")
            return

        datamodule.setup("validate")
        metrics = evaluate_checkpoint(
            cfg,
            checkpoint_path=best_checkpoint,
            datamodule=datamodule,
            project_root=project_root,
            output_dir=Path.cwd() / "evaluation",
        )
        logger.info("Evaluation metrics: {}", metrics)
        update_wandb_summary(wandb_logger, metrics)


def run_eval(cfg: Config, project_root: Path) -> None:
    checkpoint_path = cfg.evaluation.checkpoint_path or cfg.training.resume_from_checkpoint
    if checkpoint_path is None:
        raise ValueError("Set `evaluation.checkpoint_path` when running in evaluation mode.")

    eval_batch_size = cfg.evaluation.batch_size or cfg.dataset.batch_size
    datamodule = COCOJsonDataModule(
        cfg.dataset,
        project_root=project_root,
        eval_batch_size=eval_batch_size,
        seed=cfg.seed,
    )
    datamodule.setup("validate")
    metrics = evaluate_checkpoint(
        cfg,
        checkpoint_path=resolve_path(checkpoint_path, project_root),
        datamodule=datamodule,
        project_root=project_root,
        output_dir=Path.cwd() / "evaluation",
    )
    logger.info("Evaluation metrics: {}", metrics)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    typed_cfg: Config = cast(Config, OmegaConf.to_object(cfg))
    project_root = Path(hydra.utils.get_original_cwd())
    configure_torch_runtime()
    seed = int(typed_cfg.seed)
    L.seed_everything(seed, workers=True)

    logger.info(
        "Running {} pipeline in {} mode with seed {}",
        getattr(typed_cfg.models, "architecture", "yolo26"),
        typed_cfg.training.mode,
        seed,
    )
    if typed_cfg.training.mode == "fit":
        run_fit(typed_cfg, project_root)
    elif typed_cfg.training.mode == "eval":
        run_eval(typed_cfg, project_root)
    else:
        raise ValueError(f"Unsupported training.mode: {typed_cfg.training.mode}")


if __name__ == "__main__":
    main()
