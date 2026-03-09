# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
import random
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


class TrainingSummaryCallback(pl.Callback):
    def __init__(self, monitor: str, mode: str, min_delta: float = 0.0) -> None:
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported monitor mode: {mode}")

        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)

        self.final_epoch_reached: int | None = None
        self.last_improvement_epoch: int | None = None
        self.last_improvement_delta: float | None = None

        self._best_score: float | None = None

    def _is_improvement(self, score: float) -> bool:
        if self._best_score is None:
            return False
        if self.mode == "min":
            return score < (self._best_score - self.min_delta)
        return score > (self._best_score + self.min_delta)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        if isinstance(metric, torch.Tensor):
            score = float(metric.detach().cpu().item())
        else:
            score = float(metric)

        if self._best_score is None:
            self._best_score = score
            return

        if self._is_improvement(score):
            if self.mode == "min":
                delta = self._best_score - score
            else:
                delta = score - self._best_score
            self.last_improvement_delta = float(delta)
            self.last_improvement_epoch = int(trainer.current_epoch)
            self._best_score = score

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.final_epoch_reached = int(trainer.current_epoch)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    def _subsample_sessions(sessions: list[Path], fraction: float) -> list[Path]:
        if fraction >= 1.0:
            return sessions
        if fraction <= 0.0:
            raise ValueError("train_session_fraction must be > 0")

        num_sessions = max(1, int(round(len(sessions) * fraction)))
        rng = random.Random(config.seed)
        return rng.sample(sessions, k=num_sessions)

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_subsample_sessions(
            _full_session_paths(config.dataset.train),
            config.train_session_fraction,
        ),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    early_stopping_min_delta = 0.0
    for cfg in callback_configs:
        target = cfg.get("_target_", "")
        if target == "pytorch_lightning.callbacks.EarlyStopping":
            early_stopping_min_delta = float(cfg.get("min_delta", 0.0))
            break
    summary_callback = TrainingSummaryCallback(
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        min_delta=early_stopping_min_delta,
    )
    callbacks.append(summary_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
        "final_epoch_reached": summary_callback.final_epoch_reached,
        "last_improvement_epoch": summary_callback.last_improvement_epoch,
        "last_improvement_delta": summary_callback.last_improvement_delta,
    }
    pprint.pprint(results, sort_dicts=False)

    # Save per-run results in Hydra's output directory to avoid overwriting
    # between runs.
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir.joinpath("results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
