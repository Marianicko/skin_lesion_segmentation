from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from shutil import copyfile, rmtree

import torch
from accelerate import Accelerator
from torch import nn
from utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Checkpoint:
    metric_val: float
    loss_val: float  # NEW
    epoch: int
    save_path: Path


class CheckpointSaver:
    def __init__(
            self,
            accelerator: Accelerator,
            model: nn.Module,
            metric_name: str,
            save_dir: str,
            rm_save_dir: bool = False,
            max_history: int = 1,
            should_minimize: bool = True,
    ) -> None:
        self._accelerator = accelerator
        self._model = model
        self.metric_name = metric_name
        self.save_dir = Path(save_dir)
        self.max_history = max_history
        self.should_minimize = should_minimize
        self._storage: list[Checkpoint] = []
        self.best_loss = float('inf')  # NEW

        if os.path.exists(save_dir) and rm_save_dir:
            rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    def save(self, metric_val: float, loss_val: float, epoch: int) -> None:
        save_name_prefix = f"model_e{epoch:03d}_checkpoint"
        save_path = self._save_checkpoint(
            model=self._model, epoch=epoch, save_name_prefix=save_name_prefix
        )

        # Сохраняем чекпоинт с loss (NEW)
        self._storage.append(
            Checkpoint(
                metric_val=metric_val,
                loss_val=loss_val,  # NEW
                epoch=epoch,
                save_path=save_path
            )
        )

        # Сортируем по основной метрике (IoU)
        self._storage = sorted(
            self._storage,
            key=lambda x: x.metric_val,
            reverse=not self.should_minimize
        )

        # Удаляем худшие чекпоинты
        if len(self._storage) > self.max_history:
            worst_item = self._storage.pop()
            os.remove(worst_item.save_path)

        # Сохраняем лучший по IoU
        copyfile(
            src=self._storage[0].save_path,
            dst=self.save_dir / "model_checkpoint_best.pt",
        )

        # NEW: Сохраняем лучший по loss
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            copyfile(
                src=self._storage[0].save_path,
                dst=self.save_dir / "model_checkpoint_best_loss.pt",
            )

        LOGGER.info(
            f"Best epoch %s: IoU=%.4f, Loss=%.4f (epoch %d)",
            self.metric_name,
            self._storage[0].metric_val,
            self._storage[0].loss_val,  # NEW
            self._storage[0].epoch + 1,
        )

    def _save_checkpoint(self, model: nn.Module, epoch: int, save_name_prefix: str) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)

        # Получаем состояние оптимизатора
        optimizer_state = self._accelerator.optimizer_state_dict(self._optimizer) if hasattr(self,
                                                                                             '_optimizer') else None

        self._accelerator.save(
            obj={
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer_state,
                "metric": self._storage[0].metric_val if self._storage else 0.0,
                "loss": self._storage[0].loss_val if self._storage else 0.0,
            },
            f=save_path,
        )
        return Path(save_path)
    '''
    def _save_checkpoint(
            self, model: nn.Module, epoch: int, save_name_prefix: str
    ) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)
        self._accelerator.save(
            obj={
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "metric": self._storage[0].metric_val if self._storage else 0.0,
                "loss": self._storage[0].loss_val if self._storage else 0.0,  # NEW
            },
            f=save_path,
        )
        return Path(save_path)
    '''