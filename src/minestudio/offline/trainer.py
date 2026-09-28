'''
Date: 2024-11-10 13:44:13
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 13:52:32
FilePath: /MineStudio/minestudio/offline/trainer.py
'''
import os
import torch
import torch.nn as nn
import lightning as L
from rich import print
from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks import ObjectiveCallback
from typing import List

IMPORTANT_VARIABLES = [
    "MINESTUDIO_SAVE_DIR", 
    "MINESTUDIO_DATABASE_DIR", 
]

for var in IMPORTANT_VARIABLES:
    val = os.environ.get(var, "not found")
    print(f"[Env Variable]  {var}: {val}")

def tree_detach(tree):
    """
    Detaches a tree of tensors from the computation graph.

    This function recursively traverses a nested structure (dictionary or list)
    and detaches any PyTorch tensors it encounters. This is useful for
    preventing gradients from flowing back through the detached tensors.

    :param tree: The nested structure (dict, list, or tensor) to detach.
    :type tree: dict | list | torch.Tensor
    :returns: The detached tree.
    :rtype: dict | list | torch.Tensor
    """
    if isinstance(tree, dict):
        return {k: tree_detach(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_detach(v) for v in tree]
    elif isinstance(tree, torch.Tensor):
        return tree.detach()
    else:
        return tree

class MineLightning(L.LightningModule):
    """
    A PyTorch Lightning module for training MinePolicy models.

    This class encapsulates the training, validation, and optimization logic
    for MinePolicy models. It handles memory management, batch processing,
    and integration with ObjectiveCallbacks for custom training objectives.
    """

    def __init__(
        self, 
        mine_policy: MinePolicy, 
        callbacks: List[ObjectiveCallback] = [], 
        hyperparameters: dict = {},
        *,
        log_freq: int = 20,
        learning_rate: float = 1e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
    ):
        """
        Initializes the MineLightning module.

        :param mine_policy: The MinePolicy model to train.
        :type mine_policy: MinePolicy
        :param callbacks: A list of ObjectiveCallback instances for custom objectives.
        :type callbacks: List[ObjectiveCallback]
        :param hyperparameters: A dictionary of hyperparameters to save.
        :type hyperparameters: dict
        :param log_freq: The frequency (in batches) for logging metrics.
        :type log_freq: int
        :param learning_rate: The learning rate for the optimizer.
        :type learning_rate: float
        :param warmup_steps: The number of warmup steps for the learning rate scheduler.
        :type warmup_steps: int
        :param weight_decay: The weight decay for the optimizer.
        :type weight_decay: float
        """
        super().__init__()
        self.mine_policy = mine_policy
        self.callbacks = callbacks
        self.log_freq = log_freq
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps 
        self.weight_decay = weight_decay
        self.memory_dict = {
            "memory": None, 
            "init_memory": None, 
            "last_timestamp": None,
        }
        self.automatic_optimization = True
        self.save_hyperparameters(hyperparameters)

    def _make_memory(self, batch):
        """
        Manages the recurrent memory for the MinePolicy model.

        This function initializes and updates the memory state based on the
        batch data. It handles episode boundaries by resetting the memory
        when a new episode begins.

        :param batch: The input batch data.
        :type batch: dict
        :returns: The current memory state.
        :rtype: dict
        """
        if self.memory_dict["init_memory"] is None:
            self.memory_dict["init_memory"] = self.mine_policy.initial_state(batch['image'].shape[0])
        if self.memory_dict["memory"] is None:
            self.memory_dict["memory"] = self.memory_dict["init_memory"]
        if self.memory_dict["last_timestamp"] is None:
            self.memory_dict["last_timestamp"] = torch.zeros(batch['image'].shape[0], dtype=torch.long).to(self.device)
        boe = batch["timestamp"][:, 0].ne(self.memory_dict["last_timestamp"] + 1)
        self.memory_dict["last_timestamp"] = batch["timestamp"][:, -1]
        # if boe's (begin-of-episode) item is True, then we keep the original memory, otherwise we reset the memory
        mem_cache = []
        for om, im in zip(self.memory_dict["memory"], self.memory_dict["init_memory"]):
            boe_f = boe[:, None, None].expand_as(om)
            mem_line = torch.where(boe_f, im, om)
            mem_cache.append(mem_line)
        self.memory_dict["memory"] = mem_cache
        return self.memory_dict["memory"]

    def _batch_step(self, batch, batch_idx, step_name):
        """
        Performs a single training or validation step.

        This function processes a batch of data, computes the model output,
        calculates the loss using the registered callbacks, and logs the metrics.

        :param batch: The input batch data.
        :type batch: dict
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :param step_name: The name of the current step (e.g., 'train', 'val').
        :type step_name: str
        :returns: A dictionary containing the loss and other metrics.
        :rtype: dict
        """
        result = {'loss': 0}
        memory_in = self._make_memory(batch)
        for callback in self.callbacks:
            batch = callback.before_step(batch, batch_idx, step_name)
        # memory_in = None
        latents, memory_out = self.mine_policy(batch, memory_in)
        self.memory_dict["memory"] = tree_detach(memory_out)
        for callback in self.callbacks:
            call_result = callback(batch, batch_idx, step_name, latents, self.mine_policy)
            for key, val in call_result.items():
                result[key] = result.get(key, 0) + val

        if batch_idx % self.log_freq == 0:
            for key, val in result.items():
                prog_bar = ('loss' in key) and (step_name == 'train')
                self.log(f'{step_name}/{key}', val, sync_dist=False, prog_bar=prog_bar)

        return result

    def training_step(self, batch, batch_idx):
        """
        Performs a training step.

        This function calls _batch_step with step_name='train'.

        :param batch: The input batch data.
        :type batch: dict
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :returns: A dictionary containing the loss and other metrics.
        :rtype: dict
        """
        return self._batch_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step.

        This function calls _batch_step with step_name='val'.

        :param batch: The input batch data.
        :type batch: dict
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :returns: A dictionary containing the loss and other metrics.
        :rtype: dict
        """
        return self._batch_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        This function sets up an AdamW optimizer and a linear warmup learning
        rate scheduler.

        :returns: A dictionary containing the optimizer and learning rate scheduler.
        :rtype: dict
        """
        optimizer = torch.optim.AdamW(
            params=self.mine_policy.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/self.warmup_steps, 1)
        )
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
        }

if __name__ == '__main__':
    ...