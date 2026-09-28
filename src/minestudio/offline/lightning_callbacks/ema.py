# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable

import torch
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA) for PyTorch Lightning.

    This callback maintains moving averages of the trained parameters during training.
    When evaluating or testing, it can swap the original weights with the EMA weights.
    When saving a checkpoint, it saves an additional set of parameters with the prefix `ema`.

    :param decay: The exponential decay factor used for calculating the moving average. Must be between 0 and 1.
    :type decay: float
    :param validate_original_weights: If True, validates the original weights instead of the EMA weights. Defaults to False.
    :type validate_original_weights: bool
    :param every_n_steps: Apply EMA every N training steps. Defaults to 1.
    :type every_n_steps: int
    :param cpu_offload: If True, offloads EMA weights to CPU to save GPU memory. Defaults to False.
    :type cpu_offload: bool
    :raises MisconfigurationException: If the decay value is not between 0 and 1.
    """

    def __init__(
        self, decay: float, validate_original_weights: bool = False, every_n_steps: int = 1, cpu_offload: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the fit begins.

        Wraps the optimizers with EMAOptimizer.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        """
        device = pl_module.device if not self.cpu_offload else torch.device('cpu')
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the validation loop begins.

        Swaps to EMA weights if `validate_original_weights` is False.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the validation loop ends.

        Swaps back to original weights if EMA weights were used for validation.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the test loop begins.

        Swaps to EMA weights if `validate_original_weights` is False.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when the test loop ends.

        Swaps back to original weights if EMA weights were used for testing.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        """
        Determines if EMA weights should be used for validation/testing.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :returns: True if EMA weights should be used, False otherwise.
        :rtype: bool
        """
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        """
        Checks if EMA has been initialized (i.e., if any optimizer is an EMAOptimizer).

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :returns: True if EMA is initialized, False otherwise.
        :rtype: bool
        """
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        """
        Swaps the model's main parameters with the EMA parameters.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param saving_ema_model: If True, indicates that the EMA model is being saved. Defaults to False.
        :type saving_ema_model: bool
        """
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        A context manager to save an EMA copy of the model and EMA optimizer states.

        Temporarily swaps to EMA weights, yields, and then swaps back.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        """
        A context manager to temporarily set the `save_original_optimizer_state` flag in EMAOptimizers.

        This is used to ensure that the original optimizer state is saved instead of the EMA optimizer state.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        """
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """
        Called when a checkpoint is loaded.

        Handles loading of EMA weights and optimizer states. If an EMA checkpoint
        (e.g., `model-EMA.ckpt`) is loaded, it treats the EMA weights as the main
        weights. If a regular checkpoint is loaded, it looks for an associated
        EMA checkpoint and restores the EMA state from it.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        :param checkpoint: The loaded checkpoint dictionary.
        :type checkpoint: Dict[str, Any]
        :raises MisconfigurationException: If a regular checkpoint is loaded but its associated EMA checkpoint is not found.
        """
        checkpoint_callback = trainer.checkpoint_callback

        # use the connector as NeMo calls the connector directly in the exp_manager when restoring.
        connector = trainer._checkpoint_connector
        # Replace connector._ckpt_path with below to avoid calling into lightning's protected API
        ckpt_path = trainer.ckpt_path

        if ckpt_path and checkpoint_callback is not None and 'NeMo' in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f'-EMA{ext}'):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f'-EMA{ext}')
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device('cpu'))

                checkpoint['optimizer_states'] = ema_state_dict['optimizer_states']
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    """
    Performs the EMA update step.

    Updates the EMA parameters using the formula:
    `ema_weight = decay * ema_weight + (1 - decay) * current_weight`

    This function uses `torch._foreach_mul_` and `torch._foreach_add_` for efficient
    element-wise operations on tuples of tensors.

    :param ema_model_tuple: A tuple of EMA parameter tensors.
    :type ema_model_tuple: tuple[torch.Tensor]
    :param current_model_tuple: A tuple of current model parameter tensors.
    :type current_model_tuple: tuple[torch.Tensor]
    :param decay: The EMA decay factor.
    :type decay: float
    """
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    """
    Runs the EMA update on the CPU.

    This function is typically used when EMA parameters are offloaded to the CPU.
    It synchronizes with a CUDA stream if provided, then calls `ema_update`.

    :param ema_model_tuple: A tuple of EMA parameter tensors.
    :type ema_model_tuple: tuple[torch.Tensor]
    :param current_model_tuple: A tuple of current model parameter tensors.
    :type current_model_tuple: tuple[torch.Tensor]
    :param decay: The EMA decay factor.
    :type decay: float
    :param pre_sync_stream: A CUDA stream to synchronize with before the update. Defaults to None.
    :type pre_sync_stream: torch.cuda.Stream | None
    """
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    Wraps a PyTorch optimizer to compute Exponential Moving Average (EMA) of model parameters.

    EMA parameters are updated after each optimizer step using the formula:
    `ema_weight = decay * ema_weight + (1 - decay) * training_weight`

    Use the `swap_ema_weights()` context manager to temporarily swap the model's
    regular parameters with the EMA parameters, typically for evaluation.

    .. note::
        EMAOptimizer is not compatible with APEX AMP O2.

    :param optimizer: The PyTorch optimizer to wrap.
    :type optimizer: torch.optim.Optimizer
    :param device: The device to store EMA parameters on (e.g., 'cuda', 'cpu').
    :type device: torch.device
    :param decay: The EMA decay factor. Defaults to 0.9999.
    :type decay: float
    :param every_n_steps: Apply EMA update every N optimizer steps. Defaults to 1.
    :type every_n_steps: int
    :param current_step: The initial training step. Defaults to 0.
    :type current_step: int
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        """
        Returns an iterator over all parameters managed by the optimizer.

        :returns: An iterator over all parameters.
        :rtype: Iterable[torch.Tensor]
        """
        return (param for group in self.param_groups for param in group['params'])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        """
        Performs a single optimization step.

        This method calls the underlying optimizer's step() method and then,
        if applicable, updates the EMA parameters.

        :param closure: A closure that re-evaluates the model and returns the loss. Optional for most optimizers.
        :type closure: callable, optional
        :param grad_scaler: A `torch.cuda.amp.GradScaler` instance for mixed-precision training. Defaults to None.
        :type grad_scaler: torch.cuda.amp.GradScaler, optional
        :returns: The loss computed by the closure, or None if no closure is provided.
        """
        self.join() # Wait for previous EMA update to finish

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        if getattr(self.optimizer, "_step_supports_amp_scaling", False) and grad_scaler is not None:
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        """
        Determines if the EMA parameters should be updated at the current step.

        :returns: True if EMA should be updated, False otherwise.
        :rtype: bool
        """
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        """
        Performs the EMA update for the parameters.

        This method detaches the current model parameters, moves them to the EMA
        device, and then calls `ema_update` (or `run_ema_update_cpu` if offloading
        to CPU) to update the EMA parameters.
        """
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )

            if self.device.type == 'cuda':
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu, args=(self.ema_params, current_model_state, self.decay, self.stream,),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        """
        Swaps the data of two tensors in-place.

        :param tensor1: The first tensor.
        :type tensor1: torch.Tensor
        :param tensor2: The second tensor.
        :type tensor2: torch.Tensor
        """
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        """
        Swaps the main model parameters with the EMA parameters.

        This method is called by the EMA callback or the `swap_ema_weights`
        context manager.

        :param saving_ema_model: If True, indicates that the EMA model is being saved.
                                 This affects how `state_dict` behaves. Defaults to False.
        :type saving_ema_model: bool
        """
        self.join() # Ensure any ongoing EMA update is complete
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular model parameters with EMA parameters.

        Swaps back to the original regular parameters upon exiting the context.

        :param enabled: If False, the swap is not performed. Defaults to True.
        :type enabled: bool
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        """
        Delegates attribute access to the underlying optimizer if the attribute
        is not found in this EMAOptimizer instance.

        :param name: The name of the attribute.
        :type name: str
        :returns: The attribute from the underlying optimizer.
        :raises AttributeError: If the attribute is not found in either EMAOptimizer or the underlying optimizer.
        """
        return getattr(self.optimizer, name)

    def join(self):
        """
        Waits for any asynchronous EMA update (CUDA stream or CPU thread) to complete.
        """
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        """
        Returns the state of the EMAOptimizer.

        Includes the state of the underlying optimizer, the EMA parameters,
        the current step, decay, and `every_n_steps`. If `save_original_optimizer_state`
        is True, only the original optimizer's state is returned.

        :returns: A dictionary containing the EMAOptimizer state.
        :rtype: dict
        """
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(self.all_parameters())
        state_dict = {
            'opt': self.optimizer.state_dict(),
            'ema': ema_params,
            'current_step': self.current_step,
            'decay': self.decay,
            'every_n_steps': self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the EMAOptimizer state.

        Restores the state of the underlying optimizer, EMA parameters, current_step,
        decay, and `every_n_steps`.

        :param state_dict: The EMAOptimizer state dictionary to load.
        :type state_dict: dict
        """
        self.join()

        self.optimizer.load_state_dict(state_dict['opt'])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict['ema']))
        self.current_step = state_dict['current_step']
        self.decay = state_dict['decay']
        self.every_n_steps = state_dict['every_n_steps']
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        """
        Adds a parameter group to the underlying optimizer.

        Also flags that EMA parameters need to be rebuilt to include parameters
        from the new group.

        :param param_group: The parameter group to add.
        :type param_group: dict
        """
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True

if __name__ == '__main__':
    pass