'''
Date: 2024-11-28 15:37:18
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-27 14:03:44
FilePath: /MineStudio/minestudio/offline/lightning_callbacks/smart_checkpoint.py
'''
from lightning.pytorch.callbacks import ModelCheckpoint
from minestudio.offline.lightning_callbacks.ema import EMA
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import lightning.pytorch as pl # Add this import

from typing import (
    Dict, List, Union, Sequence, Mapping, Any, Optional
)

class SmartCheckpointCallback(ModelCheckpoint):
    """
    A PyTorch Lightning ModelCheckpoint callback that is aware of EMA (Exponential Moving Average) weights.

    This callback extends the standard ModelCheckpoint to also save and remove
    EMA weights if an EMA callback is present in the trainer.
    EMA checkpoints are saved with an '-EMA' suffix before the file extension.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the SmartCheckpointCallback.

        :param kwargs: Keyword arguments to be passed to the parent ModelCheckpoint class.
        """
        super().__init__(**kwargs)
    
    def _ema_callback(self, trainer: 'pl.Trainer') -> Optional[EMA]: # Changed type hint
        """
        Finds and returns the EMA callback from the trainer's list of callbacks.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :returns: The EMA callback instance if found, otherwise None.
        :rtype: Optional[EMA]
        """
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _ema_format_filepath(self, filepath: str) -> str:
        """
        Formats the filepath for an EMA checkpoint.

        Appends '-EMA' before the file extension (e.g., 'model.ckpt' -> 'model-EMA.ckpt').

        :param filepath: The original checkpoint filepath.
        :type filepath: str
        :returns: The formatted filepath for the EMA checkpoint.
        :rtype: str
        """
        return filepath.replace(self.FILE_EXTENSION, f'-EMA{self.FILE_EXTENSION}')

    def _save_checkpoint(self, trainer: 'pl.Trainer', filepath: str) -> None: # Changed type hint
        """
        Saves a checkpoint.

        If an EMA callback is present, it saves both the original model checkpoint
        and an EMA model checkpoint. The EMA checkpoint includes the EMA weights.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param filepath: The filepath for the checkpoint.
        :type filepath: str
        """
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # with ema_callback.save_original_optimizer_state(trainer):
            super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None: # Changed type hint
        """
        Removes a checkpoint.

        If an EMA callback is present, it removes both the original model checkpoint
        and its corresponding EMA model checkpoint.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param filepath: The filepath of the checkpoint to remove.
        :type filepath: str
        """
        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.
            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)