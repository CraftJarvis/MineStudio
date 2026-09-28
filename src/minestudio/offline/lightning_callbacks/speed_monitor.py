'''
Date: 2024-11-28 15:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 15:37:52
FilePath: /MineStudio/minestudio/train/lightning_callbacks/speed_monitor.py
'''
import time
import lightning.pytorch as pl

class SpeedMonitorCallback(pl.Callback):
    """
    A PyTorch Lightning callback to monitor training speed.

    This callback logs the training speed in batches per second at regular intervals.
    It only logs on the global rank 0 process to avoid redundant logging in
    distributed training setups.
    """
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of each training batch.

        Calculates and logs the training speed every `INTERVAL` batches on rank 0.

        :param trainer: The PyTorch Lightning Trainer instance.
        :type trainer: pl.Trainer
        :param pl_module: The PyTorch Lightning LightningModule instance.
        :type pl_module: pl.LightningModule
        :param outputs: The outputs of the training step.
        :type outputs: Any
        :param batch: The current batch of data.
        :type batch: Any
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        """
        INTERVAL = 16
        if trainer.global_rank != 0 or batch_idx % INTERVAL != 0:
            return 
        now = time.time()
        
        if hasattr(self, 'time_start'):
            time_cost = now - self.time_start
            trainer.logger.log_metrics({'train/speed(batch/s)': INTERVAL/time_cost}, step=trainer.global_step)
            self.time_start = now
        else:
            self.time_start = now
