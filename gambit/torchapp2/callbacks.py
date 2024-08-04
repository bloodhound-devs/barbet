import lightning as L
import time


class LogOptimizerCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            pl_module.log('lr_0', param_group['lr'])
            pl_module.log('mom_0', param_group['betas'][0]) # HACK
            pl_module.log('beta_1', param_group['betas'][1])
            pl_module.log('wd_0', param_group['weight_decay'])
            pl_module.log('eps_0', param_group['eps'])


class TimeLoggingCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        pl_module.log('epoch_time', epoch_duration, prog_bar=True)
