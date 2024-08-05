import torch
from functools import cached_property
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import Metric
import lightning as L


from .metrics import AvgSmoothLoss

class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_function, max_learning_rate:float, input_count:int=1, metrics:list[tuple[str,Metric]]|None=None):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.max_learning_rate = max_learning_rate
        self.input_count = input_count
        self.smooth_loss = AvgSmoothLoss()
        self.metricks = metrics or []
        for name, metric in self.metricks:
            setattr(self, name, metric)        
        self.current_step = 0

    @cached_property
    def steps_per_epoch(self) -> int:
        # HACK assumes DDP strategy
        gpus = torch.cuda.device_count()
        return len(self.trainer.datamodule.train_dataloader())//gpus

    def training_step(self, batch, batch_idx):
        x = batch[:self.input_count]
        y = batch[self.input_count:]
        y_hat = self.model(*x)
        loss = self.loss_function(y_hat, *y)
        self.log("raw_loss", loss, on_step=True, on_epoch=False)

        self.smooth_loss.update(loss)
        self.log("train_loss", self.smooth_loss.compute(), on_step=True, on_epoch=False)

        # Log the fractional epoch
        self.current_step += 1
        fractional_epoch = self.current_epoch + ((self.current_step%self.steps_per_epoch) / self.steps_per_epoch)
        self.log('fractional_epoch', fractional_epoch, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:self.input_count]
        y = batch[self.input_count:]
        y_hat = self.model(*x)
        loss = self.loss_function(y_hat, *y)
        self.log("valid_loss", loss, sync_dist=True)
        # Metrics
        for name, metric in self.metricks:
            result = metric(y_hat, *y)
            if isinstance(result, dict):
                for key, value in result.items():
                    self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True)

    def on_epoch_end(self):
        self.current_step = 0
    
    def optimizer(self) -> optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.1*self.max_learning_rate, weight_decay=0.01, eps=1e-5)

    def scheduler(self, optimizer) -> lr_scheduler._LRScheduler:
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_learning_rate,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.trainer.max_epochs,
        )

    def lr_scheduler_config(self, optimizer:optim.Optimizer) -> dict:
        return {
            'scheduler': self.scheduler(optimizer),
            'interval': 'step',
        }

    def configure_optimizers(self) -> dict:
        # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        optimizer = self.optimizer()
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.lr_scheduler_config(optimizer=optimizer),
        }


