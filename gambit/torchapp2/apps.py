import os
from collections.abc import Iterable
import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import Metric

from .modules import GeneralLightningModule
from .callbacks import TimeLoggingCallback, LogOptimizerCallback
from .cli import CLIApp

class TorchApp2(CLIApp):
    def setup(self) -> None:
        pass

    def model(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}.")
    
    def loss_function(self):
        raise NotImplementedError(f"Please ensure that the 'loss_function' method is implemented in {self.__class__.__name__}.")
    
    def data(self) -> Iterable|L.LightningDataModule:
        raise NotImplementedError(f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}.")
    
    def validation_dataloader(self) -> Iterable|None:
        return None

    def callbacks(self):
        return [
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]
    
    def trainer(
        self,
        max_epochs:int=20,
        run_name:str="",
        wandb:bool=False,
        wandb_project:str="",
        wandb_entity:str="",
    ) -> L.Trainer:
        loggers = [
            CSVLogger("logs", name=run_name)
        ]
        if wandb:
            if wandb_project:
                os.environ["WANDB_PROJECT"] = wandb_project
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity

            wandb_logger = WandbLogger(name=run_name)
            loggers.append(wandb_logger)
        
        # If GPUs are available, use all of them; otherwise, use CPUs
        gpus = torch.cuda.device_count()
        if gpus > 1:
            devices = gpus
            strategy = 'ddp'  # Distributed Data Parallel
        else:
            devices = "auto"  # Will use CPU if no GPU is available
            strategy = "auto"

        return L.Trainer(accelerator="gpu", devices=devices, strategy=strategy, logger=loggers, max_epochs=max_epochs, callbacks=self.callbacks())
    
    def metrics(self) -> list[tuple[str,Metric]]:
        return []
    
    def lightning_module(
        self,
        max_learning_rate:float = 1e-4,
    ) -> L.LightningModule:
        return GeneralLightningModule(
            model=self.model(),
            loss_function=self.loss_function(),
            max_learning_rate=max_learning_rate,
            input_count=2,
            metrics=self.metrics(),
        )
            
    def fit(self):
        data = self.data()
        data.setup()

        lightning_module = self.lightning_module()
        trainer = self.trainer()
        validation_dataloader = self.validation_dataloader()

        # Dummy data to set the number of weights in the model
        dummy_batch = next(iter(data.train_dataloader()))
        dummy_x = dummy_batch[:lightning_module.input_count]
        with torch.no_grad():
            lightning_module.model(*dummy_x)

        trainer.fit( lightning_module, data, validation_dataloader )


