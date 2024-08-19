from pathlib import Path
import os
from collections.abc import Iterable
import torch
from torch import nn
from typing import Type
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import Metric
from pytorch_lightning.profilers import Profiler, PyTorchProfiler

from .modules import GeneralLightningModule
from .callbacks import TimeLoggingCallback, LogOptimizerCallback
from .cli import CLIApp, method, main, tool

class TorchApp2(CLIApp):
    @method
    def setup(self) -> None:
        pass

    @method
    def model(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}.")
    
    @method
    def loss_function(self):
        raise NotImplementedError(f"Please ensure that the 'loss_function' method is implemented in {self.__class__.__name__}.")
    
    @method()
    def data(self) -> Iterable|L.LightningDataModule:
        raise NotImplementedError(f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}.")
    
    @method
    def validation_dataloader(self) -> Iterable|None:
        return None

    @method
    def prediction_dataloader(self, module) -> Iterable:
        raise NotImplementedError(f"Please ensure that the 'data' method is implemented in {self.__class__.__name__}.")

    @method
    def callbacks(self):
        return [
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]
    
    @method
    def profiler(
        self,
        profiler_path:Path,
        **kwargs,
    ) -> Profiler:
        return PyTorchProfiler(
            filename=str(profiler_path)
        )
    
    @method
    def trainer(
        self,
        max_epochs:int=20,
        run_name:str="",
        wandb:bool=False,
        wandb_project:str="",
        wandb_entity:str="",
        max_gpus:int=0,
        **kwargs,
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
        if max_gpus:
            gpus = min(max_gpus, gpus)
        
        if gpus > 1:
            devices = gpus
            strategy = 'ddp'  # Distributed Data Parallel
        else:
            devices = "auto"  # Will use CPU if no GPU is available
            strategy = "auto"

        return L.Trainer(
            accelerator="gpu", 
            devices=devices, 
            strategy=strategy, 
            logger=loggers, 
            max_epochs=max_epochs, 
            callbacks=self.callbacks(),
            profiler=self.profiler(**kwargs),
        )
    
    @method
    def metrics(self) -> list[tuple[str,Metric]]:
        return []
    
    @method
    def input_count(self) -> int:
        return 1
    
    @method
    def module_class(self) -> Type[GeneralLightningModule]:
        return GeneralLightningModule
        
    @method("model", "loss_function")
    def lightning_module(
        self,
        max_learning_rate:float = 1e-4,
        **kwargs,
    ) -> L.LightningModule:
        model = self.model(**kwargs)
        loss_function = self.loss_function(**kwargs)
        metrics = self.metrics(**kwargs)
        input_count = self.input_count(**kwargs)

        module_class = self.module_class(**kwargs)

        return module_class(
            model=model,
            loss_function=loss_function,
            max_learning_rate=max_learning_rate,
            input_count=input_count,
            metrics=metrics,
        )
    
    @tool("setup", "data", "lightning_module", "trainer")
    def train(
        self,
        **kwargs,
    ):
        """Train the model."""
        self.setup(**kwargs)
        data = self.data(**kwargs)
        data.setup()
        validation_dataloader = self.validation_dataloader(**kwargs)

        lightning_module = self.lightning_module(**kwargs)
        trainer = self.trainer(**kwargs)

        # Dummy data to set the number of weights in the model
        dummy_batch = next(iter(data.train_dataloader()))
        dummy_x = dummy_batch[:lightning_module.input_count]
        with torch.no_grad():
            lightning_module.model(*dummy_x)

        trainer.fit( lightning_module, data, validation_dataloader )

    @method
    def checkpoint(self, checkpoint:Path, **kwargs) -> Path:
        """ Returns a path to a checkpoint to use for prediction. """
        return checkpoint
    
    @method
    def load_checkpoint(self, **kwargs) -> L.LightningModule:
        module_class = self.module_class(**kwargs)
        return module_class.load_from_checkpoint(self.checkpoint(**kwargs))
    
    @method
    def prediction_trainer(self, module) -> L.Trainer:
        # TODO multigpu
        return L.Trainer()
    
    @method
    def output_results(self, results, **kwargs):
        raise NotImplementedError(f"Please ensure that the 'output_results' method is implemented in {self.__class__.__name__}.")

    @main("load_checkpoint", "prediction_trainer", "prediction_dataloader", "output_results")
    def predict(self, **kwargs):
        """ Make predictions with the model. """
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader(module, **kwargs)

        results = trainer.predict(module, dataloaders=prediction_dataloader)

        return self.output_results(results, **kwargs)
        


