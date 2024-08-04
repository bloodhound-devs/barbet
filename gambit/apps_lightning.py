import psutil
import torch
import numpy as np
from pathlib import Path
from torch import nn
import lightning as L
from corgi.seqtree import SeqTree
from seqbank import SeqBank
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from torch.utils.data import DataLoader
from collections.abc import Iterable
from torch.utils.data import Dataset
from dataclasses import dataclass
from hierarchicalsoftmax.metrics import greedy_accuracy
from torchmetrics import Metric
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from functools import cached_property
import time

from lightning.pytorch.loggers import CSVLogger

from .modelsx import GambitModel

def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]

RANKS = ["phylum", "class", "order", "family", "genus", "species"]

esm_layers = 6
DOMAIN = "bac120"
# DOMAIN = "ar53"


class AvgSmoothLoss(Metric):
    def __init__(self, beta=0.98, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.beta = beta
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("val", default=torch.tensor(0.), dist_reduce_fx="sum")

    def reset(self):
        self.count = torch.tensor(0)
        self.val = torch.tensor(0.)

    def update(self, loss):
        # Ensure loss is detached to avoid graph-related issues
        loss = loss.detach()
        self.count += 1
        self.val = torch.lerp(loss.mean(), self.val, self.beta)

    def compute(self):
        # Return the smoothed loss value
        return self.val / (1 - self.beta**self.count)


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


# class GreedyAccuracyTorchMetric(Metric):
#     def __init__(self, root:SoftmaxNode, name:str="", max_depth=None):
#         super().__init__()
#         self.root = root
#         self.max_depth = max_depth
#         self.name = name or (f"greedy_accuracy_{max_depth}" if max_depth else "greedy_accuracy")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, predictions, targets):
#         self.total += targets.size(0)
#         self.correct += int(greedy_accuracy(predictions, targets, self.root, max_depth=self.max_depth) * targets.size(0))

#     def compute(self):
#         return self.correct / self.total


@dataclass
class GambitDataset(Dataset):
    accessions: list[str]
    seqtree: SeqTree
    # seqbank: SeqBank
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]
    gene_id_dict: dict[str, int]

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_index = self.accession_to_array_index[accession]
        embedding = torch.tensor(self.array[array_index,:], dtype=torch.float16)
        gene_id = gene_id_from_accession(accession)
        return embedding, self.gene_id_dict[gene_id], self.seqtree[accession].node_id


@dataclass
class GambitDataModule(L.LightningDataModule):
    seqtree: SeqTree
    # seqbank: SeqBank
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]
    gene_id_dict: dict[str,int]
    max_items: int = 0
    batch_size: int = 16
    num_workers: int = 0
    validation_partition:int = 0

    def __init__(
        self,
        seqtree: SeqTree,
        array:np.memmap|np.ndarray,
        accession_to_array_index:dict[str,int],
        # seqbank: SeqBank,
        gene_id_dict: dict[str,int],
        max_items: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        validation_partition:int = 0           ,
    ):
        super().__init__()
        self.array = array
        self.accession_to_array_index = accession_to_array_index
        self.seqtree = seqtree
        self.gene_id_dict = gene_id_dict
        self.max_items = max_items
        self.batch_size = batch_size
        self.validation_partition = validation_partition
        self.num_workers = num_workers or os.cpu_count()

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.training = []
        self.validation = []

        for accession, details in self.seqtree.items():
            partition = details.partition
            dataset = self.validation if partition == self.validation_partition else self.training
            dataset.append( accession )

            if self.max_items and len(self.training) >= self.max_items and len(self.validation) > 0:
                break

        self.train_dataset = self.create_dataset(self.training)
        self.val_dataset = self.create_dataset(self.validation)

    def create_dataset(self, accessions:list[str]) -> GambitDataset:
        return GambitDataset(
            accessions=accessions, 
            seqtree=self.seqtree, 
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            gene_id_dict=self.gene_id_dict,
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


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


class GambitLightningApp():
    def setup(self) -> None:
        # TODO CLI
        # seqbank = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d-standardized.sb"
        # seqtree = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d.st"

        # esm_layers = 30
        
        # seqbank = f"/data/projects/punim2199/rob/release220/{DOMAIN}/esm{esm_layers}/esm{esm_layers}.sb"
        memmap = f"/data/projects/punim2199/rob/release220/{DOMAIN}/esm{esm_layers}/esm{esm_layers}.np"
        memmap_index = f"/data/projects/punim2199/rob/release220/{DOMAIN}/esm{esm_layers}/esm{esm_layers}.txt"
        seqtree = f"/data/projects/punim2199/rob/release220/{DOMAIN}/esm{esm_layers}/esm{esm_layers}.st"
        #############

        # assert seqbank is not None
        # print(f"Loading seqbank {seqbank}")
        # self.seqbank = SeqBank(seqbank)
        print(f"Loading memmap")
        dtype = "float16"
        self.accession_to_array_index = dict()
        with open(memmap_index) as f:
            for i, accession in enumerate(f):
                self.accession_to_array_index[accession.strip()] = i
        count = len(self.accession_to_array_index)
        file_size = os.path.getsize(memmap)
        dtype_size = np.dtype(dtype).itemsize
        num_elements = file_size // dtype_size
        embedding_size = num_elements // count
        shape = (count, embedding_size)
        self.array = np.memmap(memmap, dtype=dtype, mode='r', shape=shape)

        # If there's enough memory, then read into RAM
        memory_info = psutil.virtual_memory()
        if memory_info.available > file_size * 2:
            self.array = np.array(self.array)

        print(f"Loading seqtree {seqtree}")
        self.seqtree = SeqTree.load(seqtree)

        self.classification_tree = self.seqtree.classification_tree
        assert self.classification_tree is not None

        # Get list of gene families
        family_ids = set()
        for accession in self.seqtree:
            gene_id = accession.split("/")[-1]
            family_ids.add(gene_id)

        self.gene_id_dict = {family_id:index for index, family_id in enumerate(sorted(family_ids))}

    def model(self) -> nn.Module:
        # TODO CLI
        features:int=1024
        intermediate_layers:int=2
        growth_factor:float=2.0
        family_embedding_size:int=128
        #############

        return GambitModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            family_embedding_size=family_embedding_size,
            gene_family_count=len(self.gene_id_dict),
        )
    
    def loss_function(self):
        return HierarchicalSoftmaxLoss(root=self.classification_tree)
    
    def callbacks(self):
        return [
            TimeLoggingCallback(),
            LogOptimizerCallback(),
        ]
    
    def trainer(self) -> L.Trainer:
        # TODO CLI
        max_epochs=20
        wandb = False
        run_name = f"lightning-{DOMAIN}-esm{esm_layers}-b16f1024l2g2e128lr1E-4-memmap"
        wandb_project = "Gambit"
        wandb_entity = "mariadelmarq-The University of Melbourne"
        #############
        loggers = [
            CSVLogger("logs", name=run_name)
        ]
        if wandb:
            from lightning.pytorch.loggers import WandbLogger
            if wandb_project:
                os.environ["WANDB_PROJECT"] = wandb_project
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = wandb_entity

            wandb_logger = WandbLogger(name=run_name)
            loggers.append(wandb_logger)
        
        gpus = torch.cuda.device_count()

        # If GPUs are available, use them; otherwise, use CPUs
        if gpus > 1:
            devices = gpus
            strategy = 'ddp'  # Distributed Data Parallel
        else:
            devices = "auto"  # Will use CPU if no GPU is available
            strategy = "auto"

        return L.Trainer(accelerator="gpu", devices=devices, strategy=strategy, logger=loggers, max_epochs=max_epochs, callbacks=self.callbacks())
        # return L.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy="ddp", logger=logger, max_epochs=max_epochs)
    
    def metrics(self) -> list[tuple[str,Metric]]:
        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={1+i:rank for i, rank in enumerate(RANKS)},
        )
                
        return [('rank_accuracy', rank_accuracy)]
        # return [
        #     (rank, GreedyAccuracyTorchMetric(root=self.classification_tree, max_depth=i+1, name=rank))
        #     for i, rank in enumerate(RANKS)
        # ]
    
    def lightning_module(self) -> L.LightningModule:
        # TODO CLI
        max_learning_rate = 1e-4
        #############

        return GeneralLightningModule(
            model=self.model(),
            loss_function=self.loss_function(),
            max_learning_rate=max_learning_rate,
            input_count=2,
            metrics=self.metrics(),
        )
    
    def data(self) -> Iterable|L.LightningDataModule:
        # TODO CLI
        max_items = 0        
        # max_items=1000
        #############

        return GambitDataModule(
            # seqbank=self.seqbank,
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            seqtree=self.seqtree,
            gene_id_dict=self.gene_id_dict,
            max_items=max_items,
        )
    
    def validation_dataloader(self) -> Iterable|None:
        return None
        
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


def main():
    app = GambitLightningApp()
    app.setup()
    
    app.fit()


if __name__ == "__main__":
    main()