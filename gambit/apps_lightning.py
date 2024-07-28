import torch
from torch import nn
import lightning as L
from corgi.seqtree import SeqTree
from seqbank import SeqBank
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from torch.utils.data import DataLoader
from collections.abc import Iterable
from torch.utils.data import Dataset
from dataclasses import dataclass
from hierarchicalsoftmax.metrics import greedy_accuracy
from torchmetrics import Metric
import os
import time

from lightning.pytorch.loggers import CSVLogger

from .modelsx import GambitModel

def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]

RANKS = ["phylum", "class", "order", "family", "genus", "species"]



class TimeLoggingCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch_duration = time.time() - self.epoch_start_time
        pl_module.log('epoch_time', epoch_duration, prog_bar=True)


class GreedyAccuracyTorchMetric(Metric):
    def __init__(self, root:SoftmaxNode, name:str="", max_depth=None):
        super().__init__()
        self.root = root
        self.max_depth = max_depth
        self.name = name or (f"greedy_accuracy_{max_depth}" if max_depth else "greedy_accuracy")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        self.total += targets.size(0)
        self.correct += int(greedy_accuracy(predictions, targets, self.root, max_depth=self.max_depth) * targets.size(0))

    def compute(self):
        return self.correct / self.total


@dataclass
class GambitDataset(Dataset):
    accessions: list[str]
    seqtree: SeqTree
    seqbank: SeqBank
    gene_id_dict: dict[str, int]

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        data = self.seqbank[accession]
        array = torch.frombuffer(data, dtype=torch.float32)
        del data
        gene_id = gene_id_from_accession(accession)
        return array, self.gene_id_dict[gene_id], self.seqtree[accession].node_id


@dataclass
class GambitDataModule(L.LightningDataModule):
    seqtree: SeqTree
    seqbank: SeqBank
    gene_id_dict: dict[str,int]
    max_items: int = 0
    batch_size: int = 32
    num_workers: int = 0
    validation_partition:int = 0

    def __init__(
        self,
        seqtree: SeqTree,
        seqbank: SeqBank,
        gene_id_dict: dict[str,int],
        max_items: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        validation_partition:int = 0           ,
    ):
        super().__init__()
        self.seqbank = seqbank
        self.seqtree = seqtree
        self.gene_id_dict = gene_id_dict
        self.max_items = max_items
        self.batch_size = batch_size
        self.validation_partition = validation_partition
        self.num_workers = num_workers or os.cpu_count()

    def setup(self, stage):
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

        self.train_dataset = GambitDataset(self.training, self.seqtree, self.seqbank, self.gene_id_dict)
        self.val_dataset = GambitDataset(self.validation, self.seqtree, self.seqbank, self.gene_id_dict)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_function, learning_rate:float, input_count:int=1, metrics:list[tuple[str,Metric]]|None=None):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.input_count = input_count
        self.metrics = metrics or []
        for name, metric in self.metrics:
            setattr(self, name, metric)

    def training_step(self, batch, batch_idx):
        x = batch[:self.input_count]
        y = batch[self.input_count:]
        y_hat = self.model(*x)
        loss = self.loss_function(y_hat, *y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:self.input_count]
        y = batch[self.input_count:]
        y_hat = self.model(*x)
        loss = self.loss_function(y_hat, *y)
        self.log("val_loss", loss)
        # Metrics
        for name, metric in self.metrics:
            metric(y_hat, *y)
            self.log(name, metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class GambitLightningApp():
    def setup(self) -> None:
        # TODO CLI
        seqbank = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d-standardized.sb"
        seqtree = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d.st"
        #############

        assert seqbank is not None
        print(f"Loading seqbank {seqbank}")
        self.seqbank = SeqBank(seqbank)

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
        features:int=5120
        intermediate_layers:int=0 
        growth_factor:float=2.0
        family_embedding_size:int=64
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
        ]
    
    def trainer(self) -> L.Trainer:
        # TODO CLI
        max_epochs=20
        wandb = True
        run_name = "lightning_test"
        #############
        loggers = [
            CSVLogger("logs", name=run_name)
        ]
        if wandb:
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(name=run_name)
            loggers.append(wandb_logger)
        
        return L.Trainer(accelerator="gpu", logger=loggers, max_epochs=max_epochs, callbacks=self.callbacks())
        # return L.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy="ddp", logger=logger, max_epochs=max_epochs)
    
    def metrics(self) -> list[tuple[str,Metric]]:
        return [
            (rank, GreedyAccuracyTorchMetric(root=self.classification_tree, max_depth=i+1, name=rank))
            for i, rank in enumerate(RANKS)
        ]
    
    def lightning_module(self) -> L.LightningModule:
        # TODO CLI
        learning_rate = 1e-4
        #############

        return GeneralLightningModule(
            model=self.model(),
            loss_function=self.loss_function(),
            learning_rate=learning_rate,
            input_count=2,
            metrics=self.metrics(),
        )
    
    def data(self) -> Iterable|L.LightningDataModule:
        # TODO CLI
        max_items = 0        
        # max_items=100
        #############

        return GambitDataModule(
            seqbank=self.seqbank,
            seqtree=self.seqtree,
            gene_id_dict=self.gene_id_dict,
            max_items=max_items,
        )
    
    def validation_dataloader(self) -> Iterable|None:
        return None
        
    def fit(self):
        lightning_module = self.lightning_module()
        trainer = self.trainer()
        validation_dataloader = self.validation_dataloader()

        # Dummy data to set the number of weights in the model
        dummy_input = torch.randn(2, 512)#.cuda()
        dummy_input2 = torch.randint(low=0, high=6, size=(2,))#.cuda()
        with torch.no_grad():
            lightning_module.model(dummy_input, dummy_input2)

        trainer.fit( lightning_module, self.data(), validation_dataloader )


def main():
    app = GambitLightningApp()
    app.setup()
    
    app.fit()


if __name__ == "__main__":
    main()