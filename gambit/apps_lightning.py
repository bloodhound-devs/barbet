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
import os

from gambit.models import GambitModel

def gene_id_from_accession(accession:str):
    return accession.split("/")[-1]


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

        self.train_dataset = GambitDataset(self.training, self.seqtree, self.seqbank, self.gene_id_dict)
        self.val_dataset = GambitDataset(self.validation, self.seqtree, self.seqbank, self.gene_id_dict)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_function, learning_rate:float, input_count:int=1):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.input_count = input_count

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
    
    def trainer(self) -> L.Trainer:
        return L.Trainer()
    
    def lightning_module(self) -> L.LightningModule:
        # TODO CLI
        learning_rate = 1e-4
        #############

        return GeneralLightningModule(
            model=self.model(),
            loss_function=self.loss_function(),
            learning_rate=learning_rate,
            input_count=2,
        )
    
    def data(self) -> Iterable|L.LightningDataModule:
        return GambitDataModule(
            seqbank=self.seqbank,
            seqtree=self.seqtree,
            gene_id_dict=self.gene_id_dict,
        )
    
    def validation_dataloader(self) -> Iterable|None:
        return None
        
    def fit(self):
        lightning_module = self.lightning_module()
        trainer = self.trainer()
        validation_dataloader = self.validation_dataloader()

        trainer.fit( lightning_module, self.data(), validation_dataloader )


def main():
    app = GambitLightningApp()
    app.setup()
    app.fit()


if __name__ == "__main__":
    main()