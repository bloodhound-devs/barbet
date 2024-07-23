import torch
from torch import nn
from corgi.seqtree import SeqTree
from seqbank import SeqBank
import lightning as L
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from torch.utils.data import DataLoader
from collections.abc import Iterable

from .models import GambitModel

class GambitDataModule(L.LightningDataModule):
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        training = []
        validation = []
        family_ids = set()

        for accession, details in seqtree.items():
            partition = details.partition
            dataset = validation if partition == validation_partition else training
            dataset.append( accession )

            gene_id = accession.split("/")[-1]
            family_ids.add(gene_id)

            if max_items and len(training) >= max_items and len(validation) > 0:
                break

        gene_id_dict = {family_id:index for index, family_id in enumerate(sorted(family_ids))}

    def train_dataloader(self):
        return DataLoader(self.train)

    def val_dataloader(self):
        return DataLoader(self.val)

    # def test_dataloader(self):
    #     return DataLoader(self.test)

    # def on_exception(self, exception):
    #     # clean up state after the trainer faced an exception
    #     ...

    # def teardown(self):
    #     # clean up state after the trainer stops, delete files...
    #     # called on every process in DDP
    #     ...




class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_function, learning_rate):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class GambitLightningApp():
    def setup(self) -> None:
        # TODO CLI
        seqbank = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d.sb"
        seqtree = "/data/gpfs/projects/punim2199/preprocessed/ar53/prostt5/prostt5-d.st"
        #############

        assert seqbank is not None
        print(f"Loading seqbank {seqbank}")
        seqbank = SeqBank(seqbank)

        print(f"Loading seqtree {seqtree}")
        seqtree = SeqTree.load(seqtree)

        self.classification_tree = seqtree.classification_tree
        assert self.classification_tree is not None

        self.gene_id_dict = ...

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

        )
    
    def data(self) -> Iterable|L.LightningDataModule:
        return GambitDataModule()
    
    def validation_dataloader(self) -> Iterable|None:
        return None
        
    def fit(self):
        lightning_module = self.build_lightning_module()
        trainer = self.build_trainer()
        validation_dataloader = self.validation_dataloader()

        trainer.fit( lightning_module, self.data(), validation_dataloader )

