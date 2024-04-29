from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from corgi.seqtree import SeqTree
from rich.console import Console

from hierarchicalsoftmax import HierarchicalSoftmaxLoss
from hierarchicalsoftmax.metrics import GreedyAccuracy

from .models import GambitModel
from .dataloaders import create_dataloaders

console = Console()

class Gambit(ta.TorchApp):
    """
    Geometric Analysis of MicroBIal Taxonomies
    """
    def dataloaders(
        self,
        batch_size:int = ta.Param(default=32, help="The batch size."),
        seqtree:Path = None,
        validation_partition:int=0,
        base_dir:Path=None,
        max_items:int=None,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Gambit uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        assert base_dir is not None
        base_dir = Path(base_dir)
        assert seqtree is not None

        print(f"Loading seqtree {seqtree}")
        seqtree = SeqTree.load(seqtree)

        self.classification_tree = seqtree.classification_tree
        assert self.classification_tree is not None

        dataloaders = create_dataloaders(
            base_dir=base_dir,
            seqtree=seqtree,
            batch_size=batch_size,
            validation_partition=validation_partition,
            max_items=max_items,
        )
        return dataloaders

    def model(
        self,
        features:int=5120,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Gambit to use.

        Returns:
            nn.Module: The created model.
        """
        return GambitModel(
            classification_tree=self.classification_tree,
            features=features,
        )

    def loss_func(self):
        return HierarchicalSoftmaxLoss(root=self.classification_tree)
    
    def metrics(self):
        return [
            GreedyAccuracy(root=self.classification_tree, max_depth=1, name="phylum"),
            GreedyAccuracy(root=self.classification_tree, max_depth=2, name="class"),
            GreedyAccuracy(root=self.classification_tree, max_depth=3, name="order"),
            GreedyAccuracy(root=self.classification_tree, max_depth=4, name="family"),
            GreedyAccuracy(root=self.classification_tree, max_depth=5, name="genus"),
            GreedyAccuracy(root=self.classification_tree, max_depth=6, name="species"),
        ]
