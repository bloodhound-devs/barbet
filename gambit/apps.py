from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from rich.console import Console

from .models import get_esm2_model_alphabet, GambitESMModel

console = Console()

class Gambit(ta.TorchApp):
    """
    Geometric Analysis of MicroBIal Taxonomies
    """
    def dataloaders(
        self,
        inputs:Path = ta.Param(help="The input file."), 
        batch_size:int = ta.Param(default=32, help="The batch size."),
        esm_layers:int = 6,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Gambit uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        self.esm_layers = esm_layers
        self.esm, self.alphabet = get_esm2_model_alphabet(esm_layers)

        self.classification_tree = None
        assert self.classification_tree is not None

        raise NotImplemented("Dataloaders function not implemented yet.") 

    def model(
        self,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Gambit to use.

        Returns:
            nn.Module: The created model.
        """
        return GambitESMModel(
            esm=self.esm,
            esm_layers=self.esm_layers,
            alphabet=self.alphabet,
            classification_tree=self.classification_tree,
        )
