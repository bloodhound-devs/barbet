from pathlib import Path
import torch
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from corgi.seqtree import SeqTree, node_to_str
from rich.console import Console

from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax.metrics import GreedyAccuracy
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities
import pandas as pd


from .models import GambitModel, GambitModel2
from .dataloaders import create_dataloaders, species_dataloader

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
        intermediate_layers:int=0, 
        growth_factor:float=2.0,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Gambit to use.

        Returns:
            nn.Module: The created model.
        """
        return GambitModel2(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,

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

    def inference_dataloader(
        self,
        learner,
        embeddings:Path = ta.Param(None, help="A embeddings for a species saved in PyTorch format."),
        batch_size:int = 64,
        **kwargs,
    ):
        self.dataloader = species_dataloader(embeddings=embeddings, batch_size=batch_size)
        self.classification_tree = learner.dls.classification_tree
        return self.dataloader

    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        output_tips_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        output_fasta: Path = ta.Param(default=None, help="A path to output the results in FASTA format."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_format:str = "svg",
        image_threshold:float = 0.005,
        prediction_threshold:float = ta.Param(default=0.5, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):
        
        assert self.classification_tree # This should be saved from the learner
        
        classification_probabilities = node_probabilities(results[0], root=self.classification_tree)
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list if not node.is_root]

        # Average over genes
        results_df = pd.DataFrame(classification_probabilities.mean(axis=0, keepdims=True).numpy(), columns=category_names)
        
        # Get new tensors now that we've averaged over chunks
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=prediction_threshold,
        )

        results_df['greedy_prediction'] = [
            self.node_to_str(node)
            for node in predictions
        ]

        def get_prediction_probability(row):
            prediction = row["greedy_prediction"]
            if prediction in row:
                return row[prediction]
            return 1.0
        
        results_df['probability'] = results_df.apply(get_prediction_probability, axis=1)

        # Reorder columns
        results_df = results_df[["greedy_prediction", "probability" ] + category_names]

        # Output images
        if image_dir:
            console.print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = []
            for _, row in results_df.iterrows():
                image_path = image_dir / "predictions.png"
                image_paths.append(image_path)

            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        if not (image_dir or output_fasta or output_csv or output_tips_csv):
            print("No output files requested.")

        if output_tips_csv:
            output_tips_csv = Path(output_tips_csv)
            output_tips_csv.parent.mkdir(exist_ok=True, parents=True)
            non_tips = [self.node_to_str(node) for node in self.classification_tree.node_list if not node.is_leaf]
            tips_df = results_df.drop(columns=non_tips)
            tips_df.to_csv(output_tips_csv, index=False)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(exist_ok=True, parents=True)
            console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
            results_df.to_csv(output_csv, index=False)

        return results_df

    def node_to_str(self, node:SoftmaxNode) -> str:
        """ 
        Converts the node to a string
        """
        return str(node)
