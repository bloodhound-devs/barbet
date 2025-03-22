import torch
import numpy as np
from pathlib import Path
from torch import nn
import lightning as L
from torchmetrics import Metric
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from corgi.seqtree import SeqTree
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from torch.utils.data import DataLoader
from collections.abc import Iterable
from rich.console import Console
from gtdbtk.markers import Markers
from collections import defaultdict

import pandas as pd
import os
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities

from torchapp import Param, method, tool, TorchApp
from .models import BloodhoundModel
from .gtdbtk import read_tophits, read_tigrfam, read_pfam
from .embedding import get_key
from .data import read_memmap, RANKS, BloodhoundDataModule, BloodhoundPredictionDataset
from .embeddings.esm import ESMEmbedding, ESMLayers

console = Console()



class Bloodhound(TorchApp):
    @method
    def setup(
        self,
        memmap:str=None,
        memmap_index:str=None,
        seqtree:str=None,
        in_memory:bool=False,
        tip_alpha:float=None,
    ) -> None:
        if not seqtree:
            raise ValueError("seqtree is required")
        if not memmap:
            raise ValueError("memmap is required")
        if not memmap_index:
            raise ValueError("memmap_index is required")        

        print(f"Loading seqtree {seqtree}")
        individual_seqtree = SeqTree.load(seqtree)
        self.seqtree = SeqTree(classification_tree=individual_seqtree.classification_tree)

        # Sets the loss weighting for the tips
        if tip_alpha:
            for tip in self.seqtree.classification_tree.leaves:
                tip.parent.alpha = tip_alpha

        print(f"Loading memmap")
        self.accession_to_array_index = defaultdict(list)
        with open(memmap_index) as f:
            for key_index, key in enumerate(f):
                key = key.strip()
                accession = key.strip().split("/")[0]

                if len(self.accession_to_array_index[accession]) == 0:
                    self.seqtree[accession] = individual_seqtree[key]

                self.accession_to_array_index[accession].append(key_index)
        count = key_index + 1
        self.array = read_memmap(memmap, count)

        # If there's enough memory, then read into RAM
        if in_memory:
            self.array = np.array(self.array)

        self.classification_tree = self.seqtree.classification_tree
        assert self.classification_tree is not None

        # Get list of gene families
        family_ids = set()
        for accession in self.seqtree:
            gene_id = accession.split("/")[-1]
            family_ids.add(gene_id)

        self.gene_id_dict = {family_id:index for index, family_id in enumerate(sorted(family_ids))}

    @method
    def model(
        self,
        features:int=1024,
        intermediate_layers:int=2,
        growth_factor:float=2.0,
    ) -> nn.Module:
        return BloodhoundModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
        )
    
    # @method
    # def input_count(self) -> int:
    #     return 1
            
    @method
    def loss_function(self):
        # def dummy_loss(prediction, target):
        #     return prediction[0,0] * 0.0
        #     # return prediction.mean() * 0.0
        # return dummy_loss # hack
        return HierarchicalSoftmaxLoss(root=self.classification_tree)
    
    @method    
    def metrics(self) -> list[tuple[str,Metric]]:
        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={1+i:rank for i, rank in enumerate(RANKS)},
        )
                
        return [('rank_accuracy', rank_accuracy)]
    
    @method    
    def data(
        self,
        max_items:int=0,
        num_workers:int=0,
        validation_partition:int=0,
        batch_size:int = 1,
        test_partition:int=-1,
    ) -> Iterable|L.LightningDataModule:
        return BloodhoundDataModule(
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            seqtree=self.seqtree,
            gene_id_dict=self.gene_id_dict,
            max_items=max_items,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_partition=validation_partition,
            test_partition=test_partition,
        )
    
    @method
    def extra_hyperparameters(self, embedding_model:str="") -> dict:
        """ Extra hyperparameters to save with the module. """
        assert embedding_model, f"Please provide an embedding model."
        embedding_model = embedding_model.lower()
        if embedding_model.startswith("esm"):
            layers = embedding_model[3:].strip()
            embedding_model = ESMEmbedding()
            embedding_model.setup(layers=layers)
        else:
            raise ValueError(f"Cannot understand embedding model: {embedding_model}")

        return dict(
            embedding_model=embedding_model,
            classification_tree=self.seqtree.classification_tree,
            gene_id_dict=self.gene_id_dict,
        )
    
    @method
    def prediction_dataloader(
        self,
        module,
        sequence:Path=Param(help="A path to a directory of fasta files or a single fasta file."),
        out_dir:Path=Param(help="A path to the output directory."),
        extension='fa',
        prefix="gtdbtk",
        cpus:int=1,
        batch_size:int = 64,
        num_workers: int = 0,
    ) -> Iterable:
        # Get hyperparameters from checkpoint
        if 'embedding_model' not in module.hparams.keys():
            embedding_model = ESMEmbedding() # HACK
            embedding_model.setup(layers=30, hub_dir="/data/gpfs/projects/punim2199/torch-hub")
        else:
            embedding_model = module.hparams.embedding_model
            embedding_model.layers = ESMLayers.from_value(embedding_model.layers)
        self.classification_tree = module.hparams.classification_tree
        self.gene_id_dict = module.hparams.gene_id_dict
        domain = "ar53" if len(self.gene_id_dict) == 53 else "bac120"

        genomes = dict()
        sequence = Path(sequence)
        if sequence.is_dir():
            for path in sequence.rglob(f"*.{extension}"):
                genomes[path.stem] = str(path)
        else:
            genomes[sequence.stem] = str(sequence)

        self.name = sequence.name
    
        os.environ["GTDBTK_DATA_PATH"] = "/data/gpfs/projects/punim2199/gambit_data/release214/minimal/"

        markers = Markers(cpus)
        markers.identify(
            genomes,
            tln_tables=dict(),
            out_dir=out_dir,
            prefix=prefix,
            force=False,
            genes=False,
            write_single_copy_genes=True,
        )
    
        single_copy_fasta = out_dir / "identify/intermediate_results/single_copy_fasta"/domain
    
        embeddings = []
        self.gene_family_names = []
        for fasta in single_copy_fasta.rglob("*.fa"):
            if fasta.stem not in self.gene_id_dict:
                continue
            
            # read the fasta file sequence remove the header
            seq = fasta.read_text().split("\n")[1]
            vector =  embedding_model(seq)
            if vector is not None and not torch.isnan(vector).any():
                vector = vector.cpu().detach().clone().numpy()
                vector = torch.as_tensor(vector)
                embeddings.append(vector)
                self.gene_family_names.append(fasta.stem)
            del vector        

        gene_family_ids = [self.gene_id_dict[gene_family_name] for gene_family_name in self.gene_family_names]

        # TODO Save the embeddings


        dataset = BloodhoundPredictionDataset(embeddings=embeddings, gene_family_ids=gene_family_ids)
        dataloader =  DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return dataloader

    def node_to_str(self, node:SoftmaxNode) -> str:
        """ 
        Converts the node to a string
        """
        return str(node).split(",")[-1].strip()
    
    def output_results_to_df(
        self,
        names,
        results,
        output_csv: Path,
        output_tips_csv: Path,
        image_dir: Path,
        image_threshold:float,
        prediction_threshold:float,
    ) -> pd.DataFrame:
        classification_probabilities = node_probabilities(results, root=self.classification_tree)
        
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root]

        results_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)
        
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

        results_df["name"] = names

        # Reorder columns
        results_df = results_df[["name", "greedy_prediction", "probability" ] + category_names]

        results_df.set_index('name')

        if not (image_dir or output_csv or output_tips_csv):
            print("No output files requested.")

        # Output images
        if image_dir:
            console.print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = [image_dir/f"{name}.png" for name in results_df["name"]]
            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

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
            results_df.transpose().to_csv(output_csv)

        return results_df

    @method
    def output_results(
        self, 
        gene_results, 
        output_csv: Path = Param(default=None, help="A path to output the results as a CSV."),
        output_tips_csv: Path = Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        output_averaged_csv: Path = Param(default=None, help="A path to output the results as a CSV."),
        output_averaged_tips_csv: Path = Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        output_gene_csv: Path = Param(default=None, help="A path to output the results for individual genes as a CSV."),
        output_gene_tips_csv: Path = Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        image_dir: Path = Param(default=None, help="A path to output the results as images."),
        image_threshold:float = 0.005,
        prediction_threshold:float = Param(default=0.0, help="The threshold value for making hierarchical predictions."),
        gene_images:bool=False,
        **kwargs,
    ):
        assert self.gene_id_dict
        assert self.classification_tree

        if output_gene_csv or output_gene_tips_csv or gene_images:
            self.output_results_to_df(
                self.gene_family_names,
                gene_results,
                output_gene_csv,
                output_gene_tips_csv,
                image_dir if gene_images else None,
                image_threshold=image_threshold,
                prediction_threshold=prediction_threshold,
            )

        result = self.output_results_to_df(
            [self.name+"-summed"],
            gene_results.sum(axis=0, keepdims=True),
            output_csv,
            output_tips_csv,
            image_dir,
            image_threshold=image_threshold,
            prediction_threshold=prediction_threshold,
        )

        if output_averaged_csv or output_averaged_tips_csv:
            self.output_results_to_df(
                [self.name+"-averaged"],
                gene_results.mean(axis=0, keepdims=True),
                output_averaged_csv,
                output_averaged_tips_csv,
                image_dir,
                image_threshold=image_threshold,
                prediction_threshold=prediction_threshold,
            )

        return result

    # @method
    # def extra_callbacks_off(self, **kwargs):
    #     from lightning.pytorch.callbacks import Callback
    #     import tracemalloc
    #     class MemoryLeakCallback(Callback):
    #         def on_train_start(self, trainer, pl_module):
    #             # Start tracing memory allocations at the beginning of the training
    #             tracemalloc.start()
    #             print("tracemalloc started")

    #         def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
    #             # Take a snapshot before the batch starts
    #             self.snapshot_before = tracemalloc.take_snapshot()

    #         def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
    #             # Take a snapshot after the batch ends
    #             snapshot_after = tracemalloc.take_snapshot()
                
    #             # Compare the snapshots
    #             stats = snapshot_after.compare_to(self.snapshot_before, 'lineno')
                
    #             # Log the top memory-consuming lines
    #             print(f"[Batch {trainer.global_step}] Memory differences:")
    #             for stat in stats[:20]:
    #                 print(stat)

    #             # Optionally, monitor peak memory usage
    #             current, peak = tracemalloc.get_traced_memory()
    #             print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
                
    #             # Clear traces if needed to prevent tracemalloc from consuming too much memory itself
    #             tracemalloc.clear_traces()

    #     return [MemoryLeakCallback()]
    

