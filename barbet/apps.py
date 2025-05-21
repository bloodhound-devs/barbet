import torch
import numpy as np
from pathlib import Path
from torch import nn
import lightning as L
from torchmetrics import Metric
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from hierarchicalsoftmax import TreeDict
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from torch.utils.data import DataLoader
from collections.abc import Iterable
from rich.console import Console
from enum import Enum

from collections import defaultdict
from rich.progress import track

import pandas as pd
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities

from torchapp import TorchApp, Param, method, main

from barbet.markers import extract_single_copy_markers
from .models import BarbetModel
from .data import read_memmap, RANKS, BarbetDataModule, BarbetPredictionDataset
from .embeddings.esm import ESMEmbedding

console = Console()


class ImageFormat(str, Enum):
    """ The image format to use for the output images. """
    NONE = ""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    PDF = "pdf"
    DOT = "dot"

    def __str__(self):
        return self.value

    def __bool__(self) -> bool:
        """ Returns True if the image format is not empty. """
        return self.value != ""


class Barbet(TorchApp):
    @method
    def setup(
        self,
        memmap:str=None,
        memmap_index:str=None,
        treedict:str=None,
        in_memory:bool=False,
        tip_alpha:float=None,
    ) -> None:
        if not treedict:
            raise ValueError("treedict is required")
        if not memmap:
            raise ValueError("memmap is required")
        if not memmap_index:
            raise ValueError("memmap_index is required")        

        print(f"Loading treedict {treedict}")
        individual_treedict = TreeDict.load(treedict)
        self.treedict = TreeDict(classification_tree=individual_treedict.classification_tree)

        # Sets the loss weighting for the tips
        if tip_alpha:
            for tip in self.treedict.classification_tree.leaves:
                tip.parent.alpha = tip_alpha

        print(f"Loading memmap")
        self.accession_to_array_index = defaultdict(list)
        with open(memmap_index) as f:
            for key_index, key in enumerate(f):
                key = key.strip()
                accession = key.strip().split("/")[0]

                if len(self.accession_to_array_index[accession]) == 0:
                    self.treedict[accession] = individual_treedict[key]

                self.accession_to_array_index[accession].append(key_index)
        count = key_index + 1
        self.array = read_memmap(memmap, count)

        # If there's enough memory, then read into RAM
        if in_memory:
            self.array = np.array(self.array)

        self.classification_tree = self.treedict.classification_tree
        assert self.classification_tree is not None

        # Get list of gene families
        family_ids = set()
        for accession in self.treedict:
            gene_id = accession.split("/")[-1]
            family_ids.add(gene_id)

        self.gene_id_dict = {family_id:index for index, family_id in enumerate(sorted(family_ids))}

    @method
    def model(
        self,
        features:int=768,
        intermediate_layers:int=2,
        growth_factor:float=2.0,
        attention_size:int=512,
    ) -> nn.Module:
        return BarbetModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            attention_size=attention_size,
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
        num_workers:int=4,
        validation_partition:int=0,
        batch_size:int = 4,
        test_partition:int=-1,
        seq_count:int=32,
        train_all:bool = False,
    ) -> Iterable|L.LightningDataModule:
        return BarbetDataModule(
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            treedict=self.treedict,
            gene_id_dict=self.gene_id_dict,
            max_items=max_items,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_partition=validation_partition,
            test_partition=test_partition,
            seq_count=seq_count,
            train_all=train_all,
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
            classification_tree=self.treedict.classification_tree,
            gene_id_dict=self.gene_id_dict,
        )
    
    @method
    def prediction_dataloader(
        self,
        module,
        genome_path:Path,
        output_dir:Path=Param("output", help="A path to the output directory."),
        cpus:int=Param(1, help="The number of CPUs to use to extract the single copy markers."),
        pfam_db:str=Param("https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/pfam/Pfam-A.hmm", help="The Pfam database to use."),
        tigr_db:str=Param("https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/tigrfam/tigrfam.hmm", help="The TIGRFAM database to use."),
        batch_size:int = Param(64, help="The batch size for the prediction dataloader."),
        num_workers: int = 0,
        repeats:int = Param(2, help="The minimum number of times to use each protein embedding in the prediction."),
        **kwargs,
    ) -> Iterable:        
        # Get hyperparameters from checkpoint        
        seq_count = module.hparams.get('seq_count', 32)
        self.classification_tree = module.hparams.classification_tree

        genomes = dict()
        genomes[genome_path.stem] = str(genome_path)

        # or use extract it from the barbet model
        domain = "bac120"
        # domain = "ar53" if len(module.hparams.gene_id_dict) == 53 else "bac120"

        ####################
        # Extract single copy marker genes
        ####################
        single_copy_marker_result = extract_single_copy_markers(
            genomes=genomes,
            out_dir=str(output_dir),
            cpus=cpus,
            force=True,
            pfam_db=self.process_location(pfam_db),
            tigr_db=self.process_location(tigr_db),
        )
        
        #######################
        # Create Embeddings
        #######################
        embeddings = []
        accessions = []

        fastas = single_copy_marker_result[genome_path.stem][domain]

        for fasta in track(fastas, description="[cyan]Embedding...  ", total=len(fastas)):
            # read the fasta file sequence remove the header
            fasta = Path(fasta)
            seq = fasta.read_text().split("\n")[1]
            vector = module.hparams.embedding_model(seq)
            if vector is not None and not torch.isnan(vector).any():
                vector = vector.cpu().detach().clone().numpy()
                embeddings.append(vector)

                gene_family_id = fasta.stem
                accession = f"{genome_path.stem}/{gene_family_id}"
                accessions.append(accession)

            del vector        

        embeddings = np.asarray(embeddings).astype(np.float16)

        self.prediction_dataset = BarbetPredictionDataset(
            array=embeddings, 
            accessions=accessions,
            seq_count=seq_count,
            repeats=repeats,
            seed=42,
        )
        dataloader = DataLoader(self.prediction_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return dataloader

    def node_to_str(self, node:SoftmaxNode) -> str:
        """ 
        Converts the node to a string
        """
        return str(node).split(",")[-1].strip()
    
    @method
    def output_results(
        self, 
        results, 
        genome_path:Path,
        threshold:float = Param(default=0.0, help="The threshold value for making hierarchical predictions."),
        image_format: ImageFormat = Param(default="", help="A path to output the results as images."),
        image_threshold:float = 0.005,
        **kwargs,
    ) -> pd.DataFrame:
        assert self.classification_tree

        # Average results across all stacks
        results = results.mean(axis=0, keepdims=True)

        classification_probabilities = node_probabilities(results, root=self.classification_tree)
        
        node_list = self.classification_tree.node_list_softmax
        category_names = [self.node_to_str(node) for node in node_list if not node.is_root]

        results_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)
        
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=threshold,
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

        results_df["name"] = [str(genome_path)]

        # Reorder columns
        results_df = results_df[["name", "greedy_prediction", "probability" ] + category_names]

        # Output images
        if image_format:
            console.print(f"Writing inference probability renders to: {output_dir}")
            output_dir = Path(output_dir)
            image_paths = [output_dir/f"{name}.{image_format}" for name in results_df["name"]]
            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        return results_df

    @main("load_checkpoint", "prediction_trainer", "prediction_dataloader", "output_results")
    def predict(
        self,
        input:list[Path]=Param(help="FASTA files or directories of FASTA files. Requires genome in an individual FASTA file."),
        output_csv: Path = Param(default=None, help="A path to output the results as a CSV."),
        output_dir:Path=Param("output", help="A path to the output directory."),
        greedy_only:bool = True,
        **kwargs,
    ):
        """ Make predictions with the model. """
        # Get list of files
        files = []
        if isinstance(input, (str, Path)):
            input = [input]
        assert len(input) > 0, "No input files provided."
        for path in input:
            if path.is_dir():
                for file in path.rglob("*.fa") + path.rglob("*.fasta") + path.rglob("*.fna"):
                    files.append(file)
            elif path.is_file():
                files.append(path)

        # Check if any files were found
        if len(files) == 0:
            raise ValueError(f"No files found in {input}. Please provide a directory or a list of files.")

        # Check if output directory exists
        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(exist_ok=True, parents=True)
            console.print(f"Writing results for {len(files)} genomes to: {output_csv}")

        # Load the model
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)

        # Make predictions for each file
        total_df = None
        for file_index, file in enumerate(files):
            prediction_dataloader = self.prediction_dataloader(module, file, **kwargs)
            results = trainer.predict(module, dataloaders=prediction_dataloader)
            results = torch.cat(results, dim=0)
            results_df = self.output_results(results, file, **kwargs)

            if greedy_only:
                results_df = results_df[["name", "greedy_prediction", "probability"]]

            if total_df is None:
                total_df = results_df
                if output_csv:
                    results_df.to_csv(output_csv, index=False)
            else:
                total_df = pd.concat([total_df, results_df], axis=0).reset_index()

                if output_csv:
                    results_df.to_csv(output_csv, mode='a', header=False, index=False)

        return total_df
