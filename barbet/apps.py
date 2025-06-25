from typing import TYPE_CHECKING
from pathlib import Path
from enum import Enum
from collections import defaultdict
from rich.console import Console
from rich.progress import track
from torchapp import TorchApp, Param, method, main, tool


if TYPE_CHECKING:
    from collections.abc import Iterable
    from torchmetrics import Metric
    from hierarchicalsoftmax import SoftmaxNode
    from torch import nn
    import lightning as L
    import pandas as pd


console = Console()


class ImageFormat(str, Enum):
    """The image format to use for the output images."""

    NONE = ""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    PDF = "pdf"
    DOT = "dot"

    def __str__(self):
        return self.value

    def __bool__(self) -> bool:
        """Returns True if the image format is not empty."""
        return self.value != ""


class Barbet(TorchApp):
    @method
    def setup(
        self,
        memmap: str = None,
        memmap_index: str = None,
        treedict: str = None,
        in_memory: bool = False,
        tip_alpha: float = None,
    ) -> None:
        if not treedict:
            raise ValueError("treedict is required")
        if not memmap:
            raise ValueError("memmap is required")
        if not memmap_index:
            raise ValueError("memmap_index is required")

        from hierarchicalsoftmax import TreeDict
        import numpy as np
        from barbet.data import read_memmap

        print(f"Loading treedict {treedict}")
        individual_treedict = TreeDict.load(treedict)
        self.treedict = TreeDict(
            classification_tree=individual_treedict.classification_tree
        )

        # Sets the loss weighting for the tips
        if tip_alpha:
            for tip in self.treedict.classification_tree.leaves:
                tip.parent.alpha = tip_alpha

        print("Loading memmap")
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

        self.gene_id_dict = {
            family_id: index for index, family_id in enumerate(sorted(family_ids))
        }

    @method
    def model(
        self,
        features: int = 768,
        intermediate_layers: int = 2,
        growth_factor: float = 2.0,
        attention_size: int = 512,
    ) -> "nn.Module":
        from barbet.models import BarbetModel

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
        from hierarchicalsoftmax import HierarchicalSoftmaxLoss

        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    @method
    def metrics(self) -> "list[tuple[str,Metric]]":
        from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
        from barbet.data import RANKS

        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree,
            ranks={1 + i: rank for i, rank in enumerate(RANKS)},
        )

        return [("rank_accuracy", rank_accuracy)]

    @method
    def data(
        self,
        max_items: int = 0,
        num_workers: int = 4,
        validation_partition: int = 0,
        batch_size: int = 4,
        test_partition: int = -1,
        seq_count: int = 32,
        train_all: bool = False,
    ) -> "Iterable|L.LightningDataModule":
        from barbet.data import BarbetDataModule

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
    def extra_hyperparameters(self, embedding_model: str = "") -> dict:
        """Extra hyperparameters to save with the module."""
        assert embedding_model, "Please provide an embedding model."
        from barbet.embeddings.esm import ESMEmbedding

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
        genome_path: Path,
        markers: list[Path], 
        batch_size: int = Param(
            64, help="The batch size for the prediction dataloader."
        ),
        num_workers: int = 4,
        repeats: int = Param(
            2,
            help="The minimum number of times to use each protein embedding in the prediction.",
        ),
        **kwargs,
    ) -> "Iterable":
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from barbet.data import BarbetPredictionDataset
       
        # Get hyperparameters from checkpoint
        seq_count = module.hparams.get("seq_count", 32)
        self.classification_tree = module.hparams.classification_tree

        # TODO extract domain from the barbet model
        domain = "bac120"
        # domain = "ar53" if len(module.hparams.gene_id_dict) == 53 else "bac120"

        #######################
        # Create Embeddings
        #######################
        embeddings = []
        accessions = []

        fastas = markers[domain]
        for fasta in track(
            fastas, description="[cyan]Embedding...  ", total=len(fastas)
        ):
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
        dataloader = DataLoader(
            self.prediction_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return dataloader

    def node_to_str(self, node: "SoftmaxNode") -> str:
        """
        Converts the node to a string
        """
        return str(node).split(",")[-1].strip()

    @method
    def output_results(
        self,
        results,
        names: list[str],
        threshold: float = Param(
            default=0.0, help="The threshold value for making hierarchical predictions."
        ),
        image_format: ImageFormat = Param(
            default="", help="A path to output the results as images."
        ),
        image_threshold: float = 0.005,
        probabilities: bool = Param(default=False, help="If True, include probabilities for all the nodes in the taxonomic tree."),
        **kwargs,
    ) -> "pd.DataFrame":
        import torch
        import pandas as pd
        from hierarchicalsoftmax.inference import (
            node_probabilities,
            greedy_predictions,
            render_probabilities,
        )
        from barbet.data import RANKS

        assert self.classification_tree
        assert self.classification_tree.layer_size == results.shape[-1]

        classification_probabilities = node_probabilities(
            results, root=self.classification_tree
        )

        node_list = self.classification_tree.node_list_softmax
        category_names = [
            self.node_to_str(node) for node in node_list if not node.is_root
        ]

        results_df = pd.DataFrame(
            classification_probabilities.numpy(), 
            columns=category_names
        )
        results_df["name"] = names
        results_df = results_df.groupby(["name"]).mean().reset_index()

        classification_probabilities = torch.as_tensor(
            results_df[category_names].to_numpy()
        )

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities,
            root=self.classification_tree,
            threshold=threshold,
        )

        output_columns = ["name"]
        for rank in RANKS:
            output_columns += [f"{rank}_prediction", f"{rank}_probability"]
            results_df[f"{rank}_prediction"] = ""
            results_df[f"{rank}_probability"] = 0.0

        for index, node in enumerate(predictions):
            lineage = node.ancestors[1:] + (node,)
            for rank, lineage_node in zip(RANKS, lineage):
                node_name = self.node_to_str(lineage_node)
                results_df.loc[index, f"{rank}_prediction"] = node_name
                results_df.loc[index, f"{rank}_probability"] = results_df.loc[index,node_name]

        # Output images
        if image_format:
            console.print(
                f"Writing inference probability renders to: {self.output_dir}"
            )
            output_dir = Path(self.output_dir)
            image_paths = [output_dir / f"{name}.{image_format}" for name in results_df["name"]]
            render_probabilities(
                root=self.classification_tree,
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        if probabilities:
            output_columns += category_names
        results_df = results_df[output_columns]

        return results_df

    @main(
        "load_checkpoint",
        "prediction_trainer",
        "prediction_dataloader",
        "output_results",
    )
    def predict(
        self,
        input: list[Path] = Param(
            default=...,
            help="FASTA files or directories of FASTA files. Requires genome in an individual FASTA file."
        ),
        output_dir: Path = Param("output", help="A path to the output directory."),
        output_csv: Path = Param(
            default=None, help="A path to output the results as a CSV."
        ),
        cpus: int = Param(
            1, help="The number of CPUs to use to extract the single copy markers."
        ),
        pfam_db: str = Param(
            "https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/pfam/Pfam-A.hmm",
            help="The Pfam database to use.",
        ),
        tigr_db: str = Param(
            "https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/tigrfam/tigrfam.hmm",
            help="The TIGRFAM database to use.",
        ),
        **kwargs,
    ):
        """Barbet is a tool for assigning taxonomic labels to genomes using Machine Learning."""
        import torch
        import pandas as pd
        from itertools import chain
        from barbet.markers import extract_markers_genes

        # Get list of files
        files = []
        if isinstance(input, (str, Path)):
            input = [input]
        assert len(input) > 0, "No input files provided."
        for path in input:
            if path.is_dir():
                for file in chain(
                    path.rglob("*.fa"),
                    path.rglob("*.fasta"),
                    path.rglob("*.fna"),
                    path.rglob("*.fa.gz"),
                    path.rglob("*.fasta.gz"),
                    path.rglob("*.fna.gz"),
                ):
                    files.append(file)
            elif path.is_file():
                files.append(path)

        # Check if any files were found
        if len(files) == 0:
            raise ValueError(
                f"No files found in {input}. Please provide a directory or a list of files."
            )

        # Check if output directory exists
        self.output_dir = Path(output_dir)
        output_csv = output_csv or self.output_dir / "barbet-predictions.csv"
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        console.print(
            f"Writing results for {len(files)} genome{'s' if len(files) > 1 else ''} to '{output_csv}'"
        )

        ####################
        # Extract single copy marker genes
        ####################
        markers_gene_map = extract_markers_genes(
            genomes={file.stem: str(file) for file in files},
            out_dir=str(self.output_dir),
            cpus=cpus,
            force=True,
            pfam_db=self.process_location(pfam_db),
            tigr_db=self.process_location(tigr_db),
        )

        # Load the model
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)

        # Make predictions for each file
        total_df = None
        for genome_path, maker_genes in markers_gene_map.items():
            genome_path = Path(genome_path)
            prediction_dataloader = self.prediction_dataloader(module, genome_path, maker_genes, **kwargs)
            results = trainer.predict(module, dataloaders=prediction_dataloader)
            results = torch.cat(results, dim=0)
            results_df = self.output_results(results, genome_path.name, **kwargs)

            if total_df is None:
                total_df = results_df
                if output_csv:
                    results_df.to_csv(output_csv, index=False)
            else:
                total_df = pd.concat([total_df, results_df], axis=0).reset_index(drop=True)

                if output_csv:
                    results_df.to_csv(output_csv, mode="a", header=False, index=False)

        console.print(total_df[["name", "family_prediction", "genus_prediction", "species_prediction"]])
        console.print(f"Saved to: '{output_csv}'")
        return total_df

    @tool(
        "load_checkpoint",
        "prediction_trainer",
        "prediction_dataloader_memmap",
        "output_results",
    )
    def predict_memmap(
        self,
        output_csv: Path = Param(
            default=None, help="A path to output the results as a CSV."
        ),
        **kwargs,
    ):
        """Barbet is a tool for assigning taxonomic labels to genomes using Machine Learning."""
        import torch

        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader_memmap(module, **kwargs)

        results_list = trainer.predict(module, dataloaders=prediction_dataloader)

        if not results_list:
            return None

        results = torch.cat(results_list, dim=0)
        names = [stack.species for stack in self.prediction_dataset.stacks]
        results_df = self.output_results(results, names, **kwargs)

        # Add gold values if possible
        if hasattr(self, 'gold_values'):
            results_df['gold_value'] = results_df['name'].map(self.gold_values)
    
        console.print(f"writing to '{output_csv}'")
        results_df.to_csv(output_csv, index=False)

        return results_df
    
    @method
    def prediction_dataloader_memmap(
        self,
        module,
        memmap:Path = Param(None, help="A path to the memmap file containing the protein embeddings."),
        memmap_index:Path = Param(None, help="A path to the memmap index file containing the accessions."),
        batch_size: int = Param(
            64, help="The batch size for the prediction dataloader."
        ),
        num_workers: int = 4,
        repeats: int = Param(
            2,
            help="The minimum number of times to use each protein embedding in the prediction.",
        ),
        treedict:Path=Param(None, help="A path to the treedict file to use for filtering species. (Must be used with `treedict_partition`)"),
        treedict_partition:int= Param(None, help="The partition of the treedict to use for filtering species. (Must be used with `treedict`.)"),
        **kwargs,
    ) -> "Iterable":
        from barbet.data import read_memmap
        from torch.utils.data import DataLoader
        from barbet.data import BarbetPredictionDataset
        
        assert memmap is not None, "Please provide a path to the memmap file."
        assert memmap.exists(), f"Memmap file does not exist: {memmap}"
        assert memmap_index is not None, "Please provide a path to the memmap index file."
        assert memmap_index.exists(), f"Memmap index file does not exist: {memmap_index}"

        # Read the memmap array index
        console.print(f"Reading memmap array index '{memmap_index}'")
        accessions = memmap_index.read_text().strip().split("\n")
        count = len(accessions)
        console.print(f"Found {count} accessions")

        # Load the memmap array itself
        console.print(f"Loading memmap array '{memmap}'")
        array = read_memmap(memmap, count)

        # Get hyperparameters from checkpoint
        domain = "bac120" if module.model.classifier.out_features > 100_000 else "ar53"
        seq_count = module.hparams.get("seq_count", 32 if domain == "bac120" else 8)
        self.classification_tree = module.hparams.classification_tree

        # If treedict is provided, then we filter the accessions to only those that are in the treedict
        species_filter = None
        if treedict is None and treedict_partition is not None:
            print("If you provide a `treedict_partition` then you must also provide a `treedict`")
        if treedict is not None and treedict_partition is None:
            print("If you provide a `treedict` then you must also provide a `treedict_partition`")
        if treedict is not None and treedict_partition is not None:
            from hierarchicalsoftmax import TreeDict

            species_filter = set()
            console.print(f"Creating filter using partition {treedict_partition} from TreeDict '{treedict}'")
            treedict = TreeDict.load(treedict)
            self.gold_values = dict()
            for accession, details in treedict.items():
                partition = details.partition
                if partition == treedict_partition:
                    organism_name = accession.split("/")[0]
                    species_filter.add(organism_name)
                    self.gold_values[organism_name] = treedict.node(accession).name.strip()
            console.print(f"Filtering for {len(species_filter)} species")

        self.prediction_dataset = BarbetPredictionDataset(
            array=array,
            accessions=accessions,
            seq_count=seq_count,
            repeats=repeats,
            species_filter=species_filter,
            seed=42,
        )
        dataloader = DataLoader(
            self.prediction_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return dataloader
