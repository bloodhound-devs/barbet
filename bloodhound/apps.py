import psutil
import torch
import numpy as np
from pathlib import Path
from torch import nn
import lightning as L
from torchmetrics import Metric
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from corgi.seqtree import SeqTree, SeqDetail
from seqbank import SeqBank
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from torch.utils.data import DataLoader
from collections.abc import Iterable
from rich.console import Console
from torch.utils.data import Dataset
from Bio import SeqIO
from dataclasses import dataclass
from hierarchicalsoftmax.metrics import greedy_accuracy

import pandas as pd
import os
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities

from torchapp import Param, method, tool, TorchApp
from .modelsx import BloodhoundModel
from .gtdbtk import read_tophits, read_tigrfam, read_pfam
from .embedding import get_key
from .data import read_memmap, RANKS, gene_id_from_accession


console = Console()



@dataclass(kw_only=True)
class BloodhoundDataset(Dataset):
    accessions: list[str]
    array:np.memmap|np.ndarray
    gene_id_dict: dict[str, int]
    accession_to_array_index:dict[str,int]|None=None

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_index = self.accession_to_array_index[accession] if self.accession_to_array_index else idx
        embedding = torch.as_tensor(np.array(self.array[array_index,:], copy=False), dtype=torch.float16)
        # embedding = torch.tensor(self.array[array_index,:], dtype=torch.float16)
        gene_id = gene_id_from_accession(accession)
        return embedding, self.gene_id_dict[gene_id]


# @dataclass(kw_only=True)
# class BloodhoundTrainingDataset(BloodhoundDataset):
#     seqtree: SeqTree

#     def __getitem__(self, idx):
#         result = super()[idx]
#         accession = self.accessions[idx]
#         return *result, self.seqtree[accession].node_id


@dataclass(kw_only=True)
class BloodhoundTrainingDataset(Dataset):
    accessions: list[str]
    seqtree: SeqTree
    array:np.memmap|np.ndarray
    gene_id_dict: dict[str, int]
    accession_to_array_index:dict[str,int]|None=None

    def __post_init__(self):
        print('BloodhoundTrainingDataset', hex(self.array.ctypes.data))

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        array_index = self.accession_to_array_index[accession] if self.accession_to_array_index else idx
        # x = self.array[array_index,:]
        # x = np.array(self.array[array_index,:], copy=False)


        # x = self.array[array_index,:]
        # # xx = np.array(x, copy=True)
        # xxx = x.tolist()
        # embedding = torch.from_numpy(xxx)
        # # x = np.array(self.array[array_index,:], copy=False)
        # # embedding = torch.tensor(x, dtype=torch.float16)
        # del x
        # del xxx

        with torch.no_grad():
            data = np.array(self.array[array_index, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data

        # with torch.no_grad():
        #     data_numpy0 = self.array[array_index,:]
        #     data_numpy1 = np.array(data_numpy0, copy=False)
        #     data_tensor = torch.as_tensor(data_numpy1, dtype=torch.float16)
        #     embedding = data_tensor.detach().clone()
        #     del data_tensor
        #     del data_numpy1
        #     del data_numpy0

        # embedding = torch.zeros( (320,), dtype=torch.float16) # hack
        # embedding = torch.zeros( (320,), dtype=torch.float32) # hack

        # data = np.array(self.array[array_index,:], copy=False)
        # tensor = torch.as_tensor(data, dtype=torch.float16)
        # t = torch.from_numpy(data)
        # with torch.no_grad():
        #     array_index = 0
        #     x = self.array[array_index,:]
        #     data = np.array(x, copy=False)
        #     x = data.mean()
        #     # for i in range(320):
        #     #     embedding[i] = float(data[i].item())
        #     del data
        #     del x
        # del tensor
        # # embedding = torch.tensor(x, dtype=torch.float16)

        gene_id = gene_id_from_accession(accession)
        seq_detail = self.seqtree[accession]
        node_id = int(seq_detail.node_id)
        del seq_detail
        
        # return embedding, self.gene_id_dict[gene_id], self.seqtree[self.accessions[0]].node_id # hack
        return embedding, self.gene_id_dict[gene_id], node_id


@dataclass
class BloodhoundDataModule(L.LightningDataModule):
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
        validation_partition:int = 0,
    ):
        super().__init__()
        self.array = array
        self.accession_to_array_index = accession_to_array_index
        self.seqtree = seqtree
        self.gene_id_dict = gene_id_dict
        self.max_items = max_items
        self.batch_size = batch_size
        self.validation_partition = validation_partition
        self.num_workers = num_workers or min(os.cpu_count(), 8)

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

    def create_dataset(self, accessions:list[str]) -> BloodhoundTrainingDataset:
        return BloodhoundTrainingDataset(
            accessions=accessions, 
            seqtree=self.seqtree, 
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            gene_id_dict=self.gene_id_dict,
        )
    
    def train_dataloader(self):
        print('train dataloader', self.num_workers)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        print('val_dataloader', self.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Bloodhound(TorchApp):
    @method
    def setup(
        self,
        memmap:str=None,
        memmap_index:str=None,
        seqtree:str=None,
        in_memory:bool=False,
        prune:str="",
    ) -> None:
        if not seqtree:
            raise ValueError("seqtree is required")
        if not memmap:
            raise ValueError("memmap is required")
        if not memmap_index:
            raise ValueError("memmap_index is required")        

        print(f"Loading seqtree {seqtree}")
        self.seqtree = SeqTree.load(seqtree)

        # prune classification tree if requested
        # if prune:
        #     assert prune in RANKS[:-1]
        #     prune_rank_index = RANKS.index(prune) + 1
        #     for key, node_detail in self.seqtree.items():
        #         node = self.seqtree.node(key)
        #         ancestors = node.ancestors
        #         if len(ancestors) > prune_rank_index:
        #             new_node = node.ancestors[prune_rank_index]
        #             self.seqtree[key] = SeqDetail(new_node, node_detail.partition)
            
        #     # remove children of genus nodes
        #     for node in self.seqtree.classification_tree.pre_order_iter():
        #         self.readonly = False
        #         if len(node.ancestors) == prune_rank_index:
        #             node.children = []

        #     self.seqtree.set_indexes()
        #     RANKS = RANKS[:-1]# hack


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
        family_embedding_size:int=128,
    ) -> nn.Module:
        return BloodhoundModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            family_embedding_size=family_embedding_size,
            gene_family_count=len(self.gene_id_dict),
        )
    
    @method
    def input_count(self) -> int:
        return 2
            
    @method
    def loss_function(self):
        # def dummy_loss(prediction, target):
        #     return prediction[0,0] * 0.0
        #     # return prediction.mean() * 0.0
        # return dummy_loss # hack
        return HierarchicalSoftmaxLoss(root=self.classification_tree)
    
    @method    
    def metrics(self) -> list[tuple[str,Metric]]:
        # return [] # hack
        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={1+i:rank for i, rank in enumerate(RANKS)},
        )
                
        return [('rank_accuracy', rank_accuracy)]
        # return [
        #     (rank, GreedyAccuracyTorchMetric(root=self.classification_tree, max_depth=i+1, name=rank))
        #     for i, rank in enumerate(RANKS)
        # ]
    
    @method    
    def data(
        self,
        max_items:int=0,
        num_workers:int=0,
    ) -> Iterable|L.LightningDataModule:
        return BloodhoundDataModule(
            # seqbank=self.seqbank,
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            seqtree=self.seqtree,
            gene_id_dict=self.gene_id_dict,
            max_items=max_items,
            num_workers=num_workers,
        )
    
    @method
    def prediction_dataloader(
        self,
        module,
        gtdbtk_output:Path=None,
        embeddings:Path=None,
        fasta:Path=None,
        tophits:Path=None,
        tigrfam:Path=None,
        pfam:Path=None,
        batch_size:int = 64,
        num_workers: int = 0,
    ) -> Iterable:
        # Get Embedding model from dataloaders
        embedding = module.embedding        
        assert embedding is not None

        self.classification_tree = module.classification_tree
        assert self.classification_tree is not None

        # If gtdbtk output directory is given, then fetch the top hits from tigrfam
        def get_subfile(directory:Path, pattern:str) -> Path|None:
            return next(iter(directory.glob(pattern)), None)

        if gtdbtk_output is not None and gtdbtk_output.exists() and gtdbtk_output.is_dir():
            if not tophits:
                tophits = get_subfile(gtdbtk_output, '*_tigrfam_tophit.tsv')
            if not tigrfam:
                tophits = get_subfile(gtdbtk_output, '*_tigrfam.tsv')
            if not pfam:
                pfam = get_subfile(gtdbtk_output, '*_pfam.tsv')
            if not fasta:
                fasta = get_subfile(gtdbtk_output, '*_protein.faa')

        # Create dictionary from gene id to family id from the tophits or tigrfam or pfam
        if tophits:
            gene_family_dict = read_tophits(tophits)        
        elif tigrfam and tigrfam.exists():
            gene_family_dict = read_tigrfam(tigrfam)
        elif pfam:
            gene_family_dict = read_pfam(pfam)

        # Find genes where we need to create the embeddings
        accessions = list({f"{gene_id}/{family_id}" for gene_id, family_id in gene_family_dict.items()})
        genes_to_do = [accession.split("/")[0] for accession in accessions]
        count = len(accessions)

        dtype = "float16"
        if Path(embeddings).exists():
            array = read_memmap(embeddings, count, dtype=dtype)
        else:
            # Generate embeddings if necessary
            array = None
            print(f"Generating embeddings for {len(accessions)} protein sequences")
            assert fasta is not None
            fasta = Path(fasta)
            assert fasta.exists()
            for record in SeqIO.parse(fasta, "fasta"):
                if record.id in genes_to_do:
                    family_id = gene_family_dict[record.id]
                    print(record.id, family_id)
                    vector = embedding(record.seq)
                    if vector is not None and not torch.isnan(vector).any():                        
                        # Initialise array if necessary
                        if array is None:
                            size = len(vector)
                            shape = (count,size)
                            array = np.memmap(embeddings, dtype=dtype, mode='w+', shape=shape)

                        # save vector to array
                        index = genes_to_do.index(record.id)
                        array[index] = vector.cpu().detach().clone().numpy()

        dataset = BloodhoundDataset(
            accessions=accessions, 
            array=self.array,
            gene_id_dict=self.gene_id_dict,
        )

        self.items = accessions

        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=False)

    @method
    def output_results(
        self, 
        results, 
        output_csv: Path = Param(default=None, help="A path to output the results as a CSV."),
        output_gene_csv: Path = Param(default=None, help="A path to output the results for individual genes as a CSV."),
        output_tips_csv: Path = Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        image: Path = Param(default=None, help="A path to output the result as an image."),
        image_threshold:float = 0.005,
        prediction_threshold:float = Param(default=0.0, help="The threshold value for making hierarchical predictions."),
        seqtree:Path = None,
        output_correct:Path=None,
        **kwargs,
    ):
        assert self.classification_tree # This should be saved from the learner

        # Sum the scores which is equivalent of multiplying the probabilities assuming that they are independent
        results = results[0].sum(axis=0, keepdims=True)

        classification_probabilities = node_probabilities(results, root=self.classification_tree)
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list if not node.is_root]

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

        # Reorder columns
        results_df = results_df[["greedy_prediction", "probability" ] + category_names]

        # Output images
        if image:
            console.print(f"Writing inference probability renders to: {image}")
            image = Path(image)
            image_paths = [image]
            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        if seqtree is not None:
            seqtree = SeqTree.load(seqtree)
            prefix = get_key(self.accession, gene="")
            for key in seqtree.keys():
                if key.startswith(prefix):
                    break
            correct_node = seqtree.node(key)
            correct_ancestors = correct_node.ancestors + (correct_node,)
            prediction_node = predictions[0]
            prediction_ancestors = prediction_node.ancestors + (prediction_node,)
            for i, rank in enumerate(RANKS):
                prediction_rank = str(prediction_ancestors[i+1]).strip()
                correct_rank = str(correct_ancestors[i+1]).strip()
                rank_is_correct = (prediction_rank == correct_rank)
                if rank_is_correct:
                    console.print(f"[green]{rank} correctly predicted as {prediction_rank}")
                else:
                    console.print(f"[red]{rank} incorrectly predicted as {prediction_rank} instead of {correct_rank}")
                results_df[f"correct_{rank}"] = rank_is_correct

        if not (image or output_csv or output_tips_csv):
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

    @method
    def extra_callbacks_off(self, **kwargs):
        from lightning.pytorch.callbacks import Callback
        import tracemalloc
        class MemoryLeakCallback(Callback):
            def on_train_start(self, trainer, pl_module):
                # Start tracing memory allocations at the beginning of the training
                tracemalloc.start()
                print("tracemalloc started")

            def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
                # Take a snapshot before the batch starts
                self.snapshot_before = tracemalloc.take_snapshot()

            def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
                # Take a snapshot after the batch ends
                snapshot_after = tracemalloc.take_snapshot()
                
                # Compare the snapshots
                stats = snapshot_after.compare_to(self.snapshot_before, 'lineno')
                
                # Log the top memory-consuming lines
                print(f"[Batch {trainer.global_step}] Memory differences:")
                for stat in stats[:20]:
                    print(stat)

                # Optionally, monitor peak memory usage
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
                
                # Clear traces if needed to prevent tracemalloc from consuming too much memory itself
                tracemalloc.clear_traces()

        return [MemoryLeakCallback()]